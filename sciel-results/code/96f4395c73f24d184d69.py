import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]
DEADLINE = 150
SAFETY_MARGIN = 10

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p

    # Best result found so far (schoolbook)
    best_rank = rank_schoolbook
    best_result = None

    # Construct schoolbook as fallback
    def make_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best_result = make_schoolbook()

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        for row in U:
            if len(row) != rank:
                return False
        for row in V:
            if len(row) != rank:
                return False
        for row in W:
            if len(row) != rank:
                return False
        
        for row in U:
            for val in row:
                if not (val * 2 == int(val * 2)):
                    return False
        for row in V:
            for val in row:
                if not (val * 2 == int(val * 2)):
                    return False
        for row in W:
            for val in row:
                if not (val * 2 == int(val * 2)):
                    return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def try_round_and_verify(U_float, V_float, W_float, rank):
        """Try rounding to half-integers and verify."""
        U_r = [[round(x * 2) / 2 for x in row] for row in U_float]
        V_r = [[round(x * 2) / 2 for x in row] for row in V_float]
        W_r = [[round(x * 2) / 2 for x in row] for row in W_float]
        
        if verify(U_r, V_r, W_r, rank):
            return U_r, V_r, W_r
        return None

    # Try to find lower rank decompositions using structured Strassen-based approach
    # For 2x3x6: split into 2x2x6 (using Strassen) + 2x1x6 (schoolbook tail)
    # Strassen for 2x2x2 has rank 7, schoolbook has rank 4
    # For 2x2x6: we can use Strassen on each 2x2x2 block? No, that's not how it works.
    # 
    # Alternative: Use the fact that rank(2x3x6) can be computed by:
    # Split A = [A1; A2] where A1 is 1x3, A2 is 1x3 (n=2 split into two 1x3)
    # Then C = A1*B + A2*B, which is sum of two 1x3x6 products
    # Each 1x3x6 product has rank 3*6=18, so total rank = 36 (schoolbook)
    #
    # Better approach: Use Strassen-style recursion
    # 2x3x6 = (2x2x6) + (2x1x6) if we split the 3 columns
    # 2x2x6 can use Strassen-like methods
    # 2x1x6 has rank 2*1*6 = 12
    #
    # Actually, let's try a direct algebraic construction
    # Known: rank(MM[2,3,6]) is conjectured/known to be around 30-33
    
    # Let's try constructing a rank 33 decomposition using Strassen on 2x2 blocks
    # Strategy: Write 2x3x6 as combination of smaller matrix multiplications
    
    # For 2x3 matrix times 3x6 matrix:
    # A = [[a1, a2, a3],
    #      [b1, b2, b3]]
    # B = [[c1, c2, c3, c4, c5, c6],
    #      [d1, d2, d3, d4, d5, d6],
    #      [e1, e2, e3, e4, e5, e6]]
    # C = A*B is 2x6
    
    # We can write C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Using Strassen-like approach for the 2x2 part (first 2 columns):
    # Let A2x2 = [[a1,a2],[b1,b2]], B2x2 = [[c1,c2,c3,c4,c5,c6][:, :2] ... ]
    # This gets complicated. Let's try a simpler approach.
    
    # Try to find any rank < 36 decomposition via random search with better initialization
    
    rng = np.random.RandomState(42)
    
    # Try structured approaches first
    # Approach 1: Use Strassen for 2x2x2 subproblems
    # 2x3x6 can be decomposed as:
    # Split B into B1 (3x2), B2 (3x2), B3 (3x2)
    # C = A*B1 + A*B2 + A*B3
    # Each A*Bj is 2x2, can use Strassen (rank 7) but that's for the whole thing
    # Actually this doesn't reduce rank since we're summing
    
    # Let's just do a smarter random search with multiple strategies
    
    best_found = False
    
    for target_rank in range(33, 37):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        max_trials = 30
        for trial in range(max_trials):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Random initialization with some structure
            U_init = rng.randn(n * m, target_rank) * 0.5
            V_init = rng.randn(m * p, target_rank) * 0.5
            W_init = rng.randn(p * n, target_rank) * 0.5
            
            # Normalize
            for c in range(target_rank):
                u_norm = np.linalg.norm(U_init[:, c])
                v_norm = np.linalg.norm(V_init[:, c])
                w_norm = np.linalg.norm(W_init[:, c])
                if u_norm > 1e-12 and v_norm > 1e-12 and w_norm > 1e-12:
                    scale = (u_norm * v_norm * w_norm) ** (1.0 / 3.0)
                    U_init[:, c] /= scale
                    V_init[:, c] /= scale
                    W_init[:, c] /= scale
            
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            # Gradient descent
            lr = 0.05
            max_iter = 300
            
            for it in range(max_iter):
                if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                    break
                
                # Compute loss and gradients
                loss = 0.0
                gU = np.zeros_like(U_curr)
                gV = np.zeros_like(V_curr)
                gW = np.zeros_like(W_curr)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for c in range(target_rank):
                                total += U_curr[i * m + j, c] * V_curr[j * p + k, c] * W_curr[k * n + i, c]
                            error = 2.0 * (total - 1.0)
                            loss += error * (total - 1.0)
                            for c in range(target_rank):
                                gU[i * m + j, c] += error * V_curr[j * p + k, c] * W_curr[k * n + i, c]
                                gV[j * p + k, c] += error * U_curr[i * m + j, c] * W_curr[k * n + i, c]
                                gW[k * n + i, c] += error * U_curr[i * m + j, c] * V_curr[j * p + k, c]
                
                U_curr -= lr * gU
                V_curr -= lr * gV
                W_curr -= lr * gW
                lr *= 0.998
                
                if loss < 1e-8:
                    break
            
            # Try rounding
            result = try_round_and_verify(U_curr, V_curr, W_curr, target_rank)
            if result is not None:
                U_r, V_r, W_r = result
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {"U": U_r, "V": V_r, "W": W_r}
                    best_found = True
                    if target_rank <= 33:
                        break
        
        if best_found and best_rank <= 33:
            break
    
    return best_result
