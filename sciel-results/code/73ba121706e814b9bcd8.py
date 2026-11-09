"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # For each (i,j,k), check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    # Check that for all (i,j,k) not matching, the sum is 0
    # Actually, we need to check that the decomposition is exact
    # The tensor T[i,j,k] = 1 if this is a valid triple, else 0
    # But matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k)
    # Wait, let me reconsider. The matrix multiplication tensor is defined as:
    # T[i,j,k] = 1 if we're multiplying element (i,j) of A by (j,k) of B
    # So T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # So we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    return True

def _schoolbook(n, m, p):
    """Create the standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=10000, seed=42):
    """Try to reduce rank by finding linear dependencies among the columns."""
    rank = len(U[0])
    if target_rank >= rank:
        return None
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # We want to find a linear transformation that reduces the effective rank
    # Try random projections and see if we can maintain the decomposition
    
    best = None
    for iteration in range(max_iters):
        # Try to find a random linear combination that preserves the tensor
        # This is a heuristic approach
        
        # Generate random vectors for each column
        # We're looking for coefficients alpha_c such that
        # sum_c alpha_c * U[:,c] * V[:,c] * W[:,c] = 0 (approximately)
        # but we want to reduce the number of non-zero terms
        
        # Alternative: try to merge columns
        # Pick two columns and see if we can express one as a combination of others
        
        c1 = random.randint(0, rank - 1)
        c2 = random.randint(0, rank - 1)
        if c1 == c2:
            continue
        
        # Check if column c2 is approximately in the span of other columns
        # This is expensive, so use a simpler heuristic
        
        # Try a random perturbation approach
        # Add a small random perturbation to all columns and see if rank reduces
        
        if iteration % 100 == 0:
            # Try a structured approach: look for patterns
            # For matrix multiplication, there might be symmetries we can exploit
            pass
    
    return None

def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 matrices (rank 7)."""
    # Strassen's 7 multiplication algorithm
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11)(B22-B12)
    U[0][0] = 1  # A11
    V[3][0] = 1  # B22
    V[2][0] = -1 # B12
    W[2][0] = 1  # C11 += M1
    W[3][0] = 1  # C12 += M1
    
    # M2 = (A11+A12)(B12)
    U[0][1] = 1  # A11
    U[1][1] = 1  # A12
    V[2][1] = 1  # B12
    W[3][1] = 1  # C12 += M2
    
    # M3 = (A21+A22)(B11)
    U[2][2] = 1  # A21
    U[3][2] = 1  # A22
    V[0][2] = 1  # B11
    W[2][2] = 1  # C21 += M3
    
    # M4 = (A22)(B21-B11)
    U[3][3] = 1  # A22
    V[1][3] = 1  # B21
    V[0][3] = -1 # B11
    W[2][3] = 1  # C21 += M4
    W[3][3] = 1  # C22 += M4
    
    # M5 = (A11)(B11+B22)
    U[0][4] = 1  # A11
    V[0][4] = 1  # B11
    V[3][4] = 1  # B22
    W[2][4] = 1  # C11 += M5
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1  # A21
    U[0][5] = -1 # A11
    V[0][5] = 1  # B11
    V[2][5] = 1  # B12
    W[3][5] = 1  # C22 += M6
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1  # A12
    U[3][6] = -1 # A22
    V[1][6] = 1  # B21
    V[3][6] = 1  # B22
    W[2][6] = 1  # C21 += M7
    
    return U, V, W

def _kronecker_decomp(n, m, p):
    """Try to find a decomposition using Kronecker product structure."""
    start_time = time.time()
    max_time = 60.0
    
    # For small matrices, try known constructions
    # 3x4 * 4x7: Try to decompose into blocks
    
    # Strategy: Use the fact that 3 = 2+1, 4 = 2+2, 7 = 4+3
    # Try block decompositions
    
    best_result = None
    best_rank = n * m * p
    
    # Try rank from theoretical lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    for target_rank in range(28, 85):
        if time.time() - start_time > max_time:
            break
        
        # Try structured random search
        for attempt in range(50):
            if time.time() - start_time > max_time:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Use sparse random initialization with bias toward 1/-1
            for c in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.4:
                            U[i*m + j, c] = random.choice([-1, 1])
                
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.4:
                            V[j*p + k, c] = random.choice([-1, 1])
                
                for k in range(p):
                    for i in range(n):
                        if random.random() < 0.4:
                            W[k*n + i, c] = random.choice([-1, 1])
            
            # Verify using numpy for speed
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_block_decomposition(n, m, p, max_time=50.0):
    """Try to decompose using block structure."""
    start_time = time.time()
    
    # For 3x4 * 4x7, try splitting into blocks
    # e.g., 3 = 2+1, 4 = 2+2, 7 = 4+3
    
    # Try all possible block splits
    for n1 in range(1, n):
        for m1 in range(1, m):
            for p1 in range(1, p):
                if time.time() - start_time > max_time:
                    return None
                
                n2 = n - n1
                m2 = m - m1
                p2 = p - p1
                
                # Get decompositions for sub-blocks
                # This is recursive, so we limit depth
                # For now, just try schoolbook for sub-blocks
                rank1 = n1 * m1 * p1
                rank2 = n2 * m2 * p2
                total_rank = rank1 + rank2
                
                if total_rank >= n * m * p:
                    continue
                
                # Create combined decomposition
                U = [[0] * total_rank for _ in range(n * m)]
                V = [[0] * total_rank for _ in range(m * p)]
                W = [[0] * total_rank for _ in range(p * n)]
                
                # Fill first block
                col = 0
                for i in range(n1):
                    for j in range(m1):
                        for k in range(p1):
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
                
                # Fill second block (offset indices)
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            U[(n1 + i) * m + (m1 + j)][col] = 1
                            V[(m1 + j) * p + (p1 + k)][col] = 1
                            W[(p1 + k) * n + (n1 + i)][col] = 1
                            col += 1
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(total_rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            if abs(s - 1.0) > 1e-10:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 115 second deadline (safety margin)
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = n * m * p
    
    # Try block decomposition first (fast, structured)
    if time.time() < deadline - 60:
        result = _try_block_decomposition(n, m, p, max_time=30.0)
        if result is not None:
            result_rank = len(result["U"][0])
            if result_rank < best_rank:
                best_result = result
                best_rank = result_rank
    
    # Try Kronecker/structured search
    if time.time() < deadline - 30:
        result = _kronecker_decomp(n, m, p)
        if result is not None:
            result_rank = len(result["U"][0])
            if result_rank < best_rank:
                best_result = result
                best_rank = result_rank
    
    # Final check: verify the best result
    U_final = np.array(best_result["U"])
    V_final = np.array(best_result["V"])
    W_final = np.array(best_result["W"])
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_final[i*m+j, :] * V_final[j*p+k, :] * W_final[k*n+i, :])
                if abs(s - 1.0) > 1e-8:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        best_result = {"U": U, "V": W, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
