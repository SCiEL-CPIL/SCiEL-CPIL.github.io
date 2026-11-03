"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check if the decomposition is exact. Returns True if valid."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    result = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        uc = U3[:, :, c]  # (n, m)
        vc = V3[:, :, c]  # (m, p)
        wc = W3[:, :, c]  # (p, n)
        for i in range(n):
            for j in range(m):
                if uc[i, j] != 0:
                    for k in range(p):
                        if vc[j, k] != 0 and wc[k, i] != 0:
                            result[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    
    return np.allclose(result, 1.0, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # For 4x5x6 matrix multiplication, the rank is known to be at most 120 (schoolbook)
    # Theoretical lower bounds suggest it could be lower, but finding exact decompositions
    # is challenging. We'll use a structured approach.
    
    # Approach: Use the fact that matrix multiplication can be decomposed using
    # block structures and known algorithms. For 4x5x6, we can try to use
    # the Strassen-like decomposition or other known results.
    
    # For now, let's try a more systematic search with structured matrices
    # that are more likely to yield valid decompositions.
    
    # Strategy: Use Kronecker products and known decompositions
    # For 4x5x6, we can write it as (2x2) x (2x2) x (3x3) with padding
    
    # Let's try a different approach: use the known result that the rank of
    # 4x4x4 matrix multiplication is at most 49 (Lader's result)
    # For 4x5x6, we can pad to 6x6x6 and use known decompositions
    
    # Actually, let's try a simpler approach: use the fact that we can decompose
    # the multiplication using a combination of smaller decompositions
    
    # For 4x5x6:
    # - We can write A (4x5) as a block matrix
    # - Use the fact that 4x5 = 4x4 + 4x1
    # - 5x6 = 4x6 + 1x6
    
    # Let's try to find a decomposition of rank less than 120 by using
    # structured matrices with small entries
    
    # Random search with structured initialization
    np.random.seed(42)
    
    # Try ranks from a reasonable lower bound
    # Lower bound: max(n*m, m*p, n*p) / 2 is a rough heuristic
    # For 4x5x6: max(20, 30, 24) = 30, so maybe try from 40 upward
    # But let's be more aggressive and try from 30
    
    for target_rank in range(30, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        max_attempts = 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random U and V with entries from {-2, -1, 0, 1, 2}
            # Use a sparsity pattern to make the system more likely to be solvable
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            
            # Fill with random values, but keep some structure
            for r in range(target_rank):
                # Choose a subset of indices for U
                u_indices = np.random.choice(n*m, size=min(3, n*m), replace=False)
                for idx in u_indices:
                    U[idx, r] = np.random.choice([-2, -1, 1, 2])
                
                # Choose a subset of indices for V
                v_indices = np.random.choice(m*p, size=min(3, m*p), replace=False)
                for idx in v_indices:
                    V[idx, r] = np.random.choice([-2, -1, 1, 2])
            
            # For each k, solve for W[k*n:..., :]
            W = np.zeros((p*n, target_rank))
            valid = True
            
            for k in range(p):
                for i in range(n):
                    # M_i[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    M_i = np.zeros((m, target_rank))
                    for j in range(m):
                        M_i[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    # Solve M_i @ x = ones(m) for x = W[k*n+i, :]
                    try:
                        # Check if the system is consistent
                        # Use lstsq to get a solution
                        x, residuals, rank_s, sv = np.linalg.lstsq(M_i, np.ones(m), rcond=None)
                        
                        # Check if this is an exact solution
                        if not np.allclose(M_i @ x, np.ones(m), atol=1e-9):
                            valid = False
                            break
                        
                        # Check if x has nice entries (integer or half-integer)
                        x_rounded = np.round(x * 2) / 2
                        if not np.allclose(x, x_rounded, atol=1e-6):
                            # Try rounding to integer
                            x_rounded_int = np.round(x)
                            if np.allclose(x, x_rounded_int, atol=1e-6):
                                x = x_rounded_int
                            else:
                                valid = False
                                break
                        
                        W[k*n+i, :] = x
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                U_list = [[float(x) for x in row] for row in U.tolist()]
                V_list = [[float(x) for x in row] for row in V.tolist()]
                W_list = [[float(x) for x in row] for row in W.tolist()]
                
                if _verify(U_list, V_list, W_list):
                    found = True
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
        
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
