"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Build rank-42 Strassen decomposition
    rank42 = 42
    U42 = np.zeros((n * m, rank42))
    V42 = np.zeros((m * p, rank42))
    W42 = np.zeros((p * n, rank42))
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        """Add a 2x2x2 Strassen block"""
        # Product 0: (A00+A11)*(B00+B11)
        U42[(i_start+0)*m + (j_start+0), offset+0] += 1
        U42[(i_start+1)*m + (j_start+1), offset+0] += 1
        V42[(j_start+0)*p + (k_start+0), offset+0] += 1
        V42[(j_start+1)*p + (k_start+1), offset+0] += 1
        W42[(k_start+0)*n + (i_start+0), offset+0] += 1
        W42[(k_start+1)*n + (i_start+1), offset+0] += 1
        
        # Product 1: (A10+A11)*B00
        U42[(i_start+1)*m + (j_start+0), offset+1] += 1
        U42[(i_start+1)*m + (j_start+1), offset+1] += 1
        V42[(j_start+0)*p + (k_start+0), offset+1] += 1
        W42[(k_start+0)*n + (i_start+1), offset+1] += 1
        W42[(k_start+1)*n + (i_start+1), offset+1] -= 1
        
        # Product 2: A00*(B01-B11)
        U42[(i_start+0)*m + (j_start+0), offset+2] += 1
        V42[(j_start+0)*p + (k_start+1), offset+2] += 1
        V42[(j_start+1)*p + (k_start+1), offset+2] -= 1
        W42[(k_start+1)*n + (i_start+0), offset+2] += 1
        W42[(k_start+1)*n + (i_start+1), offset+2] += 1
        
        # Product 3: A11*(B10-B00)
        U42[(i_start+1)*m + (j_start+1), offset+3] += 1
        V42[(j_start+1)*p + (k_start+0), offset+3] += 1
        V42[(j_start+0)*p + (k_start+0), offset+3] -= 1
        W42[(k_start+0)*n + (i_start+1), offset+3] += 1
        W42[(k_start+0)*n + (i_start+0), offset+3] -= 1
        
        # Product 4: (A00+A01)*B11
        U42[(i_start+0)*m + (j_start+0), offset+4] += 1
        U42[(i_start+0)*m + (j_start+1), offset+4] += 1
        V42[(j_start+1)*p + (k_start+1), offset+4] += 1
        W42[(k_start+1)*n + (i_start+0), offset+4] -= 1
        W42[(k_start+1)*n + (i_start+1), offset+4] += 1
        
        # Product 5: (A10-A00)*(B00+B01)
        U42[(i_start+1)*m + (j_start+0), offset+5] += 1
        U42[(i_start+0)*m + (j_start+0), offset+5] -= 1
        V42[(j_start+0)*p + (k_start+0), offset+5] += 1
        V42[(j_start+0)*p + (k_start+1), offset+5] += 1
        W42[(k_start+0)*n + (i_start+1), offset+5] += 1
        W42[(k_start+1)*n + (i_start+1), offset+5] += 1
        
        # Product 6: (A01-A11)*(B10+B11)
        U42[(i_start+0)*m + (j_start+1), offset+6] += 1
        U42[(i_start+1)*m + (j_start+1), offset+6] -= 1
        V42[(j_start+1)*p + (k_start+0), offset+6] += 1
        V42[(j_start+1)*p + (k_start+1), offset+6] += 1
        W42[(k_start+0)*n + (i_start+0), offset+6] += 1
        W42[(k_start+1)*n + (i_start+0), offset+6] += 1
    
    # Three blocks for the 6 columns of B (2 columns each)
    # Block 1: cols 0-1 of B
    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 2, 0, 7)
    # Block 2: cols 2-3 of B
    add_strassen_block(0, 0, 2, 14)
    add_strassen_block(0, 2, 2, 21)
    # Block 3: cols 4-5 of B
    add_strassen_block(0, 0, 4, 28)
    add_strassen_block(0, 2, 4, 35)
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U42, V42, W42, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # For each factor r, check if it can be expressed as a combination of others
    # This is equivalent to checking if the tensor equation holds without factor r
    
    def try_remove_factors(U, V, W, factors_to_remove, n, m, p):
        rank = U.shape[1]
        keep = [i for i in range(rank) if i not in factors_to_remove]
        if len(keep) == 0:
            return None
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        new_rank = len(keep)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(new_rank):
                        total += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return None
        return (U_new, V_new, W_new)
    
    # Try removing single factors
    best_result = None
    for r in range(rank42):
        if time.time() > deadline:
            break
        result = try_remove_factors(U42, V42, W42, [r], n, m, p)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try removing pairs of factors
    if time.time() < deadline - 30:
        for r1 in range(rank42):
            for r2 in range(r1 + 1, rank42):
                if time.time() > deadline:
                    break
                result = try_remove_factors(U42, V42, W42, [r1, r2], n, m, p)
                if result is not None:
                    U, V, W = result
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no reduction found, return the rank-42 solution
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
# EVOLVE-BLOCK-END
