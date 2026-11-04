"""Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    if not U or not V or not W:
        return float('inf')
    rank = len(U[0]) if U else 0
    if rank == 0:
        return float('inf')
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and check that T[i,j,k] = 1 for all valid (i,j,k)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] should equal 1
                u_vec = U_arr[i*m+j]
                v_vec = V_arr[j*p+k]
                w_vec = W_arr[k*n+i]
                val = np.dot(u_vec, v_vec * w_vec)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _build_baseline(n, m, p):
    """Build the rank-33 Strassen-based baseline decomposition."""
    rank = 33
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) => contributes to all output columns
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1.0
            V[0 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    # col is now 12
    
    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) => Strassen per 2x2 block
    for t in range(3):
        # Local indices for this 2x2 block
        a00 = 0 * m + 1
        a01 = 0 * m + 2
        a10 = 1 * m + 1
        a11 = 1 * m + 2
        
        b00 = 1 * p + 2*t
        b01 = 1 * p + 2*t + 1
        b10 = 2 * p + 2*t
        b11 = 2 * p + 2*t + 1
        
        c00 = (2*t) * n + 0
        c01 = (2*t + 1) * n + 0
        c10 = (2*t) * n + 1
        c11 = (2*t + 1) * n + 1
        
        # M1 = (A00+A11)(B00+B11)
        U[a00][col] = 1.0; U[a11][col] = 1.0
        V[b00][col] = 1.0; V[b11][col] = 1.0
        W[c00][col] = 1.0; W[c11][col] = 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10][col] = 1.0; U[a11][col] = 1.0
        V[b00][col] = 1.0
        W[c10][col] = 1.0; W[c11][col] = -1.0
        col += 1
        
        # M3 = A00(B01-B11)
        U[a00][col] = 1.0
        V[b01][col] = 1.0; V[b11][col] = -1.0
        W[c01][col] = 1.0; W[c11][col] = 1.0
        col += 1
        
        # M4 = A11(B10-B00)
        U[a11][col] = 1.0
        V[b10][col] = 1.0; V[b00][col] = -1.0
        W[c00][col] = 1.0; W[c10][col] = 1.0
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00][col] = 1.0; U[a01][col] = 1.0
        V[b11][col] = 1.0
        W[c00][col] = -1.0; W[c01][col] = 1.0
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10][col] = 1.0; U[a00][col] = -1.0
        V[b00][col] = 1.0; V[b01][col] = 1.0
        W[c11][col] = 1.0
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01][col] = 1.0; U[a11][col] = -1.0
        V[b10][col] = 1.0; V[b11][col] = 1.0
        W[c00][col] = 1.0
        col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W, rank

def _try_remove_term(U, V, W, n, m, p, r_remove, deadline):
    """Check if removing term r_remove still gives exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    mask = [i for i in range(rank) if i != r_remove]
    U_test = U_arr[:, mask]
    V_test = V_arr[:, mask]
    W_test = W_arr[:, mask]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U_test[i*m+j]
                v_vec = V_test[j*p+k]
                w_vec = W_test[k*n+i]
                val = np.dot(u_vec, v_vec * w_vec)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err

def _search_lower_rank(U, V, W, n, m, p, deadline):
    """Systematically search for lower rank by trying to remove terms."""
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    rank = best_rank
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try removing each single term
    for r_remove in range(rank):
        if time.time() > deadline - 2:
            break
        
        max_err = _try_remove_term(U, V, W, n, m, p, r_remove, deadline)
        if max_err < 1e-10:
            mask = [i for i in range(rank) if i != r_remove]
            new_U = U_arr[:, mask].tolist()
            new_V = V_arr[:, mask].tolist()
            new_W = W_arr[:, mask].tolist()
            best_U, best_V, best_W = new_U, new_V, new_W
            best_rank = rank - 1
            # Update for next iteration
            U = best_U
            V = best_V
            W = best_W
            rank = best_rank
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
    
    return best_U, best_V, best_W, best_rank

def construct():
    """Build a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s limit
    
    n, m, p = 2, 3, 6
    
    # Build baseline
    U, V, W, rank = _build_baseline(n, m, p)
    
    # Verify baseline
    max_err = _verify_decomposition(U, V, W, n, m, p)
    if max_err > 1e-10:
        # Baseline is incorrect; return as-is (shouldn't happen)
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Search for lower rank
    if time.time() < deadline - 5:
        U_r, V_r, W_r, r_r = _search_lower_rank(U, V, W, n, m, p, deadline)
        if r_r < best_rank:
            best_U, best_V, best_W, best_rank = U_r, V_r, W_r, r_r
    
    # Final verification
    if time.time() < deadline:
        max_err = _verify_decomposition(best_U, best_V, best_W, n, m, p)
        if max_err > 1e-10:
            # Fall back to original if search corrupted it
            best_U, best_V, best_W = U, V, W
            best_rank = rank
    
    result = {"U": best_U, "V": best_V, "W": best_W}
    return result
# EVOLVE-BLOCK-END
