import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute the decomposition
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    T_recon = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    return np.allclose(T_recon, T, atol=1e-10)

def _try_reduce_rank(U, V, W, deadline):
    """
    Attempt to reduce rank by finding linear dependencies among the rank-1 terms.
    Uses null-space projection on the tensor slices.
    Returns (U_new, V_new, W_new, new_rank) or None if no reduction found.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # First, verify current decomposition
    if not _verify(U, V, W):
        return None
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to remove each column and check if the remaining columns can still span
    for r_remove in range(best_rank):
        if time.time() > deadline - 10:
            break
        
        # Create reduced matrices by removing column r_remove
        U_red = np.delete(U, r_remove, axis=1)
        V_red = np.delete(V, r_remove, axis=1)
        W_red = np.delete(W, r_remove, axis=1)
        
        # Verify
        T_recon = np.einsum('ar,br,cr->abc', U_red, V_red, W_red)
        if np.allclose(T_recon, T, atol=1e-10):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = best_rank - 1
            # Continue trying to remove more columns from the reduced set
            U, V, W = U_red, V_red, W_red
            rank = best_rank
            # Restart the loop with the new set
            r_remove = -1  # Will be incremented to 0
            continue
        else:
            # Restore
            U, V, W = best_U, best_V, best_W
            rank = best_rank
    
    if best_rank < rank:
        return best_U, best_V, best_W, best_rank
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    # Target Rank 39 Construction
    # 5 blocks of 2x2x2 (Strassen, rank 7) + 1 block of 2x2x1 (Schoolbook, rank 4)
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0

    # Helper to add a Strassen block for columns k0, k1
    def add_strassen_block(k0, k1):
        nonlocal r
        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]

        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]

        # Strassen terms for 2x2 * 2x2
        # m1 = (a+d)(e+h)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (c+d)e
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a(f-h)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = d(g-e)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a+b)h
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

    # Add 5 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        if time.time() > deadline - 5:
            break
        k0 = block * 2
        k1 = block * 2 + 1
        add_strassen_block(k0, k1)

    # Last column k=10: Schoolbook
    if r < rank:
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = 11 + k2 # B[1, 10]
        w_c02 = 2 * k2      # C[0, 10]
        w_c12 = 2 * k2 + 1  # C[1, 10]

        # a * B[0, 10]
        U[0, r] += 1
        V[v_e2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # b * B[1, 10]
        U[1, r] += 1
        V[v_g2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # c * B[0, 10]
        U[2, r] += 1
        V[v_e2, r] += 1
        W[w_c12, r] += 1
        r += 1

        # d * B[1, 10]
        U[3, r] += 1
        V[v_g2, r] += 1
        W[w_c12, r] += 1
        r += 1

    # Verify the construction
    if time.time() < deadline - 2:
        if _verify(U, V, W):
            # Try to reduce rank via null-space projection
            if time.time() < deadline - 15:
                result = _try_reduce_rank(U, V, W, deadline)
                if result is not None:
                    U_red, V_red, W_red, new_rank = result
                    if _verify(U_red, V_red, W_red):
                        return {
                            "U": U_red.tolist(),
                            "V": V_red.tolist(),
                            "W": W_red.tolist()
                        }
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback: Schoolbook (Rank 44)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
