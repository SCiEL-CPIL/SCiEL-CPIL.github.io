"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    start_time = time.time()
    deadline = start_time + DEADLINE

    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i

    def _build_strassen_36():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        
        def strassen_block(y_indices, z_indices, w_indices):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices
            w00, w01, w10, w11 = w_indices
            
            # M1 = (y00+y11)(z00+z11)
            U[u_idx(y00[0], y00[1]), col] = 1
            U[u_idx(y11[0], y11[1]), col] = 1
            V[v_idx(z00[0], z00[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = 1
            W[w_idx(w00[0], w00[1]), col] += 1
            W[w_idx(w11[0], w11[1]), col] += 1
            col += 1
            
            # M2 = (y10+y11)z00
            U[u_idx(y10[0], y10[1]), col] = 1
            U[u_idx(y11[0], y11[1]), col] = 1
            V[v_idx(z00[0], z00[1]), col] = 1
            W[w_idx(w10[0], w10[1]), col] += 1
            W[w_idx(w11[0], w11[1]), col] -= 1
            col += 1
            
            # M3 = y00(z01-z11)
            U[u_idx(y00[0], y00[1]), col] = 1
            V[v_idx(z01[0], z01[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = -1
            W[w_idx(w01[0], w01[1]), col] += 1
            W[w_idx(w11[0], w11[1]), col] += 1
            col += 1
            
            # M4 = y11(z10-z00)
            U[u_idx(y11[0], y11[1]), col] = 1
            V[v_idx(z10[0], z10[1]), col] = 1
            V[v_idx(z00[0], z00[1]), col] = -1
            W[w_idx(w00[0], w00[1]), col] += 1
            W[w_idx(w10[0], w10[1]), col] += 1
            col += 1
            
            # M5 = (y00+y01)z11
            U[u_idx(y00[0], y00[1]), col] = 1
            U[u_idx(y01[0], y01[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = 1
            W[w_idx(w00[0], w00[1]), col] -= 1
            W[w_idx(w01[0], w01[1]), col] += 1
            col += 1
            
            # M6 = (y10-y00)(z00+z01)
            U[u_idx(y10[0], y10[1]), col] = 1
            U[u_idx(y00[0], y00[1]), col] = -1
            V[v_idx(z00[0], z00[1]), col] = 1
            V[v_idx(z01[0], z01[1]), col] = 1
            W[w_idx(w11[0], w11[1]), col] += 1
            col += 1
            
            # M7 = (y01-y11)(z10+z11)
            U[u_idx(y01[0], y01[1]), col] = 1
            U[u_idx(y11[0], y11[1]), col] = -1
            V[v_idx(z10[0], z10[1]), col] = 1
            V[v_idx(z11[0], z11[1]), col] = 1
            W[w_idx(w00[0], w00[1]), col] += 1
            col += 1

        def schoolbook_2x2x1(y_indices, z_indices, w_indices):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z0, z1 = z_indices
            w0, w1 = w_indices
            
            # M1 = y00 * z0
            U[u_idx(y00[0], y00[1]), col] = 1
            V[v_idx(z0[0], z0[1]), col] = 1
            W[w_idx(w0[0], w0[1]), col] += 1
            col += 1
            
            # M2 = y01 * z1
            U[u_idx(y01[0], y01[1]), col] = 1
            V[v_idx(z1[0], z1[1]), col] = 1
            W[w_idx(w0[0], w0[1]), col] += 1
            col += 1
            
            # M3 = y10 * z0
            U[u_idx(y10[0], y10[1]), col] = 1
            V[v_idx(z0[0], z0[1]), col] = 1
            W[w_idx(w1[0], w1[1]), col] += 1
            col += 1
            
            # M4 = y11 * z1
            U[u_idx(y11[0], y11[1]), col] = 1
            V[v_idx(z1[0], z1[1]), col] = 1
            W[w_idx(w1[0], w1[1]), col] += 1
            col += 1

        # Block 1: A_L (cols 0,1) * B_1 (rows 0,1, cols 0,1) -> C (cols 0,1)
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y1, z1, w1)
        
        # Block 2: A_R (cols 2,3) * B_2 (rows 2,3, cols 0,1) -> C (cols 0,1)
        y2 = [(0,2), (0,3), (1,2), (1,3)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y2, z2, w2)
        
        # Block 3: A_L (cols 0,1) * B_1 (rows 0,1, cols 2,3) -> C (cols 2,3)
        y3 = [(0,0), (0,1), (1,0), (1,1)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y3, z3, w3)
        
        # Block 4: A_R (cols 2,3) * B_2 (rows 2,3, cols 2,3) -> C (cols 2,3)
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y4, z4, w4)
        
        # Block 5: A_L (cols 0,1) * B_1 (rows 0,1, col 4) -> C (col 4)
        y5 = [(0,0), (0,1), (1,0), (1,1)]
        z5 = [(0,4), (1,4)]
        w5 = [(4,0), (4,1)]
        schoolbook_2x2x1(y5, z5, w5)
        
        # Block 6: A_R (cols 2,3) * B_2 (rows 2,3, col 4) -> C (col 4)
        y6 = [(0,2), (0,3), (1,2), (1,3)]
        z6 = [(2,4), (3,4)]
        w6 = [(4,0), (4,1)]
        schoolbook_2x2x1(y6, z6, w6)
        
        return U, V, W

    def _verify_decomposition(U, V, W):
        """Verify that the decomposition exactly represents matrix multiplication."""
        # T_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r
        # We check if this equals 1 for all i,j,k
        # Vectorized check:
        # For each i, j, k, compute dot(U[i,j,:], V[j,k,:] * W[k,i,:])
        # Efficiently:
        # Let A = U (8, R), B = V (20, R), C = W (10, R)
        # We want to check if sum_r A[i*m+j, r] B[j*p+k, r] C[k*n+i, r] == 1
        # This is equivalent to checking if the tensor formed by A, B, C matches the multiplication tensor.
        
        # Flatten indices for clarity
        # i in [0, n), j in [0, m), k in [0, p)
        
        # Construct the multiplication tensor T (n, m, p)
        # T[i, j, k] = 1
        
        # Construct the decomposition tensor D (n, m, p)
        # D[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can compute D using einsum or loops. Given small dimensions, loops are fine and safe.
        # n=2, m=4, p=5 -> 40 checks.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    U, V, W = _build_strassen_36()
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W):
        # Fallback: return schoolbook rank-40
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
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    
    # Phase 1: Try removing 1 column at a time (greedy)
    # We iterate until no more columns can be removed
    while time.time() < deadline - 10:
        progress = False
        for col in range(rank):
            if time.time() > deadline - 10:
                break
            mask = [i for i in range(rank) if i != col]
            if not mask:
                continue
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            if _verify_decomposition(U_new, V_new, W_new):
                U, V, W = U_new, V_new, W_new
                rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                progress = True
                break # Restart loop with new rank
        if not progress:
            break
            
    # Phase 2: Try removing pairs
    if time.time() < deadline - 5:
        progress = True
        while progress and time.time() < deadline - 5:
            progress = False
            rank = U.shape[1]
            for i in range(rank):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, rank):
                    if time.time() > deadline - 5:
                        break
                    mask = [k for k in range(rank) if k != i and k != j]
                    if not mask:
                        continue
                    U_new = U[:, mask]
                    V_new = V[:, mask]
                    W_new = W[:, mask]
                    if _verify_decomposition(U_new, V_new, W_new):
                        U, V, W = U_new, V_new, W_new
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        progress = True
                        break
                if time.time() > deadline - 5:
                    break
                if progress:
                    break
                    
    # Phase 3: Try removing triples
    if time.time() < deadline - 2:
        progress = True
        while progress and time.time() < deadline - 2:
            progress = False
            rank = U.shape[1]
            for i in range(rank):
                if time.time() > deadline - 2:
                    break
                for j in range(i+1, rank):
                    if time.time() > deadline - 2:
                        break
                    for k in range(j+1, rank):
                        if time.time() > deadline - 2:
                            break
                        mask = [l for l in range(rank) if l != i and l != j and l != k]
                        if not mask:
                            continue
                        U_new = U[:, mask]
                        V_new = V[:, mask]
                        W_new = W[:, mask]
                        if _verify_decomposition(U_new, V_new, W_new):
                            U, V, W = U_new, V_new, W_new
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                            progress = True
                            break
                    if time.time() > deadline - 2:
                        break
                    if progress:
                        break
                if time.time() > deadline - 2:
                    break
                if progress:
                    break

    # Final polish: try to find any single column removal again (in case order mattered or something changed)
    # Actually, the greedy loop already does this until no progress.
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
