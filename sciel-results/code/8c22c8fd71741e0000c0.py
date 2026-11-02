import numpy as np
import time
from scipy.linalg import qr

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    block_size = 2
    n_blocks = n // block_size  # 3

    # Total rank is n_blocks^3 * 7
    rank = n_blocks ** 3 * 7

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Strassen 2x2 block decomposition
    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                # M1 = (X00 + X11)(Y00 + Y11)
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M2 = (X10 + X11)Y00
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                W[w_idx(ib, jb, 1, 1), col] -= 1
                col += 1

                # M3 = X00(Y01 - Y11)
                U[u_idx(ib, kb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M4 = X11(Y10 - Y00)
                U[u_idx(ib, kb, 1, 1), col] += 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                W[w_idx(ib, jb, 1, 0), col] += 1
                col += 1

                # M5 = (X00 + X01)Y11
                U[u_idx(ib, kb, 0, 0), col] += 1
                U[u_idx(ib, kb, 0, 1), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] -= 1
                W[w_idx(ib, jb, 0, 1), col] += 1
                col += 1

                # M6 = (X10 - X00)(Y00 + Y01)
                U[u_idx(ib, kb, 1, 0), col] += 1
                U[u_idx(ib, kb, 0, 0), col] -= 1
                V[v_idx(kb, jb, 0, 0), col] += 1
                V[v_idx(kb, jb, 0, 1), col] += 1
                W[w_idx(ib, jb, 1, 1), col] += 1
                col += 1

                # M7 = (X01 - X11)(Y10 + Y11)
                U[u_idx(ib, kb, 0, 1), col] += 1
                U[u_idx(ib, kb, 1, 1), col] -= 1
                V[v_idx(kb, jb, 1, 0), col] += 1
                V[v_idx(kb, jb, 1, 1), col] += 1
                W[w_idx(ib, jb, 0, 0), col] += 1
                col += 1

    # Rank reduction: find linearly dependent columns in the tensor sense
    # Each column r contributes a rank-1 tensor T_r[i,j,k] = U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r]
    # If T_r is a linear combination of other T_s, we can remove column r.
    
    if time.time() < deadline:
        try:
            Uf = U.astype(np.float64)
            Vf = V.astype(np.float64)
            Wf = W.astype(np.float64)
            
            # Build tensor matrix: rows = (i,j,k) triples, cols = rank indices
            # T_matrix[i,j,k,r] = U[i*m+k,r] * V[k*p+j,r] * W[j*n+i,r]
            
            U_reshaped = Uf.reshape(n, m, rank)  # [i,k,r]
            V_reshaped = Vf.reshape(m, p, rank)  # [k,j,r]
            W_reshaped = Wf.reshape(p, n, rank)  # [j,i,r]
            
            # T[i,j,k,r] = U_reshaped[i,k,r] * V_reshaped[k,j,r] * W_reshaped[j,i,r]
            tensor_matrix = np.einsum('ikr,kjr,jir->ijk r', U_reshaped, V_reshaped, W_reshaped)
            tensor_matrix = tensor_matrix.reshape(n * m * p, rank)  # (216, 189)
            
            # Use QR with column pivoting to find independent columns
            Q, R_qr, piv = qr(tensor_matrix, pivoting=True)
            
            # Determine numerical rank
            tol = 1e-10
            diag_R = np.abs(np.diag(R_qr))
            num_independent = np.sum(diag_R > tol * diag_R[0] if len(diag_R) > 0 else 0)
            
            if num_independent < rank and num_independent > 0:
                # Keep the first num_independent columns in the pivoted order
                keep_cols = np.sort(piv[:num_independent])
                
                U = U[:, keep_cols]
                V = V[:, keep_cols]
                W = W[:, keep_cols]
                rank = len(keep_cols)
                
        except Exception:
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
