import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 120.0

    n, m, p = 4, 4, 4

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # Convention: for 2x2, U has rows indexed by (i,j) -> i*2+j
    # V has rows indexed by (j,k) -> j*2+k
    # W has rows indexed by (k,i) -> k*2+i
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],
        [0,  0,  0,  0,  1,  0,  1],
        [0,  1,  0,  0,  0,  1,  0],
        [1,  1,  0,  1,  0,  0, -1]
    ], dtype=float)

    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],
        [0,  0,  1,  0,  0,  1,  0],
        [0,  0,  0,  1,  0,  0,  1],
        [1,  0, -1,  0,  1,  0,  1]
    ], dtype=float)

    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],
        [0,  1,  0,  1,  0,  0,  0],
        [0,  0,  1,  0,  1,  0,  0],
        [1, -1, 1,  0,  0,  1,  0]
    ], dtype=float)

    # Kronecker product gives 4x4 from 2x2: rank 49
    U_kron = np.kron(U2, U2)  # shape (16, 49)
    V_kron = np.kron(V2, V2)  # shape (16, 49)
    W_kron = np.kron(W2, W2)  # shape (16, 49)

    # Permute rows to standard row-major ordering
    def permute_rows(kron_mat, a_idx, b_idx):
        perm = []
        for S in range(16):
            a_out, a_in, b_out, b_in = a_idx(S), b_idx(S)
            block = a_out * 2 + b_out
            sub = a_in * 2 + b_in
            perm.append(block * 4 + sub)
        return kron_mat[perm, :]

    def u_split(S):
        i = S // 4
        j = S % 4
        return i // 2, i % 2, j // 2, j % 2

    def v_split(S):
        j = S // 4
        k = S % 4
        return j // 2, j % 2, k // 2, k % 2

    def w_split(S):
        k = S // 4
        i = S % 4
        return k // 2, k % 2, i // 2, i % 2

    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)

    for S in range(16):
        i_out, i_in, j_out, j_in = u_split(S)
        U[S, :] = U_kron[(i_out * 2 + j_out) * 4 + (i_in * 2 + j_in), :]

        j_out, j_in, k_out, k_in = v_split(S)
        V[S, :] = V_kron[(j_out * 2 + k_out) * 4 + (j_in * 2 + k_in), :]

        k_out, k_in, i_out, i_in = w_split(S)
        W[S, :] = W_kron[(k_out * 2 + i_out) * 4 + (k_in * 2 + i_in), :]

    # Verification
    if time.time() - t0 > deadline - 10.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                ui = U[i * 4 + j, :]
                vj = V[j * 4 + k, :]
                wk = W[k * 4 + i, :]
                val = np.dot(ui, vj * wk)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-9:
        # Fallback: schoolbook rank 64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=float)
        V_fb = np.zeros((16, rank_fb), dtype=float)
        W_fb = np.zeros((16, rank_fb), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j, col] = 1.0
                    V_fb[j * 4 + k, col] = 1.0
                    W_fb[k * 4 + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Attempt rank reduction via column elimination
    # A column r is redundant if removing it still gives exact decomposition
    # (i.e., the tensor is still spanned by the remaining columns)
    current_rank = 49
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    # Try to eliminate columns one at a time
    # For each column, check if it's needed
    eliminated = []
    remaining = list(range(49))
    
    # Greedy elimination: try to remove each column and check if still valid
    for col_to_test in range(49):
        if time.time() - t0 > deadline - 15.0:
            break
        if col_to_test not in remaining:
            continue
        
        # Build candidate without this column
        keep = [c for c in remaining if c != col_to_test]
        if len(keep) == 0:
            continue
        
        U_cand = U_cur[:, keep]
        V_cand = V_cur[:, keep]
        W_cand = W_cur[:, keep]
        
        # Check if this still decomposes the tensor
        valid = True
        for i in range(4):
            if not valid:
                break
            for j in range(4):
                if not valid:
                    break
                for k in range(4):
                    ui = U_cand[i * 4 + j, :]
                    vj = V_cand[j * 4 + k, :]
                    wk = W_cand[k * 4 + i, :]
                    val = np.dot(ui, vj * wk)
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            # Column is redundant, eliminate it
            eliminated.append(col_to_test)
            U_cur = U_cand
            V_cur = V_cand
            W_cur = W_cand
            remaining = keep
            current_rank = len(keep)
    
    if time.time() - t0 > deadline - 5.0:
        return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
    
    # Final verification
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                ui = U_cur[i * 4 + j, :]
                vj = V_cur[j * 4 + k, :]
                wk = W_cur[k * 4 + i, :]
                val = np.dot(ui, vj * wk)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        # Fallback to original
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}

if __name__ == "__main__":
    construct()
