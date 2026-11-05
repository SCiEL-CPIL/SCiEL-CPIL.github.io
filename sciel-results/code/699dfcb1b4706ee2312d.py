import time
import numpy as np

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 130

def construct():
    n, m, p = DIMENSIONS
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE

    # Rank 66 decomposition
    # M(3,4,6) = 3 * M(3,4,2)
    # M(3,4,2) = M(1,4,2) rank 8 + M(2,4,2) rank 14
    # M(2,4,2) = 2 * M(2,2,2) rank 7 = 14
    
    rank = 66
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank or not check_deadline():
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx, current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx, current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx, current_rank] += val
        current_rank += 1
        
    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
                
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # Strassen 7
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
        
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
        
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
    
    # Attempt rank reduction via iterative component removal
    # Check if removing a component still allows exact reconstruction
    def verify_decomposition(U_mat, V_mat, W_mat, r):
        """Verify that the decomposition is exact."""
        # The tensor T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
        # For matrix mult: T[i,j,k] = 1 if i,j,k correspond to (row, mid, col) in A*B
        # A is n x m, B is m x p, C is n x p
        # T[(i,j),(j,k),(k,i)] = 1 for valid indices
        # Actually the tensor is: T[a,b,c] where a in [0,n*m), b in [0,m*p), c in [0,p*n)
        # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
        # Otherwise T[a,b,c] = 0
        # Check: sum_r U[a,r]*V[b,r]*W[c,r] should equal T[a,b,c]
        
        # Build the target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # Compute the reconstructed tensor
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is: for each r, outer product U[:,r] @ V[:,r] @ W[:,r]
        # More efficiently: T_recon = np.einsum('ar,br,cr->abc', U_mat[:, :r], V_mat[:, :r], W_mat[:, :r])
        T_recon = np.einsum('ar,br,cr->abc', U_mat[:, :r], V_mat[:, :r], W_mat[:, :r])
        
        return np.allclose(T, T_recon, atol=1e-10)
    
    def try_reduce_rank(U_mat, V_mat, W_mat, r, max_time=60):
        """Try to reduce rank by removing one component at a time."""
        start = time.time()
        current_r = r
        U_cur = U_mat[:, :current_r].copy()
        V_cur = V_mat[:, :current_r].copy()
        W_cur = W_mat[:, :current_r].copy()
        
        # Try removing each component
        while current_r > 1 and (time.time() - start) < max_time:
            if not check_deadline():
                break
            found_removal = False
            for remove_idx in range(current_r):
                if (time.time() - start) > max_time:
                    break
                # Try removing component remove_idx
                keep = [i for i in range(current_r) if i != remove_idx]
                if len(keep) == 0:
                    continue
                U_test = U_cur[:, keep]
                V_test = V_cur[:, keep]
                W_test = W_cur[:, keep]
                
                if verify_decomposition(U_test, V_test, W_test, len(keep)):
                    # Successfully removed a component
                    U_cur = U_test
                    V_cur = V_test
                    W_cur = W_test
                    current_r = len(keep)
                    found_removal = True
                    break
            
            if not found_removal:
                break
        
        return U_cur, V_cur, W_cur, current_r
    
    # Try to reduce the rank
    if check_deadline():
        U_red, V_red, W_red, reduced_rank = try_reduce_rank(U, V, W, rank, max_time=100)
        if reduced_rank < rank:
            U = U_red
            V = V_red
            W = W_red
            rank = reduced_rank
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
