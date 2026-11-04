import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Seed solution
    rank_seed = n * m * p
    U_seed = [[0] * rank_seed for _ in range(n * m)]
    V_seed = [[0] * rank_seed for _ in range(m * p)]
    W_seed = [[0] * rank_seed for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j][col] = 1
                V_seed[j * p + k][col] = 1
                W_seed[k * n + i][col] = 1
                col += 1
    
    # Create target tensor T
    # T is size (n*m) x (m*p) x (p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_solution(U, V, W):
        # Check if U, V, W exactly decompose T
        # T_approx[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
        # i in 0..nm-1, j in 0..mp-1, k in 0..pn-1
        # Note: The indices in T are flattened.
        # We need to verify that for all i, j, k, the sum matches T[i, j, k].
        # T is sparse.
        # We can compute the full tensor and compare.
        # Size is 6 * 9 * 6 = 324. Very small.
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        r = U_np.shape[1]
        
        # Compute tensor
        # T_approx = np.einsum('ic,jc,kc->ijk', U_np, V_np, W_np)
        # This might be slow if done in pure python, but numpy is fine.
        # However, we only need to check the non-zero entries of T.
        # But wait, if there are spurious non-zero entries in T_approx, it's invalid.
        # So we must check the whole tensor or at least ensure T_approx == T.
        
        # Efficient check:
        # For each c, add U[:, c] * V[:, c] * W[:, c] to T_approx
        T_approx = np.zeros((n * m, m * p, p * n))
        for c in range(r):
            T_approx += np.outer(np.outer(U_np[:, c], V_np[:, c]), W_np[:, c])
            
        if not np.allclose(T_approx, T, atol=1e-9):
            return False
        return True

    def try_rank(r, time_limit):
        start_time = time.time()
        # Try a few random restarts
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random integers in {-1, 0, 1} or similar
            # Or random floats?
            # Integer/half-integer requirement.
            # Let's try random integers in {-1, 0, 1}
            U = np.random.randint(-1, 2, (n * m, r))
            V = np.random.randint(-1, 2, (m * p, r))
            W = np.random.randint(-1, 2, (p * n, r))
            
            # ALS optimization
            for it in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # For fixed V, W, we want to minimize || T - sum_c U[:, c] (V * W)[:, c] ||^2
                # This is a linear least squares problem for each i?
                # Actually, T is a 3D array.
                # We can reshape T to (n*m, (m*p)*(p*n)).
                # T_flat[i, j*pn + k] = sum_c U[i, c] * (V[j, c] * W[k, c])
                # Let M[i, c] = U[i, c].
                # Let B[j, k, c] = V[j, c] * W[k, c].
                # We want U such that U @ B == T_flat?
                # T_flat[i, idx] = sum_c U[i, c] B[idx, c]
                # Where idx corresponds to (j, k).
                
                B = np.einsum('jc,kc->jkc', V, W).reshape(m * p, p * n, r)
                B_flat = B.reshape(m * p * p * n, r)
                T_flat = T.reshape(n * m, m * p * p * n)
                
                # Solve U @ B_flat.T = T_flat.T ?
                # T_flat[i, :] = sum_c U[i, c] B_flat[:, c]
                # So T_flat = U @ B_flat.T
                # U = T_flat @ pinv(B_flat.T)
                try:
                    U = np.linalg.lstsq(B_flat.T, T_flat.T, rcond=None)[0].T
                except:
                    break
                
                # Update V
                # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # T_flat[j, i*pn + k] = sum_c V[j, c] (U[i, c] * W[k, c])
                # A[i, k, c] = U[i, c] * W[k, c]
                # A_flat[i*pn + k, c]
                A = np.einsum('ic,kc->ikc', U, W).reshape(n * m, p * n, r)
                A_flat = A.reshape(n * m * p * n, r)
                # T is indexed by i, j, k.
                # We need to reorder T to (j, i, k)
                T_reorder = np.transpose(T, (1, 0, 2)) # j, i, k
                T_reorder_flat = T_reorder.reshape(m * p, n * m * p * n)
                
                try:
                    V = np.linalg.lstsq(A_flat.T, T_reorder_flat.T, rcond=None)[0].T
                except:
                    break
                
                # Update W
                # T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]
                # T_flat[k, i*mp + j] = sum_c W[k, c] (U[i, c] * V[j, c])
                # C[i, j, c] = U[i, c] * V[j, c]
                # C_flat[i*mp + j, c]
                C = np.einsum('ic,jc->ijc', U, V).reshape(n * m, m * p, r)
                C_flat = C.reshape(n * m * m * p, r)
                # T is indexed by i, j, k.
                # We need to reorder T to (k, i, j)
                T_reorder2 = np.transpose(T, (2, 0, 1)) # k, i, j
                T_reorder2_flat = T_reorder2.reshape(p * n, n * m * m * p)
                
                try:
                    W = np.linalg.lstsq(C_flat.T, T_reorder2_flat.T, rcond=None)[0].T
                except:
                    break
            
            # Round to half-integers and check
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if check_solution(U_rounded, V_rounded, W_rounded):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        return None

    # Try ranks from 14 to 17
    # Allocate time for each
    total_time = 140
    ranks_to_try = [14, 15, 16, 17]
    
    for r in ranks_to_try:
        time_per_rank = total_time / len(ranks_to_try)
        result = try_rank(r, time_per_rank)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        total_time -= time_per_rank
        if total_time < 10:
            break

    # Fallback to seed
    return {"U": U_seed, "V": V_seed, "W": W_seed}
# EVOLVE-BLOCK-END
