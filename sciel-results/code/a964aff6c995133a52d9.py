import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 28
    fallback_rank = n * m * p # 30
    
    start_time = time.time()
    max_time = 140.0
    
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
    def residual(U, V, W):
        res = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            res += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.linalg.norm(res.flatten() - T_flat)

    def solve_rank(R):
        # Random sparse init
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Sparsify
        for r in range(R):
            U[:, r][np.random.choice(n*m, max(1, int(n*m*0.3)), replace=False)] = 0
            V[:, r][np.random.choice(m*p, max(1, int(m*p*0.3)), replace=False)] = 0
            W[:, r][np.random.choice(p*n, max(1, int(p*n*0.3)), replace=False)] = 0
            
        for step in range(50):
            if time.time() - start_time > max_time:
                return None, None, None, 1e9
            
            # Solve W
            A_w = np.zeros((n*m*p, p*n*R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_t = i*m*p + j*p + k
                        for r in range(R):
                            idx_w = r*(p*n) + k*n + i
                            A_w[idx_t, idx_w] = U[i*m+j, r] * V[j*p+k, r]
            W_flat, _, _, _ = np.linalg.lstsq(A_w, T_flat, rcond=None)
            W = W_flat.reshape(p*n, R)
            
            # Solve U
            A_u = np.zeros((n*m*p, n*m*R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_t = i*m*p + j*p + k
                        for r in range(R):
                            idx_u = r*(n*m) + i*m + j
                            A_u[idx_t, idx_u] = V[j*p+k, r] * W[k*n+i, r]
            U_flat, _, _, _ = np.linalg.lstsq(A_u, T_flat, rcond=None)
            U = U_flat.reshape(n*m, R)
            
            # Solve V
            A_v = np.zeros((n*m*p, m*p*R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_t = i*m*p + j*p + k
                        for r in range(R):
                            idx_v = r*(m*p) + j*p + k
                            A_v[idx_t, idx_v] = U[i*m+j, r] * W[k*n+i, r]
            V_flat, _, _, _ = np.linalg.lstsq(A_v, T_flat, rcond=None)
            V = V_flat.reshape(m*p, R)
            
            res = residual(U, V, W)
            if res < 1e-9:
                return U, V, W, res
        return U, V, W, residual(U, V, W)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    best_rank = fallback_rank
    best_U, best_V, best_W = None, None, None

    # Try rank 28
    for attempt in range(30):
        if time.time() - start_time > max_time:
            break
        U, V, W, res = solve_rank(target_rank)
        if res < 1e-9:
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            res_r = residual(U_r, V_r, W_r)
            if res_r < 1e-9:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
            else:
                # If rounding breaks it, keep floats? 
                # Prompt says "integer/half-integer entries". 
                # If my solution doesn't have them, it might be invalid?
                # "exactly decompose ... with integer/half-integer entries"
                # So I must return half-integers.
                # If the float solution is not close to half-integers, it's a bad solution.
                pass

    if best_rank == fallback_rank:
        U, V, W = get_schoolbook()
    else:
        U, V, W = best_U, best_V, best_W

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
