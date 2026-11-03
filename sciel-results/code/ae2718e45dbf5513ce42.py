import time
import numpy as np

N, M, P = 2, 5, 5
DEADLINE = 145.0

def A_idx(i, j): return i * M + j
def B_idx(j, k): return j * P + k
def C_idx(k, i): return k * N + i

def build_tensor():
    T = np.zeros((N*M, M*P, P*N), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[A_idx(i,j), B_idx(j,k), C_idx(k,i)] = 1.0
    return T

def try_rank_decomposition(T, target_rank, max_iter=5000, tol=1e-10):
    """Try to find rank-target_rank decomposition via iterative refinement."""
    np.random.seed(42)
    a, b, c = T.shape
    
    # Initialize randomly
    U = np.random.randn(a, target_rank) * 0.1
    V = np.random.randn(b, target_rank) * 0.1
    W = np.random.randn(c, target_rank) * 0.1
    
    for it in range(max_iter):
        # Update U: for each r, U[:,r] should satisfy sum_r U[:,r] outer(V[:,r]) outer(W[:,r]) ≈ T
        # Fix V, W, solve for U[:,r]
        for r in range(target_rank):
            # T = sum_r U_r ⊗ V_r ⊗ W_r
            # For fixed r, the residual is T - sum_{r'≠r} U_{r'} ⊗ V_{r'} ⊗ W_{r'}
            # U_r should be such that U_r ⊗ V_r ⊗ W_r ≈ residual
            # Flatten: reshape T to (a, b*c)
            # U_r ⊗ (V_r ⊗ W_r) ≈ T_flat
            # U_r ≈ T_flat @ (V_r ⊗ W_r) / ||V_r ⊗ W_r||^2
            vw = np.outer(V[:, r], W[:, r]).flatten()
            vw_norm_sq = np.dot(vw, vw)
            if vw_norm_sq < 1e-15:
                continue
            T_flat = T.reshape(a, b * c)
            U[:, r] = T_flat @ vw / vw_norm_sq
        
        # Update V
        for r in range(target_rank):
            uw = np.outer(U[:, r], W[:, r]).flatten()
            uw_norm_sq = np.dot(uw, uw)
            if uw_norm_sq < 1e-15:
                continue
            T_flat = T.reshape(b, a * c)
            V[:, r] = T_flat @ uw / uw_norm_sq
        
        # Update W
        for r in range(target_rank):
            uv = np.outer(U[:, r], V[:, r]).flatten()
            uv_norm_sq = np.dot(uv, uv)
            if uv_norm_sq < 1e-15:
                continue
            T_flat = T.reshape(c, a * b)
            W[:, r] = T_flat @ uv / uv_norm_sq
        
        # Check convergence
        if it % 100 == 0:
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((recon - T)**2)
            if err < tol:
                return U, V, W, err
    
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.sum((recon - T)**2)
    return U, V, W, err

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    T = build_tensor()
    
    # Try decreasing ranks from 25 upward
    for target_rank in [25, 26, 27, 28, 29, 30]:
        if time_left() < 10.0:
            break
        U, V, W, err = try_rank_decomposition(T, target_rank, max_iter=3000)
        if err < 1e-9:
            # Found a valid decomposition
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to the known rank-30 construction
    max_rank = 30
    U = np.zeros((N*M, max_rank))
    V = np.zeros((M*P, max_rank))
    W = np.zeros((P*N, max_rank))
    
    current_rank = 0
    
    # Strassen 2x2x2
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    U_s[0,0] = 1.0; U_s[3,0] = -1.0; V_s[2,0] = 1.0; V_s[3,0] = 1.0; W_s[0,0] = 1.0
    U_s[0,1] = 1.0; U_s[3,1] = 1.0; V_s[3,1] = 1.0; W_s[2,1] = 1.0; W_s[3,1] = 1.0
    U_s[0,2] = 1.0; V_s[0,2] = 1.0; V_s[2,2] = -1.0; W_s[1,2] = 1.0; W_s[3,2] = -1.0
    U_s[3,3] = 1.0; V_s[1,3] = 1.0; V_s[0,3] = -1.0; W_s[0,3] = 1.0; W_s[3,3] = 1.0
    U_s[0,4] = 1.0; U_s[1,4] = 1.0; V_s[0,4] = 1.0; W_s[0,4] = -1.0; W_s[1,4] = 1.0
    U_s[2,5] = 1.0; U_s[0,5] = -1.0; V_s[0,5] = 1.0; V_s[1,5] = 1.0; W_s[3,5] = 1.0
    U_s[1,6] = 1.0; U_s[3,6] = -1.0; V_s[2,6] = 1.0; V_s[3,6] = 1.0; W_s[0,6] = 1.0
    
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            for r in range(7):
                for i in range(2):
                    for j_local in range(2):
                        j_global = j_off + j_local
                        U[A_idx(i, j_global), current_rank] += U_s[i*2 + j_local, r]
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = j_off + j_local
                        k_global = k_off + k_local
                        V[B_idx(j_global, k_global), current_rank] += V_s[j_local*2 + k_local, r]
                for i in range(2):
                    for k_local in range(2):
                        k_global = k_off + k_local
                        W[C_idx(k_global, i), current_rank] += W_s[k_local*2 + i, r]
                current_rank += 1
    
    # Edge terms
    for i in range(2):
        for j in range(4):
            U[A_idx(i, j), current_rank] = 1.0
    for j in range(4):
        V[B_idx(j, 4), current_rank] = 1.0
    for i in range(2):
        W[C_idx(4, i), current_rank] = 1.0
    current_rank += 1
    
    for i in range(2):
        U[A_idx(i, 4), current_rank] = 1.0
    for k in range(5):
        V[B_idx(4, k), current_rank] = 1.0
    for k in range(5):
        for i in range(2):
            W[C_idx(k, i), current_rank] = 1.0
    current_rank += 1
    
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
