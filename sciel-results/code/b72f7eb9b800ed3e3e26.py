"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 148.0
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Known Rank 18 Solution (Fallback)
    # 2 Strassen blocks (rank 7 each) for first 4 cols, 1 Schoolbook (rank 4) for last col
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(t_offset, k_start):
            # Strassen products M1..M7
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            
            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1
            
            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        
        t = 14
        U[0][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[1][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[2][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 1][t] = 1
        t += 1
        
        U[3][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 1][t] = 1
        
        return {"U": U, "V": V, "W": W}

    def check_exact_full(U, V, W, rank, tol=1e-5):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Target: T[i, j, k] where C[i,k] = sum_j A[i,j] B[j,k]
        # T[i,j,k] = 1 if j==j (identity in middle), 0 otherwise?
        # No, the tensor is T[i, j, k] for A[i,j] * B[j,k].
        # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == delta_{j,j'}?
        # Wait, the standard tensor for matrix mult C = AB is:
        # T[i, j, k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        # Here T[i,j,k] = 1 if j==j (which is always 1 for the index j in the sum).
        # Actually, the tensor indices are (i, j, k) where i is row of C, j is inner dim, k is col of C.
        # The value is 1.
        # So we check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integers(arr, tol=1e-5):
        """Snap values to nearest half-integer if close enough."""
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def als_step_U(V, W, rank):
        """Vectorized ALS step for U."""
        U_new = np.zeros((n*m, rank))
        for i in range(n):
            for j in range(m):
                # Z[k, c] = W[k*n+i, c] for k in 0..p-1
                Z = np.array([W[k*n + i, :] for k in range(p)])  # (p, rank)
                V_j = V[j*p : (j+1)*p, :]  # (p, rank)
                
                P = Z * V_j  # (p, rank)
                A_mat = P.T @ P  # (rank, rank)
                b_vec = np.sum(P, axis=0)  # (rank,)
                
                try:
                    u = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    u = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                U_new[i*m + j, :] = u
        return U_new

    def als_step_V(U, W, rank):
        """Vectorized ALS step for V."""
        V_new = np.zeros((m*p, rank))
        for j in range(m):
            for k in range(p):
                # U_i[i, c] = U[i*m+j, c] for i in 0..n-1
                U_i = np.array([U[i*m + j, :] for i in range(n)])  # (n, rank)
                # W_k[i, c] = W[k*n+i, c] for i in 0..n-1
                W_k = np.array([W[k*n + i, :] for i in range(n)])  # (n, rank)
                
                P = U_i * W_k  # (n, rank)
                A_mat = P.T @ P  # (rank, rank)
                b_vec = np.sum(P, axis=0)  # (rank,)
                
                try:
                    v = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    v = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                V_new[j*p + k, :] = v
        return V_new

    def als_step_W(U, V, rank):
        """Vectorized ALS step for W."""
        W_new = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                # U_i[j, c] = U[i*m+j, c] for j in 0..m-1
                U_i = np.array([U[i*m + j, :] for j in range(m)])  # (m, rank)
                # V_k[j, c] = V[j*p+k, c] for j in 0..m-1
                V_k = np.array([V[j*p + k, :] for j in range(m)])  # (m, rank)
                
                P = U_i * V_k  # (m, rank)
                A_mat = P.T @ P  # (rank, rank)
                b_vec = np.sum(P, axis=0)  # (rank,)
                
                try:
                    w = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                W_new[k*n + i, :] = w
        return W_new

    def polish(U, V, W, rank, max_iter=20, tol=1e-8):
        """High-precision polish pass with tighter tolerance."""
        for _ in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            U = als_step_U(V, W, rank)
            V = als_step_V(U, W, rank)
            W = als_step_W(U, V, rank)
            if check_exact_full(U, V, W, rank, tol=tol):
                return U, V, W
        return U, V, W

    def try_rank_als(target_rank, max_iter=50):
        if time.time() > DEADLINE - 10:
            return None
        
        # Initializations
        inits = []
        
        # 1. Truncated Rank 18 solution
        sol18 = get_rank_18_solution()
        U18 = np.array(sol18['U'])
        V18 = np.array(sol18['V'])
        W18 = np.array(sol18['W'])
        
        if target_rank < 18:
            U0 = U18[:, :target_rank]
            V0 = V18[:, :target_rank]
            W0 = W18[:, :target_rank]
            inits.append((U0, V0, W0))
            
            # 2. Random initialization
            np.random.seed(42)
            U_r = np.random.uniform(-1, 1, (n*m, target_rank))
            V_r = np.random.uniform(-1, 1, (m*p, target_rank))
            W_r = np.random.uniform(-1, 1, (p*n, target_rank))
            inits.append((U_r, V_r, W_r))
            
            # 3. Perturbed Rank 18
            U_p = U18[:, :target_rank].copy()
            V_p = V18[:, :target_rank].copy()
            W_p = W18[:, :target_rank].copy()
            U_p += np.random.uniform(-0.1, 0.1, U_p.shape)
            V_p += np.random.uniform(-0.1, 0.1, V_p.shape)
            W_p += np.random.uniform(-0.1, 0.1, W_p.shape)
            inits.append((U_p, V_p, W_p))
            
            # 4. Another random with different seed
            np.random.seed(123)
            U_r2 = np.random.uniform(-0.5, 0.5, (n*m, target_rank))
            V_r2 = np.random.uniform(-0.5, 0.5, (m*p, target_rank))
            W_r2 = np.random.uniform(-0.5, 0.5, (p*n, target_rank))
            inits.append((U_r2, V_r2, W_r2))

        for U, V, W in inits:
            if time.time() > DEADLINE - 15:
                break
            
            for _ in range(max_iter):
                if time.time() > DEADLINE - 15:
                    break
                U = als_step_U(V, W, target_rank)
                V = als_step_V(U, W, target_rank)
                W = als_step_W(U, V, target_rank)
                
                # Check convergence
                if check_exact_full(U, V, W, target_rank, tol=1e-6):
                    # Polish with tighter tolerance
                    U, V, W = polish(U, V, W, target_rank)
                    
                    # Try half-integer snap
                    U_r = snap_to_half_integers(U)
                    V_r = snap_to_half_integers(V)
                    W_r = snap_to_half_integers(W)
                    if U_r is not None and V_r is not None and W_r is not None:
                        if check_exact_full(U_r, V_r, W_r, target_rank, tol=1e-6):
                            return (U_r, V_r, W_r)
                    
                    # Try integer snap
                    U_i = np.round(U)
                    V_i = np.round(V)
                    W_i = np.round(W)
                    if check_exact_full(U_i, V_i, W_i, target_rank, tol=1e-6):
                        return (U_i, V_i, W_i)
        return None

    # Try Rank 17
    decomp_17 = None
    if time.time() < DEADLINE - 25:
        decomp_17 = try_rank_als(17)
    
    if decomp_17:
        U, V, W = decomp_17
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try Rank 16
    decomp_16 = None
    if time.time() < DEADLINE - 25:
        decomp_16 = try_rank_als(16)
        
    if decomp_16:
        U, V, W = decomp_16
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try Rank 15 if time permits
    decomp_15 = None
    if time.time() < DEADLINE - 25:
        decomp_15 = try_rank_als(15)
        
    if decomp_15:
        U, V, W = decomp_15
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return get_rank_18_solution()

# EVOLVE-BLOCK-END
