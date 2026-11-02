"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
from scipy.optimize import minimize
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
        # Solve for U: U[i*m+j, c]
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For fixed i, j:
        # sum_c U[i*m+j, c] * (sum_k V[j*p+k, c] * W[k*n+i, c]) = 1
        # Let M[c] = sum_k V[j*p+k, c] * W[k*n+i, c]
        # Then sum_c U[i*m+j, c] * M[c] = 1
        # This is a linear system: U_vec dot M_vec = 1.
        # Wait, this is an underdetermined system if rank > 1.
        # We want the minimum norm solution? Or just any solution?
        # ALS usually minimizes squared error.
        # Error = sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
        # Gradient w.r.t U[i,j,c] is -2 * sum_k (residual[i,j,k]) * V[j,k,c] * W[k,i,c]
        # Setting gradient to 0:
        # sum_c U[i,j,c] * sum_k V[j,k,c'] W[k,i,c'] V[j,k,c] W[k,i,c] = sum_k V[j,k,c] W[k,i,c]
        # Let A[c, c'] = sum_k V[j,k,c] W[k,i,c'] V[j,k,c'] W[k,i,c]
        # Let b[c] = sum_k V[j,k,c] W[k,i,c]
        # A U = b
        
        U_new = np.zeros((n*m, rank))
        for i in range(n):
            for j in range(m):
                # Construct A and b
                # A is rank x rank
                # A[c, c'] = sum_k V[j*p+k, c] * W[k*n+i, c'] * V[j*p+k, c'] * W[k*n+i, c]
                # This can be computed as:
                # M1 = V[j, :] * W[:, i]  -> shape (p, rank) ? No.
                # V[j*p : (j+1)*p, :] is (p, rank)
                # W[:, i*n : (i+1)*n] is (p, n) -> we need W[k, i] for k in 0..p-1
                # W is (p*n, rank). W[k*n+i, :] is row k*n+i.
                # So we need a matrix Z of shape (p, rank) where Z[k, c] = W[k*n+i, c]
                Z = np.array([W[k*n + i, :] for k in range(p)])
                V_j = V[j*p : (j+1)*p, :]
                
                # A[c, c'] = sum_k Z[k, c] * V_j[k, c'] * Z[k, c'] * V_j[k, c]
                # A = (Z * V_j) .T @ (Z * V_j) ?
                # Let P = Z * V_j (element-wise). P[k, c] = Z[k, c] * V_j[k, c]
                # A[c, c'] = sum_k P[k, c] * P[k, c']
                # So A = P.T @ P
                P = Z * V_j
                A_mat = P.T @ P
                b_vec = np.sum(Z * V_j, axis=0) # sum_k Z[k,c] * V_j[k,c]
                
                # Solve A_mat u = b_vec
                try:
                    u = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    u = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                U_new[i*m + j, :] = u
        return U_new

    def als_step_V(U, W, rank):
        # Sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c] = 1
        # For fixed j, k:
        # sum_c V[j*p+k, c] * (sum_i U[i*m+j, c] * W[k*n+i, c]) = 1
        # Let M[c] = sum_i U[i*m+j, c] * W[k*n+i, c]
        # sum_c V[j*p+k, c] * M[c] = 1
        # A[c, c'] = sum_i U[i*m+j, c] * W[k*n+i, c'] * U[i*m+j, c'] * W[k*n+i, c]
        # b[c] = sum_i U[i*m+j, c] * W[k*n+i, c]
        
        V_new = np.zeros((m*p, rank))
        for j in range(m):
            for k in range(p):
                # U_i = U[i*m+j, :] for i in 0..n-1 -> shape (n, rank)
                U_i = np.array([U[i*m + j, :] for i in range(n)])
                # W_k = W[k*n+i, :] for i in 0..n-1 -> shape (n, rank)
                W_k = np.array([W[k*n + i, :] for i in range(n)])
                
                P = U_i * W_k
                A_mat = P.T @ P
                b_vec = np.sum(P, axis=0)
                
                try:
                    v = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    v = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                V_new[j*p + k, :] = v
        return V_new

    def als_step_W(U, V, rank):
        # Sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
        # For fixed k, i:
        # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1
        # Let M[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        # sum_c W[k*n+i, c] * M[c] = 1
        # A[c, c'] = sum_j U[i*m+j, c] * V[j*p+k, c'] * U[i*m+j, c'] * V[j*p+k, c]
        # b[c] = sum_j U[i*m+j, c] * V[j*p+k, c]
        
        W_new = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                # U_i = U[i*m+j, :] for j in 0..m-1 -> shape (m, rank)
                U_i = np.array([U[i*m + j, :] for j in range(m)])
                # V_k = V[j*p+k, :] for j in 0..m-1 -> shape (m, rank)
                V_k = np.array([V[j*p + k, :] for j in range(m)])
                
                P = U_i * V_k
                A_mat = P.T @ P
                b_vec = np.sum(P, axis=0)
                
                try:
                    w = np.linalg.solve(A_mat, b_vec)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                
                W_new[k*n + i, :] = w
        return W_new

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
            
            # 2. Random sparse
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

        for U, V, W in inits:
            if time.time() > DEADLINE - 10:
                break
            
            for _ in range(max_iter):
                if time.time() > DEADLINE - 10:
                    break
                U = als_step_U(V, W, target_rank)
                V = als_step_V(U, W, target_rank)
                W = als_step_W(U, V, target_rank)
                
                # Check convergence
                if check_exact_full(U, V, W, target_rank, tol=1e-6):
                    # Snap
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
                        
                    # If float is exact, return float (evaluator might accept? No, contract says integer/half-integer)
                    # So we must snap.
        return None

    # Try Rank 17
    decomp_17 = None
    if time.time() < DEADLINE - 20:
        decomp_17 = try_rank_als(17)
    
    if decomp_17:
        U, V, W = decomp_17
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try Rank 16
    decomp_16 = None
    if time.time() < DEADLINE - 20:
        decomp_16 = try_rank_als(16)
        
    if decomp_16:
        U, V, W = decomp_16
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return get_rank_18_solution()

# EVOLVE-BLOCK-END
