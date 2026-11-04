"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is a valid product index."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _discrete_coordinate_descent(n, m, p, rank, T, max_passes=1000, time_limit=None):
    """
    Discrete coordinate descent for finding a rank-rank decomposition with half-integer entries.
    """
    start_time = time.time()
    n_vars = (n * m + m * p + p * n) * rank
    # Values to try
    vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    # Initialize
    U = np.random.choice(vals, size=(n * m, rank))
    V = np.random.choice(vals, size=(m * p, rank))
    W = np.random.choice(vals, size=(p * n, rank))
    
    # Compute initial residual
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    R = T - recon
    E = np.sum(R**2)
    
    # Norms of rows of V and W for quick Q calculation
    # Q_ar = (delta_u)^2 * ||v_r||^2 * ||w_r||^2
    # We need ||v_r||^2 for each r and ||w_r||^2 for each r
    v_norms_sq = np.sum(V**2, axis=0) # shape (rank,)
    w_norms_sq = np.sum(W**2, axis=0) # shape (rank,)
    
    for _ in range(max_passes):
        if time_limit and (time.time() - start_time > time_limit):
            break
            
        improved = False
        # Random order of variables
        # We have U (n*m*rank), V (m*p*rank), W (p*n*rank)
        # We can iterate through all of them
        
        # Update U
        for a in range(n * m):
            for r in range(rank):
                # S_ar = sum_{bc} R_{abc} V_{br} W_{cr}
                # This is (R[:, :, :] * V[r, :, None] * W[r, None, :]).sum() ?
                # R shape (A, B, C). V[r] shape (B,). W[r] shape (C,).
                # S_ar = np.einsum('abc,b,c->', R, V[r], W[r])
                S_ar = np.einsum('abc,b,c->', R, V[r], W[r])
                Q_ar = v_norms_sq[r] * w_norms_sq[r]
                
                # E_new = E - 2 * du * S_ar + (du)^2 * Q_ar
                # du = u_new - u_old
                u_old = U[a, r]
                best_val = u_old
                best_e = E
                for v in vals:
                    if v == u_old: continue
                    du = v - u_old
                    e_new = E - 2 * du * S_ar + (du**2) * Q_ar
                    if e_new < best_e:
                        best_e = e_new
                        best_val = v
                        
                if best_val != u_old:
                    U[a, r] = best_val
                    du = best_val - u_old
                    # Update R
                    # R -= du * V[r, :, None] * W[r, None, :]
                    R[a, :, :] -= du * np.outer(V[r], W[r])
                    E = best_e
                    improved = True
                    
        # Update V
        for b in range(m * p):
            for r in range(rank):
                # S_br = sum_{ac} R_{abc} U_{ar} W_{cr}
                S_br = np.einsum('abc,a,c->', R, U[:, r], W[r])
                Q_br = np.sum(U[:, r]**2) * w_norms_sq[r]
                
                v_old = V[b, r]
                best_val = v_old
                best_e = E
                for v in vals:
                    if v == v_old: continue
                    du = v - v_old
                    e_new = E - 2 * du * S_br + (du**2) * Q_br
                    if e_new < best_e:
                        best_e = e_new
                        best_val = v
                        
                if best_val != v_old:
                    V[b, r] = best_val
                    du = best_val - v_old
                    R[:, b, :] -= du * np.outer(U[:, r], W[r])
                    E = best_e
                    v_norms_sq[r] += 2 * du * v_old + du**2 # Update norm
                    improved = True
                    
        # Update W
        for c in range(p * n):
            for r in range(rank):
                # S_cr = sum_{ab} R_{abc} U_{ar} V_{br}
                S_cr = np.einsum('abc,a,b->', R, U[:, r], V[:, r])
                Q_cr = np.sum(U[:, r]**2) * np.sum(V[:, r]**2)
                
                w_old = W[c, r]
                best_val = w_old
                best_e = E
                for v in vals:
                    if v == w_old: continue
                    du = v - w_old
                    e_new = E - 2 * du * S_cr + (du**2) * Q_cr
                    if e_new < best_e:
                        best_e = e_new
                        best_val = v
                        
                if best_val != w_old:
                    W[c, r] = best_val
                    du = best_val - w_old
                    R[:, :, c] -= du * np.outer(U[:, r], V[:, r])
                    E = best_e
                    w_norms_sq[r] += 2 * du * w_old + du**2 # Update norm
                    improved = True
                    
        if not improved:
            break
            
    if E < 1e-9:
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    total_start = time.time()
    time_budget = 140.0
    
    # Fallback: Schoolbook
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
                
    # Try rank 19
    np.random.seed(42)
    for restart in range(20):
        if time.time() - total_start > time_budget:
            break
            
        result = _discrete_coordinate_descent(n, m, p, 19, T, max_passes=50, time_limit=5.0)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
