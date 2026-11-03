import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6
    
    deadline = time.time() + 140
    
    # Block size
    bs = 3
    
    # We attempt to find a rank 19 decomposition for 3x3 matrix multiplication.
    # If successful, total rank is 7 * 19 = 133.
    # If not, we fall back to rank 27 for 3x3, total rank 7 * 27 = 189.
    
    def find_3x3_decomposition(rank):
        # Target tensor for 3x3 multiplication
        # C_ki = sum_j A_kj B_ji
        # Indices: k (0..2), j (0..2), i (0..2)
        # U index: k*3 + j
        # V index: j*3 + i
        # W index: k*3 + i
        
        def compute_error(params):
            U = params[:9*rank].reshape(9, rank)
            V = params[9*rank:18*rank].reshape(9, rank)
            W = params[18*rank:].reshape(9, rank)
            
            error = 0.0
            # Iterate over all k, i to check C_ki
            for k in range(3):
                for i in range(3):
                    # Target C_ki = sum_j A_kj B_ji
                    # We need to check if the bilinear form matches.
                    # The bilinear form from decomposition is:
                    # Sum_r W[k*3+i, r] * (Sum_j U[k*3+j, r] A_kj) * (Sum_l V[l*3+i, r] B_li)
                    # This must equal Sum_j A_kj B_ji.
                    # This means the matrix of coefficients for A_kj B_li must match delta_jl.
                    
                    # Let's expand the bilinear form:
                    # Sum_r W[k*3+i, r] Sum_j U[k*3+j, r] A_kj Sum_l V[l*3+i, r] B_li
                    # = Sum_j Sum_l ( Sum_r W[k*3+i, r] U[k*3+j, r] V[l*3+i, r] ) A_kj B_li
                    
                    # We need the coefficient of A_kj B_li to be 1 if j==l, else 0.
                    for j in range(3):
                        for l in range(3):
                            coeff = 0.0
                            for r in range(rank):
                                coeff += W[k*3+i, r] * U[k*3+j, r] * V[l*3+i, r]
                            target = 1.0 if j == l else 0.0
                            error += (coeff - target) ** 2
            return error

        # Random initialization
        np.random.seed(42)
        best_error = float('inf')
        best_params = None
        
        # Try a few random starts
        for _ in range(10):
            init_params = np.random.randn(27 * rank) * 0.1
            res = minimize(compute_error, init_params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
        if best_error < 1e-6:
            U = best_params[:9*rank].reshape(9, rank)
            V = best_params[9*rank:18*rank].reshape(9, rank)
            W = best_params[18*rank:].reshape(9, rank)
            
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Check error with rounded values
            err_rounded = compute_error(np.concatenate([U_rounded.flatten(), V_rounded.flatten(), W_rounded.flatten()]))
            if err_rounded < 1e-6:
                return U_rounded, V_rounded, W_rounded
            else:
                return U, V, W
        else:
            return None

    # Try to find rank 19 decomposition
    dec_3x3 = find_3x3_decomposition(19)
    
    if dec_3x3 is not None:
        U_3, V_3, W_3 = dec_3x3
        rank_3 = 19
    else:
        # Fallback to schoolbook 3x3 (Rank 27)
        rank_3 = 27
        U_3 = np.zeros((9, 27))
        V_3 = np.zeros((9, 27))
        W_3 = np.zeros((9, 27))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U_3[i * 3 + j, r] = 1.0
                    V_3[j * 3 + k, r] = 1.0
                    W_3[k * 3 + i, r] = 1.0
                    
    rank = 7 * rank_3
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen definitions
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]
    
    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # U update
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        # The local 3x3 multiplication index for A is u, v
                        # In U_3, row is u*3 + v
                        u_local = u * 3 + v
                        for r_3 in range(rank_3):
                            U[u_row][r_idx + r_3] += coeff_a * U_3[u_local, r_3]
                            
                    # V update
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        # In V_3, row is v*3 + j
                        v_local = v * 3 + j
                        for r_3 in range(rank_3):
                            V[v_row][r_idx + r_3] += coeff_b * V_3[v_local, r_3]
                            
                    # W update
                    for c_block_idx in range(4):
                        coeffs = recon_coeffs[c_block_idx]
                        gamma = 0.0
                        for (s_idx, g) in coeffs:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            br = c_block_idx // 2
                            bc = c_block_idx % 2
                            i_glob = br * bs + u
                            j_glob = bc * bs + j
                            w_row = j_glob * n + i_glob
                            # In W_3, row is j*3 + u (since W index is k*n+i, here k=j, i=u)
                            # Wait, in 3x3, C_ji = sum A_jk B_ki.
                            # W_3 index is j*3 + i.
                            # Here i_glob is row of C block, j_glob is col of C block.
                            # So W_3 row is j_glob_local * 3 + i_glob_local?
                            # i_glob = br*bs + u -> u is local row.
                            # j_glob = bc*bs + j -> j is local col.
                            # So W_3 row is j * 3 + u.
                            w_local = j * 3 + u
                            for r_3 in range(rank_3):
                                W[w_row][r_idx + r_3] += gamma * W_3[w_local, r_3]
                                
                    r_idx += rank_3
    
    assert r_idx == rank
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
