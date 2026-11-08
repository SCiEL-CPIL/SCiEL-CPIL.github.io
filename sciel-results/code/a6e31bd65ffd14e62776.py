import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 33 solution (current best)
    def get_rank_33():
        rank = 33
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        terms = []
        # Strassen for 2x2
        terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
        terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
        terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
        terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
        terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
        terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
        terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
        # Naive for 2x1x2
        terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
        terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
        terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
        terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
        
        for block in range(3):
            for t_idx, term in enumerate(terms):
                col = block * 11 + t_idx
                for idx, val in term['u'].items():
                    U[idx][col] = val
                for l_idx, val in term['v'].items():
                    j_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = j_local * p + (block * 2 + k_local)
                    V[g_idx][col] = val
                for l_idx, val in term['w'].items():
                    i_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = (block * 2 + k_local) * n + i_local
                    W[g_idx][col] = val
        return {"U": U, "V": V, "W": W}

    def search_rank_r_m232(r):
        """
        Search for rank r decomposition of M_{2,3,2}.
        Returns (U, V, W) lists or None.
        U: r x 6
        V: r x 6
        W: 4 x r
        """
        # Target tensors for M_{2,3,2}
        targets = []
        for ik in range(4):
            M = np.zeros((6, 6))
            if ik == 0: # c0 = a0*b0 + a1*b2 + a2*b4
                M[0,0] = 1; M[1,2] = 1; M[2,4] = 1
            elif ik == 1: # c1 = a0*b1 + a1*b3 + a2*b5
                M[0,1] = 1; M[1,3] = 1; M[2,5] = 1
            elif ik == 2: # c2 = a3*b0 + a4*b2 + a5*b4
                M[3,0] = 1; M[4,2] = 1; M[5,4] = 1
            elif ik == 3: # c3 = a3*b1 + a4*b3 + a5*b5
                M[3,1] = 1; M[4,3] = 1; M[5,5] = 1
            targets.append(M)
            
        np.random.seed(42 + r)
        best_solution = None
        max_iters = 5000 if r == 10 else 2000
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                return best_solution
                
            # Random initialization with small integers
            U = np.random.choice([-1, 0, 1], size=(r, 6))
            V = np.random.choice([-1, 0, 1], size=(r, 6))
            
            # Compute M_r = u_r v_r^T
            M_r = np.einsum('ri,rj->rij', U, V) # r x 6 x 6
            
            W_valid = True
            W_vals = []
            for ik in range(4):
                A_mat = M_r.reshape(r, 36).T # 36 x r
                b_vec = targets[ik].flatten()
                
                # Solve least squares
                W_real, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check residual
                if np.linalg.norm(A_mat @ W_real - b_vec) > 1e-6:
                    W_valid = False
                    break
                    
                # Check if close to half-integers
                W_half = np.round(W_real * 2) / 2
                if not np.allclose(W_real, W_half, atol=1e-6):
                    W_valid = False
                    break
                W_vals.append(W_half)
                
            if W_valid:
                best_solution = (U.tolist(), V.tolist(), [w.tolist() for w in W_vals])
                # We can return early if we find one, or keep searching for "cleaner" ones?
                # Rank is the score, so any valid rank r is good.
                return best_solution
                
        return best_solution

    # Try rank 10 first
    sol_10 = search_rank_r_m232(10)
    
    if sol_10:
        U_10, V_10, W_10 = sol_10
        rank = 30
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(3):
            for r_idx in range(10):
                col = block * 10 + r_idx
                for i in range(6):
                    U[i][col] = U_10[r_idx][i]
                for j in range(3):
                    for k in range(2):
                        l_idx = j * 2 + k
                        g_idx = j * p + (block * 2 + k)
                        V[g_idx][col] = V_10[r_idx][l_idx]
                for i in range(2):
                    for k in range(2):
                        l_idx = i * 2 + k
                        g_idx = (block * 2 + k) * n + i
                        W[g_idx][col] = W_10[l_idx][r_idx]
                        
        # Verify
        if time.time() < deadline - 5:
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            
            C_computed = np.zeros((n, p))
            for r_idx in range(rank):
                u_val = np.dot(U_arr[:, r_idx], A_flat)
                v_val = np.dot(V_arr[:, r_idx], B_flat)
                for k in range(p):
                    for i in range(n):
                        C_computed[i, k] += W_arr[k * n + i, r_idx] * u_val * v_val
                        
            if np.allclose(C_computed, C_expected, atol=1e-9):
                return {"U": U, "V": V, "W": W}

    # Try rank 9
    sol_9 = search_rank_r_m232(9)
    if sol_9:
        U_9, V_9, W_9 = sol_9
        rank = 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(3):
            for r_idx in range(9):
                col = block * 9 + r_idx
                for i in range(6):
                    U[i][col] = U_9[r_idx][i]
                for j in range(3):
                    for k in range(2):
                        l_idx = j * 2 + k
                        g_idx = j * p + (block * 2 + k)
                        V[g_idx][col] = V_9[r_idx][l_idx]
                for i in range(2):
                    for k in range(2):
                        l_idx = i * 2 + k
                        g_idx = (block * 2 + k) * n + i
                        W[g_idx][col] = W_9[l_idx][r_idx]
                        
        # Verify
        if time.time() < deadline - 5:
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            
            C_computed = np.zeros((n, p))
            for r_idx in range(rank):
                u_val = np.dot(U_arr[:, r_idx], A_flat)
                v_val = np.dot(V_arr[:, r_idx], B_flat)
                for k in range(p):
                    for i in range(n):
                        C_computed[i, k] += W_arr[k * n + i, r_idx] * u_val * v_val
                        
            if np.allclose(C_computed, C_expected, atol=1e-9):
                return {"U": U, "V": V, "W": W}

    return get_rank_33()
# EVOLVE-BLOCK-END
