import time
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 120.0
    n, m, p = 2, 4, 4

    # Build the target tensor T[a,b,c] = 1 if a=4i+j, b=4j+k, c=2k+i
    T = np.zeros((8, 16, 8), dtype=float)
    for j in range(4):
        for i in range(2):
            for k in range(4):
                a = 4 * i + j
                b = 4 * j + k
                c = 2 * k + i
                T[a, b, c] = 1.0

    # Fallback: The known Strassen-like decomposition of rank 28
    def make_strassen_solution():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_block(col_offset, X_rows, Y_rows, Z_rows):
            # M1
            U[X_rows[0]][col_offset + 0] += 1
            U[X_rows[3]][col_offset + 0] += 1
            V[Y_rows[0]][col_offset + 0] += 1
            V[Y_rows[3]][col_offset + 0] += 1
            W[Z_rows[0]][col_offset + 0] += 1
            W[Z_rows[3]][col_offset + 0] += 1

            # M2
            U[X_rows[2]][col_offset + 1] += 1
            U[X_rows[3]][col_offset + 1] += 1
            V[Y_rows[0]][col_offset + 1] += 1
            W[Z_rows[2]][col_offset + 1] += 1
            W[Z_rows[3]][col_offset + 1] -= 1

            # M3
            U[X_rows[0]][col_offset + 2] += 1
            V[Y_rows[1]][col_offset + 2] += 1
            V[Y_rows[3]][col_offset + 2] -= 1
            W[Z_rows[1]][col_offset + 2] += 1
            W[Z_rows[3]][col_offset + 2] += 1

            # M4
            U[X_rows[3]][col_offset + 3] += 1
            V[Y_rows[2]][col_offset + 3] += 1
            V[Y_rows[0]][col_offset + 3] -= 1
            W[Z_rows[0]][col_offset + 3] += 1
            W[Z_rows[2]][col_offset + 3] += 1

            # M5
            U[X_rows[0]][col_offset + 4] += 1
            U[X_rows[1]][col_offset + 4] += 1
            V[Y_rows[3]][col_offset + 4] += 1
            W[Z_rows[0]][col_offset + 4] -= 1
            W[Z_rows[1]][col_offset + 4] += 1

            # M6
            U[X_rows[2]][col_offset + 5] += 1
            U[X_rows[0]][col_offset + 5] -= 1
            V[Y_rows[0]][col_offset + 5] += 1
            V[Y_rows[1]][col_offset + 5] += 1
            W[Z_rows[3]][col_offset + 5] += 1

            # M7
            U[X_rows[1]][col_offset + 6] += 1
            U[X_rows[3]][col_offset + 6] -= 1
            V[Y_rows[2]][col_offset + 6] += 1
            V[Y_rows[3]][col_offset + 6] += 1
            W[Z_rows[0]][col_offset + 6] += 1

        # Block 1: A_L B_TL -> C_LL
        X1 = [0, 1, 4, 5]
        Y1 = [0, 1, 4, 5]
        Z1 = [0, 2, 1, 3]
        add_block(0, X1, Y1, Z1)

        # Block 2: A_R B_BL -> C_LR
        X2 = [2, 3, 6, 7]
        Y2 = [8, 9, 12, 13]
        Z2 = [0, 2, 1, 3]
        add_block(7, X2, Y2, Z2)

        # Block 3: A_L B_TR -> C_RL
        X3 = [0, 1, 4, 5]
        Y3 = [2, 3, 6, 7]
        Z3 = [4, 6, 5, 7]
        add_block(14, X3, Y3, Z3)

        # Block 4: A_R B_BR -> C_RR
        X4 = [2, 3, 6, 7]
        Y4 = [10, 11, 14, 15]
        Z4 = [4, 6, 5, 7]
        add_block(21, X4, Y4, Z4)

        return {"U": U, "V": V, "W": W}

    def verify_solution(U_list, V_list, W_list, tol=1e-9):
        """Verify that the solution exactly decomposes T."""
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        rank = U_arr.shape[1]
        T_check = np.zeros((8, 16, 8))
        for r in range(rank):
            T_check += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
        return np.allclose(T_check, T, atol=tol)

    # Greedy Search for lower rank with improved reconstruction
    def try_greedy_search(max_steps=25, n_restarts=8):
        rng = np.random.RandomState(42)
        
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                return None
            
            R = T.copy()
            components = []
            
            # For restarts > 0, add small random perturbation to escape local minima
            if restart > 0:
                R += rng.randn(*R.shape) * (0.01 + 0.005 * restart)
            
            for step in range(max_steps):
                if time.time() > deadline - 15:
                    return None
                
                # 1. Find w (mode 3) - use top singular vector
                R3 = R.reshape(8 * 16, 8)
                try:
                    _, _, Vh = np.linalg.svd(R3)
                    w = Vh[0].copy()
                except:
                    return None
                
                # 2. Contract to get R' (8, 16)
                R_prime = np.einsum('abc,c->ab', R, w)
                
                # 3. Find v (mode 2)
                try:
                    _, _, Vh2 = np.linalg.svd(R_prime)
                    v = Vh2[0].copy()
                except:
                    return None
                
                # 4. Contract to get R'' (8)
                R_double_prime = np.einsum('ab,b->a', R_prime, v)
                
                # 5. Find u (mode 1)
                u_norm = np.linalg.norm(R_double_prime)
                if u_norm < 1e-10:
                    break
                u = R_double_prime / u_norm
                
                # 6. Calculate alpha
                inner = np.einsum('abc,a,b,c->', R, u, v, w)
                norm_sq = (u_norm**2) * (np.linalg.norm(v)**2) * (np.linalg.norm(w)**2)
                if norm_sq < 1e-10:
                    break
                alpha = inner / norm_sq
                
                # 7. Update R
                R -= alpha * np.einsum('a,b,c->abc', u, v, w)
                
                # Store component
                components.append((u.copy(), v.copy(), w.copy(), alpha))
                
                # Check residual
                if np.linalg.norm(R) < 1e-8:
                    # Found a decomposition of rank step+1
                    result = reconstruct_integer_solution(components)
                    if result is not None:
                        return result
                    
        return None

    def reconstruct_integer_solution(components):
        """Attempt to find integer/half-integer representation of the greedy solution."""
        rank = len(components)
        if rank == 0:
            return None
        
        U = np.zeros((8, rank))
        V = np.zeros((16, rank))
        W = np.zeros((8, rank))
        
        for idx, (u, v, w, alpha) in enumerate(components):
            # Try multiple scale factors to find half-integer representation
            best_err = float('inf')
            best_U = None
            best_V = None
            best_W = None
            
            for su in [1, 2, 4, 8, 16]:
                for sv in [1, 2, 4, 8, 16]:
                    for sw in [1, 2, 4, 8, 16]:
                        u_s = u / su
                        v_s = v / sv
                        w_s = alpha * w / sw
                        
                        # Check closeness to half-integers
                        u_close = np.allclose(u_s * 2, np.round(u_s * 2), atol=0.05)
                        v_close = np.allclose(v_s * 2, np.round(v_s * 2), atol=0.05)
                        w_close = np.allclose(w_s * 2, np.round(w_s * 2), atol=0.05)
                        
                        if u_close and v_close and w_close:
                            U_cand = np.round(u_s * 2) / 2
                            V_cand = np.round(v_s * 2) / 2
                            W_cand = np.round(w_s * 2) / 2
                            
                            # Compute reconstruction error
                            T_cand = np.einsum('a,b,c->abc', U_cand, V_cand, W_cand)
                            T_target = alpha * np.einsum('a,b,c->abc', u, v, w)
                            err = np.linalg.norm(T_cand - T_target)
                            
                            if err < best_err:
                                best_err = err
                                best_U = U_cand
                                best_V = V_cand
                                best_W = W_cand
            
            if best_U is not None:
                U[:, idx] = best_U
                V[:, idx] = best_V
                W[:, idx] = best_W
            else:
                # Fallback: just round
                U[:, idx] = np.round(u * 2) / 2
                V[:, idx] = np.round(v * 2) / 2
                W[:, idx] = np.round(alpha * w * 2) / 2
        
        # Verify the full decomposition
        if not verify_solution(U.tolist(), V.tolist(), W.tolist(), tol=0.01):
            return None
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try greedy search
    result = try_greedy_search()
    
    if result is not None:
        return result
    else:
        return make_strassen_solution()
