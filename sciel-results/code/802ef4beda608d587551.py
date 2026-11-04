import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    def verify_float(Uf, Vf, Wf, rnk, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rnk):
                        total += Uf[i*m+j, r] * Vf[j*p+k, r] * Wf[k*n+i][r] if False else Vf[j*p+k, r] * Wf[k*n+i, r] * Uf[i*m+j, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Build the tensor matrix M (48 x rank)
    def build_M(Uf, Vf, Wf, rnk):
        flat_terms = []
        for r in range(rnk):
            term = np.outer(np.outer(Uf[:, r], Vf[:, r]), Wf[:, r])
            flat_terms.append(term.flatten())
        return np.array(flat_terms).T

    M = build_M(U_np, V_np, W_np, rank)
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    def to_list(arr):
        result = []
        for row in arr:
            result.append([])
            for val in row:
                frac = round(val * 2) / 2
                if abs(frac - round(frac)) < 1e-10:
                    frac = int(round(frac))
                result[-1].append(frac)
        return result

    # Greedy elimination to reduce rank to 36
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    current_rank = rank
    current_M = M.copy()
    
    target_rank = 36
    
    for step in range(current_rank - target_rank):
        if time.time() > deadline - 10:
            break
            
        eliminated_any = False
        # Try to eliminate each column
        for col_to_elim in range(current_rank):
            if time.time() > deadline - 10:
                break
                
            if col_to_elim >= current_rank:
                continue
                
            remaining = [c for c in range(current_rank) if c != col_to_elim]
            if len(remaining) == 0:
                continue
                
            M_sub = current_M[:, remaining]
            M_col = current_M[:, col_to_elim]
            
            # Solve M_sub * x = M_col
            try:
                coeffs, residuals, rank_sub, singular_vals = np.linalg.lstsq(M_sub, M_col, rcond=1e-12)
                # Check residual
                residual = np.linalg.norm(M_sub @ coeffs - M_col)
                if residual < 1e-8:
                    # Apply elimination
                    new_U = current_U.copy()
                    new_V = current_V.copy()
                    new_W = current_W.copy()
                    
                    for local_c, global_c in enumerate(remaining):
                        if abs(coeffs[local_c]) > 1e-10:
                            new_U[:, global_c] += coeffs[local_c] * new_U[:, col_to_elim]
                            new_V[:, global_c] += coeffs[local_c] * new_V[:, col_to_elim]
                            new_W[:, global_c] += coeffs[local_c] * new_W[:, col_to_elim]
                    
                    # Remove column
                    final_U = new_U[:, remaining]
                    final_V = new_V[:, remaining]
                    final_W = new_W[:, remaining]
                    new_rank = current_rank - 1
                    
                    if verify_float(final_U, final_V, final_W, new_rank, tol=1e-7):
                        current_U = final_U
                        current_V = final_V
                        current_W = final_W
                        current_rank = new_rank
                        current_M = build_M(current_U, current_V, current_W, current_rank)
                        best_rank = current_rank
                        best_result = {
                            "U": to_list(current_U),
                            "V": to_list(current_V),
                            "W": to_list(current_W)
                        }
                        eliminated_any = True
                        break # Restart loop with new rank
            except Exception:
                continue
        
        if not eliminated_any:
            break
            
    return best_result
