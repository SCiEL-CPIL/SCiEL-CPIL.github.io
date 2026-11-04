"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Convention: W[k*n + i, r] contributes to C[i,k]
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # First, build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        """
        a_idx: [a00, a01, a10, a11] indices into U
        b_idx: [b00, b01, b10, b11] indices into V
        w_idx: [w00, w01, w10, w11] indices into W for the 2x2 output block
        """
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
    
    # A is 2x4: indices i*m+j
    a_L = [0, 1, 4, 5]  # a00, a01, a10, a11
    a_R = [2, 3, 6, 7]
    
    # B is 4x6: indices j*p+k
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
    
    # Verify the decomposition
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linear dependencies among the 42 rank-1 terms
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    
    M = np.array(flat_terms).T  # (2304, 42)
    
    best_result = {"U": U, "V": V, "W": W}
    
    # SVD-based null space elimination
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        effective_rank = np.sum(s_vals > tol)
        
        if effective_rank < rank:
            null_space = Vt_svd[effective_rank:, :]
            num_deps = null_space.shape[0]
            
            new_U = U_np.copy()
            new_V = V_np.copy()
            new_W = W_np.copy()
            eliminated = set()
            
            for _ in range(num_deps):
                if time.time() > deadline - 15:
                    break
                    
                remaining = [c for c in range(rank) if c not in eliminated]
                if len(remaining) <= 1:
                    break
                
                M_sub = M[:, remaining]
                
                try:
                    U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
                    eff_rank_sub = np.sum(s_sub > tol)
                    if eff_rank_sub >= len(remaining):
                        break
                    null_sub = Vt_sub[eff_rank_sub:, :]
                except:
                    break
                
                for null_vec in null_sub:
                    if time.time() > deadline - 10:
                        break
                    candidates = [c for c in remaining if c not in eliminated]
                    if not candidates:
                        break
                    max_idx = max(candidates, key=lambda c: abs(null_vec[c]))
                    max_val = abs(null_vec[max_idx])
                    if max_val < 1e-8:
                        continue
                    
                    for c in candidates:
                        if c == max_idx:
                            continue
                        coeff = -null_vec[c] / null_vec[max_idx]
                        if abs(coeff) > 1e-10:
                            new_U[:, c] += coeff * new_U[:, max_idx]
                            new_V[:, c] += coeff * new_V[:, max_idx]
                            new_W[:, c] += coeff * new_W[:, max_idx]
                    eliminated.add(max_idx)
            
            if eliminated:
                keep = [c for c in range(rank) if c not in eliminated]
                new_rank = len(keep)
                final_U = new_U[:, keep]
                final_V = new_V[:, keep]
                final_W = new_W[:, keep]
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for r in range(new_rank):
                                total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                            if abs(total - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    def to_fraction_list(arr):
                        result = []
                        for row in arr:
                            result.append([])
                            for val in row:
                                frac = round(val * 2) / 2
                                if abs(frac - round(frac)) < 1e-10:
                                    frac = int(round(frac))
                                result[-1].append(frac)
                        return result
                    
                    best_result = {
                        "U": to_fraction_list(final_U),
                        "V": to_fraction_list(final_V),
                        "W": to_fraction_list(final_W)
                    }
    except Exception:
        pass
    
    # Exact rational arithmetic approach for dependencies
    try:
        U_frac = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
        V_frac = [[Fraction(V[i][r]) for r in range(rank)] for i in range(m*p)]
        W_frac = [[Fraction(W[i][r]) for r in range(rank)] for i in range(p*n)]
        
        dim = n * m * m * p * p * n
        terms = []
        for r in range(rank):
            term = [Fraction(0)] * dim
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        idx = i * (m*p) * (p*n) + j * (p*n) + k
                        term[idx] = U_frac[i][r] * V_frac[j][r] * W_frac[k][r]
            terms.append(term)
        
        if time.time() < deadline - 30:
            matrix_T = [[terms[r][c] for r in range(rank)] for c in range(dim)]
            
            pivot_cols = []
            current_row = 0
            for col in range(rank):
                if current_row >= dim:
                    break
                pivot = None
                for row in range(current_row, dim):
                    if matrix_T[row][col] != 0:
                        pivot = row
                        break
                if pivot is None:
                    continue
                matrix_T[current_row], matrix_T[pivot] = matrix_T[pivot], matrix_T[current_row]
                for row in range(current_row + 1, dim):
                    if matrix_T[row][col] != 0:
                        factor = matrix_T[row][col] / matrix_T[current_row][col]
                        for c in range(col, rank):
                            matrix_T[row][c] -= factor * matrix_T[current_row][c]
                pivot_cols.append(col)
                current_row += 1
            
            effective_rank_exact = len(pivot_cols)
            if effective_rank_exact < rank:
                null_vecs = []
                free_cols = [c for c in range(rank) if c not in pivot_cols]
                for free_col in free_cols:
                    if time.time() > deadline - 15:
                        break
                    null_vec = [Fraction(0)] * rank
                    null_vec[free_col] = Fraction(1)
                    for i in range(len(pivot_cols) - 1, -1, -1):
                        col = pivot_cols[i]
                        total = Fraction(0)
                        for c in range(col + 1, rank):
                            if matrix_T[i][c] != 0:
                                total += matrix_T[i][c] * null_vec[c]
                        null_vec[col] = -total / matrix_T[i][col]
                    null_vecs.append(null_vec)
                
                new_U = U_np.copy()
                new_V = V_np.copy()
                new_W = W_np.copy()
                eliminated = set()
                
                for null_vec in null_vecs:
                    if time.time() > deadline - 10:
                        break
                    candidates = [c for c in range(rank) if c not in eliminated]
                    if not candidates:
                        break
                    null_vec_float = [float(nv) for nv in null_vec]
                    max_idx = max(candidates, key=lambda c: abs(null_vec_float[c]))
                    max_val = abs(null_vec_float[max_idx])
                    if max_val < 1e-8:
                        continue
                    
                    for c in candidates:
                        if c == max_idx:
                            continue
                        coeff = -float(null_vec[c]) / float(null_vec[max_idx])
                        if abs(coeff) > 1e-10:
                            new_U[:, c] += coeff * new_U[:, max_idx]
                            new_V[:, c] += coeff * new_V[:, max_idx]
                            new_W[:, c] += coeff * new_W[:, max_idx]
                    eliminated.add(max_idx)
                
                if eliminated:
                    keep = [c for c in range(rank) if c not in eliminated]
                    new_rank = len(keep)
                    final_U = new_U[:, keep]
                    final_V = new_V[:, keep]
                    final_W = new_W[:, keep]
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                total = 0.0
                                for r in range(new_rank):
                                    total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                                if abs(total - 1.0) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        def to_fraction_list(arr):
                            result = []
                            for row in arr:
                                result.append([])
                                for val in row:
                                    frac = round(val * 2) / 2
                                    if abs(frac - round(frac)) < 1e-10:
                                        frac = int(round(frac))
                                    result[-1].append(frac)
                            return result
                        
                        new_result = {
                            "U": to_fraction_list(final_U),
                            "V": to_fraction_list(final_V),
                            "W": to_fraction_list(final_W)
                        }
                        # Compare with best result
                        if len(final_U[0]) < len(best_result["U"][0]):
                            best_result = new_result
    except Exception:
        pass
    
    return best_result
# EVOLVE-BLOCK-END
