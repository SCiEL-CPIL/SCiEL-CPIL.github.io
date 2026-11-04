"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the matrix multiplication tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Convention: W[k*n + i, r] contributes to C[i,k]
    # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
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
    
    # Convert to float arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor matrix: column r = flattened (u_r ⊗ v_r ⊗ w_r)
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())
    M = np.array(flat_terms).T  # shape (2304, 42)
    
    # Best result so far
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    def to_list(arr):
        result = []
        for row in arr:
            result.append([])
            for val in row:
                # Round to nearest half-integer
                frac = round(val * 2) / 2
                if abs(frac - round(frac)) < 1e-10:
                    frac = int(round(frac))
                result[-1].append(frac)
        return result
    
    def verify_float(final_U, final_V, final_W, new_rank, tol=1e-7):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(new_rank):
                        total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def eliminate_columns(U_in, V_in, W_in, M_in, max_eliminations, deadline, tol=1e-9, random_seed=None):
        """Eliminate dependent columns using SVD null space."""
        new_U = U_in.copy()
        new_V = V_in.copy()
        new_W = W_in.copy()
        eliminated = set()
        current_rank = M_in.shape[1]
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        for iteration in range(max_eliminations):
            if time.time() > deadline:
                break
            
            remaining = [c for c in range(current_rank) if c not in eliminated]
            if len(remaining) <= 1:
                break
            
            M_sub = M_in[:, remaining]
            
            try:
                U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            
            eff_rank_sub = int(np.sum(s_sub > tol))
            
            if eff_rank_sub >= len(remaining):
                break
            
            null_sub = Vt_sub[eff_rank_sub:, :]
            
            # Pick the best column to eliminate
            best_col = None
            best_score = -1
            best_null_idx = 0
            
            for nv_idx in range(null_sub.shape[0]):
                nv = null_sub[nv_idx]
                for local_c, global_c in enumerate(remaining):
                    score = abs(nv[local_c])
                    if random_seed is not None:
                        score += np.random.random() * 0.05
                    if score > best_score and abs(nv[local_c]) > 1e-8:
                        best_score = score
                        best_col = global_c
                        best_null_idx = nv_idx
            
            if best_col is None:
                break
            
            nv = null_sub[best_null_idx]
            local_best = remaining.index(best_col)
            pivot_val = nv[local_best]
            
            if abs(pivot_val) < 1e-10:
                continue
            
            for local_c, global_c in enumerate(remaining):
                if global_c == best_col:
                    continue
                coeff = -nv[local_c] / pivot_val
                if abs(coeff) > 1e-10:
                    new_U[:, global_c] += coeff * new_U[:, best_col]
                    new_V[:, global_c] += coeff * new_V[:, best_col]
                    new_W[:, global_c] += coeff * new_W[:, best_col]
            
            eliminated.add(best_col)
        
        if not eliminated:
            return None, None, None, 0
        
        keep = [c for c in range(current_rank) if c not in eliminated]
        new_rank = len(keep)
        return new_U[:, keep], new_V[:, keep], new_W[:, keep], new_rank
    
    # Strategy 1: Systematic SVD-based elimination with multiple tolerances
    for tol in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
        if time.time() > deadline - 30:
            break
        
        elim_U, elim_V, elim_W, elim_rank = eliminate_columns(
            U_np, V_np, W_np, M, 10, deadline - 30, tol=tol
        )
        
        if elim_U is not None and elim_rank < best_rank:
            if verify_float(elim_U, elim_V, elim_W, elim_rank, tol=1e-7):
                best_rank = elim_rank
                best_result = {
                    "U": to_list(elim_U),
                    "V": to_list(elim_V),
                    "W": to_list(elim_W)
                }
    
    # Strategy 2: Random restarts with different seeds
    if time.time() < deadline - 25:
        for seed in range(50):
            if time.time() > deadline - 25:
                break
            
            elim_U, elim_V, elim_W, elim_rank = eliminate_columns(
                U_np, V_np, W_np, M, 10, deadline - 25, tol=1e-9, random_seed=seed
            )
            
            if elim_U is not None and elim_rank < best_rank:
                if verify_float(elim_U, elim_V, elim_W, elim_rank, tol=1e-7):
                    best_rank = elim_rank
                    best_result = {
                        "U": to_list(elim_U),
                        "V": to_list(elim_V),
                        "W": to_list(elim_W)
                    }
    
    # Strategy 3: Try to find a rank-40 decomposition by exploiting D6 symmetry
    # The 6 columns can be grouped into orbits under dihedral group D6
    # Try to construct a decomposition that respects this symmetry
    if time.time() < deadline - 20:
        # Alternative blocking: use 2x2 blocks in A, 2x2 blocks in B (3 blocks)
        # This gives 6 blocks of 2x2x2, same as before
        # But try a different Strassen variant that might have more dependencies
        
        # Try: Can we find a rank-40 by combining some Strassen terms across blocks?
        # The key is that terms from different blocks might be linearly dependent
        # when viewed in the full tensor space
        
        # Use exact rational arithmetic to check for dependencies
        from sympy import Matrix, Rational
        
        try:
            # Build exact tensor matrix
            U_exact = Matrix(U)
            V_exact = Matrix(V)
            W_exact = Matrix(W)
            
            # Compute the rank of the tensor matrix exactly
            # This is expensive, so only try if we have time
            if time.time() < deadline - 40:
                # Instead of full exact rank, try to find specific dependencies
                # Check if any 43x43 minor is zero (would prove rank <= 42)
                # Or check if the current 42 terms span a space of dimension < 42
                
                # Use floating point with very high precision
                M_high = np.array(flat_terms).T
                U_h, s_h, Vt_h = np.linalg.svd(M_high, full_matrices=False)
                
                # Count singular values above various thresholds
                for thresh in [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10]:
                    eff_r = int(np.sum(s_h > thresh))
                    if eff_r < best_rank:
                        # Found potential dependency, try to exploit it
                        null_h = Vt_h[eff_r:, :]
                        elim_U, elim_V, elim_W, elim_rank = eliminate_columns(
                            U_np, V_np, W_np, M, best_rank - eff_r + 2, deadline - 20, tol=thresh
                        )
                        if elim_U is not None and elim_rank < best_rank:
                            if verify_float(elim_U, elim_V, elim_W, elim_rank, tol=1e-7):
                                best_rank = elim_rank
                                best_result = {
                                    "U": to_list(elim_U),
                                    "V": to_list(elim_V),
                                    "W": to_list(elim_W)
                                }
        except Exception:
            pass
    
    # Strategy 4: Try a completely different initial decomposition
    # Use the fact that 4 = 2*2 and 6 = 3*2 to create a 2x3 grid of 2x2 blocks
    # Then try to find a lower-rank decomposition of each 2x2x2 block combined
    
    if time.time() < deadline - 15:
        # Try Winograd's approach: use the structure of Z_2 x Z_3 for the 6-dim
        # This might allow rank reduction
        
        # For now, try one more aggressive elimination pass
        for tol in [1e-10, 1e-11, 1e-12]:
            if time.time() > deadline - 15:
                break
            
            elim_U, elim_V, elim_W, elim_rank = eliminate_columns(
                U_np, V_np, W_np, M, 15, deadline - 15, tol=tol
            )
            
            if elim_U is not None and elim_rank < best_rank:
                if verify_float(elim_U, elim_V, elim_W, elim_rank, tol=1e-7):
                    best_rank = elim_rank
                    best_result = {
                        "U": to_list(elim_U),
                        "V": to_list(elim_V),
                        "W": to_list(elim_W)
                    }
    
    return best_result
# EVOLVE-BLOCK-END
