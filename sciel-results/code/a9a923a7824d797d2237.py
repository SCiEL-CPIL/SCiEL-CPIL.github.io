import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 16 decomposition
    rank = 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # col: column index in U, V, W
    
    def add_term(u_idx, v_idx, w_idx, col):
        U[u_idx, col] = 1
        V[v_idx, col] = 1
        W[w_idx, col] = 1

    # Part 1: A1 (cols 0,1) * B_top (rows 0,1)
    # Sub-part 1a: B_top_left (cols 0,1)
    # Terms:
    # 0: A(0,0) B(0,0) -> C(0,0)
    add_term(0*4+0, 0*4+0, 0*2+0, 0)
    # 1: A(0,1) B(1,0) -> C(0,0)
    add_term(0*4+1, 1*4+0, 0*2+0, 1)
    # 2: A(0,0) B(0,1) -> C(0,1)
    add_term(0*4+0, 0*4+1, 1*2+0, 2)
    # 3: A(0,1) B(1,1) -> C(0,1)
    add_term(0*4+1, 1*4+1, 1*2+0, 3)
    # 4: A(1,0) B(0,0) -> C(1,0)
    add_term(1*4+0, 0*4+0, 0*2+1, 4)
    # 5: A(1,1) B(1,0) -> C(1,0)
    add_term(1*4+1, 1*4+0, 0*2+1, 5)
    # 6: A(1,0) B(0,1) -> C(1,1)
    add_term(1*4+0, 0*4+1, 1*2+1, 6)
    # 7: A(1,1) B(1,1) -> C(1,1)
    add_term(1*4+1, 1*4+1, 1*2+1, 7)
    
    # Sub-part 1b: B_top_right (cols 2,3)
    # 8: A(0,0) B(0,2) -> C(0,2)
    add_term(0*4+0, 0*4+2, 2*2+0, 8)
    # 9: A(0,1) B(1,2) -> C(0,2)
    add_term(0*4+1, 1*4+2, 2*2+0, 9)
    # 10: A(0,0) B(0,3) -> C(0,3)
    add_term(0*4+0, 0*4+3, 3*2+0, 10)
    # 11: A(0,1) B(1,3) -> C(0,3)
    add_term(0*4+1, 1*4+3, 3*2+0, 11)
    # 12: A(1,0) B(0,2) -> C(1,2)
    add_term(1*4+0, 0*4+2, 2*2+1, 12)
    # 13: A(1,1) B(1,2) -> C(1,2)
    add_term(1*4+1, 1*4+2, 2*2+1, 13)
    # 14: A(1,0) B(0,3) -> C(1,3)
    add_term(1*4+0, 0*4+3, 3*2+1, 14)
    # 15: A(1,1) B(1,3) -> C(1,3)
    add_term(1*4+1, 1*4+3, 3*2+1, 15)
    
    # Wait, I only used 16 columns.
    # Part 2: A2 (cols 2,3) * B_bottom (rows 2,3)
    # This requires another 16 terms?
    # No, the rank is the number of columns.
    # If I add more terms, the rank increases.
    # My previous logic said rank 16 is the total.
    # But I listed 16 terms for Part 1.
    # Part 2 also has 16 terms?
    # No, Part 1 is A1 * B_top.
    # A1 is 2x2. B_top is 2x4.
    # This is 8 terms.
    # I listed 16 terms above for Part 1?
    # Let's re-count.
    # Sub-part 1a: 8 terms.
    # Sub-part 1b: 8 terms.
    # Total 16 terms for Part 1.
    # But Part 1 is only half of the multiplication.
    # C = A1 B_top + A2 B_bottom.
    # So I need another 16 terms for Part 2?
    # That would make rank 32.
    
    # Let's re-evaluate mu(2,2,4).
    # Is it 8?
    # Standard 2x2x2 is rank 4.
    # 2x2x4 is two 2x2x2s.
    # If they are independent, rank 8.
    # So A1 B_top is rank 8.
    # A2 B_bottom is rank 8.
    # Total rank 16.
    
    # So my construction above for Part 1 used 16 terms?
    # Let's check the indices.
    # Term 0: A(0,0) B(0,0)
    # Term 1: A(0,1) B(1,0)
    # Term 2: A(0,0) B(0,1)
    # Term 3: A(0,1) B(1,1)
    # Term 4: A(1,0) B(0,0)
    # Term 5: A(1,1) B(1,0)
    # Term 6: A(1,0) B(0,1)
    # Term 7: A(1,1) B(1,1)
    # These 8 terms compute A1 @ B_top[:, 0:2].
    
    # Term 8: A(0,0) B(0,2)
    # Term 9: A(0,1) B(1,2)
    # Term 10: A(0,0) B(0,3)
    # Term 11: A(0,1) B(1,3)
    # Term 12: A(1,0) B(0,2)
    # Term 13: A(1,1) B(1,2)
    # Term 14: A(1,0) B(0,3)
    # Term 15: A(1,1) B(1,3)
    # These 8 terms compute A1 @ B_top[:, 2:4].
    
    # So Part 1 (A1 @ B_top) is done in 16 terms.
    # Part 2 (A2 @ B_bottom) requires another 16 terms.
    # Total rank 32.
    
    # This is not an improvement over 28.
    
    # Can we do better?
    # Strassen on 2x2x2 is rank 7.
    # If we use Strassen for 2x2x2 blocks:
    # A1 @ B_top[:, 0:2] uses 7 terms.
    # A1 @ B_top[:, 2:4] uses 7 terms.
    # A2 @ B_bottom[:, 0:2] uses 7 terms.
    # A2 @ B_bottom[:, 2:4] uses 7 terms.
    # Total 28 terms.
    # This matches the seed program.
    
    # Is there a way to share terms between A1 and A2?
    # Or between the two column blocks of B?
    # A1 is shared in the two column blocks of B_top.
    # A2 is shared in the two column blocks of B_bottom.
    # But A1 and A2 are disjoint.
    
    # What if we view the whole thing as 2x4x4?
    # Can we use a 4x4x4 decomposition and project?
    # Rank of 4x4x4 is 49 (Strassen) or 64 (Standard).
    # Projecting to 2x4x4 might reduce rank?
    # Unlikely to go below 28 easily.
    
    # Let's check if rank 16 is possible.
    # Lower bound for 2x4x4.
    # If mu(2,4,4) = 16, then we need a different structure.
    # Maybe not splitting into 4 blocks.
    
    # Let's try to find a rank 16 decomposition via search.
    # The seed program tried L-BFGS-B and failed.
    # Maybe the initialization was bad.
    
    # Let's try a simple random search with rounding.
    
    def check_decomposition(U, V, W, rank):
        if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
            return False
        if not np.allclose(2 * U, np.round(2 * U), atol=1e-5): return False
        if not np.allclose(2 * V, np.round(2 * V), atol=1e-5): return False
        if not np.allclose(2 * W, np.round(2 * W), atol=1e-5): return False
        
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C = A @ B
        
        Cf = np.zeros((n, p))
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        T_rec = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U_r[:, :, r]
            v_r = V_r[:, :, r]
            w_r = W_r[:, :, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_rec[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
                        
        T_target = np.ones((n, m, p))
        if not np.allclose(T_rec, T_target, atol=1e-5):
            return False
        return True

    # Try rank 16
    # Use a more robust search
    # Since the space is small, we can try to solve the system of equations.
    # There are n*m*p = 32 equations.
    # Variables: U, V, W.
    # Symmetry: We can fix some variables to 1 or 0 to break gauge symmetry.
    
    # Let's stick to the Strassen 28 if we can't find 16 quickly.
    # But 28 is the seed. We need to beat it.
    # Score 1.0 is seed.
    # If we return 28, score is 1.0.
    # We need score > 1.0.
    # So we need rank < 28.
    
    # Let's try rank 24, 20, 16.
    
    # I will implement a simple local search starting from Strassen 28 and trying to merge terms?
    # No, that's hard.
    
    # Let's try the L-BFGS-B again with better initialization.
    # Initialize with the Strassen 28 solution, padded with zeros to rank 28.
    # Then try to reduce rank?
    # No, L-BFGS-B minimizes error for a fixed rank.
    
    # Let's try rank 16 with random restarts.
    
    def solve_rank(rank):
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        total_size = size_u + size_v + size_w
        
        def objective(params):
            U_flat = params[:size_u].reshape(n, m, rank)
            V_flat = params[size_u:size_u+size_v].reshape(m, p, rank)
            W_flat = params[size_u+size_v:].reshape(p, n, rank)
            
            U_b = U_flat[:, :, np.newaxis, np.newaxis, :]
            V_b = V_flat[np.newaxis, :, :, np.newaxis, :]
            W_b = W_flat[np.newaxis, np.newaxis, :, :, :]
            
            prod = U_b * V_b * W_b
            S = np.sum(prod, axis=4)
            idx = np.arange(n)
            S_diag = S[idx, :, :, idx]
            err = np.sum((S_diag - 1.0) ** 2)
            return err

        best_fun = np.inf
        best_params = None
        
        for restart in range(50):
            if time.time() > 140:
                break
            np.random.seed(rank * 1000 + restart)
            params = np.random.randn(total_size) * 0.5
            
            try:
                from scipy.optimize import minimize
                res = minimize(objective, params, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-15})
                
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_params = res.x
                    
                if best_fun < 1e-10:
                    break
                    
            except Exception:
                pass
        
        if best_params is not None and best_fun < 1e-8:
            U_opt = best_params[:size_u].reshape(n, m, rank)
            V_opt = best_params[size_u:size_u+size_v].reshape(m, p, rank)
            W_opt = best_params[size_u+size_v:].reshape(p, n, rank)
            
            U_int = np.round(2 * U_opt) / 2.0
            V_int = np.round(2 * V_opt) / 2.0
            W_int = np.round(2 * W_opt) / 2.0
            
            if check_decomposition(U_int, V_int, W_int, rank):
                return {"U": U_int.reshape(n*m, rank).tolist(), 
                        "V": V_int.reshape(m*p, rank).tolist(), 
                        "W": W_int.reshape(p*n, rank).tolist()}
            
        return None

    # Try rank 16
    result = solve_rank(16)
    if result:
        return result
        
    # Try rank 20
    result = solve_rank(20)
    if result:
        return result
        
    # Fallback to Strassen 28
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    def apply_strassen(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_terms):
            col = start_col + t_idx
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_offset * 2 + c)
                U[g_idx, col] += coeff
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = b_row_offset * 2 + r
                g_col = b_col_offset * 2 + c
                g_idx = g_row * p + g_col
                V[g_idx, col] += coeff
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_row = r
                g_col = c_col_offset * 2 + c
                g_idx = g_col * n + g_row
                W[g_idx, col] += coeff

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    
    if not np.allclose(C, Cf):
        print("Verification Failed")
    else:
        print("Verification Passed")
        
    print("Rank:", U.shape[1])
    print("Time taken:", time.time() - start_time)
