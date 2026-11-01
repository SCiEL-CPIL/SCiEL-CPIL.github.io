import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [2, 5, 6]
DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # 1. Define the target tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if we are multiplying A[i,j] * B[j,k]
    # The decomposition requires: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: The interface defines U as (n*m, rank), V as (m*p, rank), W as (p*n, rank).
    # Indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # 2. Generate a pool of candidate rank-1 tensors.
    candidates = []
    
    # Type 1: Standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec = np.zeros(n * m * p)
                vec[i * m * p + j * p + k] = 1.0
                u_c = {(i, j): 1}
                v_c = {(j, k): 1}
                w_c = {(k, i): 1}
                candidates.append((vec, u_c, v_c, w_c))
                
    # Type 2: Strassen 2x2x2 blocks
    def get_strassen_terms(io, jo, ko):
        terms = []
        # M1
        u = {(io, jo): 1, (io+1, jo+1): 1}
        v = {(jo, ko): 1, (jo+1, ko+1): 1}
        w = {(ko, io): 1, (ko+1, io+1): 1}
        terms.append((u, v, w))
        
        # M2
        u = {(io+1, jo): 1, (io+1, jo+1): 1}
        v = {(jo, ko): 1}
        w = {(ko, io+1): 1, (ko+1, io+1): -1}
        terms.append((u, v, w))
        
        # M3
        u = {(io, jo): 1}
        v = {(jo, ko+1): 1, (jo+1, ko+1): -1}
        w = {(ko+1, io): 1, (ko+1, io+1): 1}
        terms.append((u, v, w))
        
        # M4
        u = {(io+1, jo+1): 1}
        v = {(jo+1, ko): 1, (jo, ko): -1}
        w = {(ko, io): 1, (ko, io+1): 1}
        terms.append((u, v, w))
        
        # M5
        u = {(io, jo): 1, (io, jo+1): 1}
        v = {(jo+1, ko+1): 1}
        w = {(ko, io): -1, (ko+1, io): 1}
        terms.append((u, v, w))
        
        # M6
        u = {(io+1, jo): 1, (io, jo): -1}
        v = {(jo, ko): 1, (jo, ko+1): 1}
        w = {(ko+1, io+1): 1}
        terms.append((u, v, w))
        
        # M7
        u = {(io, jo+1): 1, (io+1, jo+1): -1}
        v = {(jo+1, ko): 1, (jo+1, ko+1): 1}
        w = {(ko, io): 1}
        terms.append((u, v, w))
        
        return terms

    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in get_strassen_terms(io, jo, ko):
                    vec = np.zeros(n * m * p)
                    for (i, j), uv in u_c.items():
                        for (j2, k), vv in v_c.items():
                            if j == j2:
                                for (k2, i2), wv in w_c.items():
                                    if k == k2 and i == i2:
                                        vec[i * m * p + j * p + k] += uv * vv * wv
                    candidates.append((vec, u_c, v_c, w_c))
    
    # Type 3: Additional structured candidates - "half-Strassen" and other patterns
    # These involve 2x1x1, 1x2x1, 1x1x2 sub-blocks with sign variations
    # Pattern: U has 2 entries, V has 1, W has 1 (or permutations)
    
    # Type 3a: U has two entries in same row, V and W single
    for i in range(n):
        for j1 in range(m):
            for j2 in range(j1 + 1, m):
                for k in range(p):
                    # U[i,j1] + U[i,j2], V[j1,k] or V[j2,k], W[k,i]
                    # Try: (U[i,j1]+U[i,j2]) * V[j1,k] * W[k,i]
                    vec = np.zeros(n * m * p)
                    vec[i * m * p + j1 * p + k] = 1.0
                    vec[i * m * p + j2 * p + k] = 1.0
                    u_c = {(i, j1): 1, (i, j2): 1}
                    v_c = {(j1, k): 1}
                    w_c = {(k, i): 1}
                    candidates.append((vec, u_c, v_c, w_c))
                    
                    # Try: (U[i,j1]-U[i,j2]) * V[j1,k] * W[k,i]
                    vec = np.zeros(n * m * p)
                    vec[i * m * p + j1 * p + k] = 1.0
                    vec[i * m * p + j2 * p + k] = -1.0
                    u_c = {(i, j1): 1, (i, j2): -1}
                    v_c = {(j1, k): 1}
                    w_c = {(k, i): 1}
                    candidates.append((vec, u_c, v_c, w_c))
    
    # Type 3b: V has two entries in same row, U and W single
    for j in range(m):
        for k1 in range(p):
            for k2 in range(k1 + 1, p):
                for i in range(n):
                    vec = np.zeros(n * m * p)
                    vec[i * m * p + j * p + k1] = 1.0
                    vec[i * m * p + j * p + k2] = 1.0
                    u_c = {(i, j): 1}
                    v_c = {(j, k1): 1, (j, k2): 1}
                    w_c = {(k1, i): 1}
                    candidates.append((vec, u_c, v_c, w_c))
                    
                    vec = np.zeros(n * m * p)
                    vec[i * m * p + j * p + k1] = 1.0
                    vec[i * m * p + j * p + k2] = -1.0
                    u_c = {(i, j): 1}
                    v_c = {(j, k1): 1, (j, k2): -1}
                    w_c = {(k1, i): 1}
                    candidates.append((vec, u_c, v_c, w_c))
    
    # Type 3c: W has two entries in same column, U and V single
    for k in range(p):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                for j in range(m):
                    vec = np.zeros(n * m * p)
                    vec[i1 * m * p + j * p + k] = 1.0
                    vec[i2 * m * p + j * p + k] = 1.0
                    u_c = {(i1, j): 1}
                    v_c = {(j, k): 1}
                    w_c = {(k, i1): 1, (k, i2): 1}
                    candidates.append((vec, u_c, v_c, w_c))
                    
                    vec = np.zeros(n * m * p)
                    vec[i1 * m * p + j * p + k] = 1.0
                    vec[i2 * m * p + j * p + k] = -1.0
                    u_c = {(i1, j): 1}
                    v_c = {(j, k): 1}
                    w_c = {(k, i1): 1, (k, i2): -1}
                    candidates.append((vec, u_c, v_c, w_c))
    
    # Type 4: 2x2x1 blocks (Strassen-like but only in i-j plane)
    for io in range(n - 1):
        for jo in range(m - 1):
            for k in range(p):
                # M1: (A[i0,j0]+A[i1,j1]) * B[j0,k] * W[k,i0] + ...
                # Actually let's do proper 2x2x1 Strassen terms
                # Term 1: U[i0,j0]*V[j0,k]*W[k,i0] + U[i1,j1]*V[j1,k]*W[k,i1]
                # This is just two standard terms, not helpful.
                # Let's try: (U[i0,j0]+U[i1,j1]) * (V[j0,k]+V[j1,k]) * W[k,i0]
                # That gives 4 terms, not rank 1.
                # For true rank-1 in 2x2x1: need U, V, W each with at most 2 entries
                # and the product gives a 2x2x1 tensor.
                
                # Pattern: U has 2 entries, V has 2 entries, W has 1 entry
                # U[i0,j0]=1, U[i1,j1]=1; V[j0,k]=1, V[j1,k]=1; W[k,i0]=1
                # contrib: [i0,j0,k]=1, [i1,j1,k]=1
                vec = np.zeros(n * m * p)
                vec[io * m * p + jo * p + k] = 1.0
                vec[(io+1) * m * p + (jo+1) * p + k] = 1.0
                u_c = {(io, jo): 1, (io+1, jo+1): 1}
                v_c = {(jo, k): 1, (jo+1, k): 1}
                w_c = {(k, io): 1}
                candidates.append((vec, u_c, v_c, w_c))
                
                # U[i0,j0]=1, U[i1,j1]=1; V[j0,k]=1, V[j1,k]=1; W[k,i1]=1
                vec = np.zeros(n * m * p)
                vec[io * m * p + jo * p + k] = 1.0
                vec[(io+1) * m * p + (jo+1) * p + k] = 1.0
                u_c = {(io, jo): 1, (io+1, jo+1): 1}
                v_c = {(jo, k): 1, (jo+1, k): 1}
                w_c = {(k, io+1): 1}
                candidates.append((vec, u_c, v_c, w_c))
                
                # U[i0,j0]=1, U[i1,j0]=1; V[j0,k]=1, V[j1,k]=1; W[k,i0]=1
                vec = np.zeros(n * m * p)
                vec[io * m * p + jo * p + k] = 1.0
                vec[io * m * p + (jo+1) * p + k] = 1.0
                u_c = {(io, jo): 1, (io, jo+1): 1}
                v_c = {(jo, k): 1, (jo+1, k): 1}
                w_c = {(k, io): 1}
                candidates.append((vec, u_c, v_c, w_c))
                
                # U[i0,j0]=1, U[i1,j1]=1; V[j0,k]=1; W[k,i0]=1, W[k,i1]=1
                vec = np.zeros(n * m * p)
                vec[io * m * p + jo * p + k] = 1.0
                vec[(io+1) * m * p + (jo+1) * p + k] = 1.0
                u_c = {(io, jo): 1, (io+1, jo+1): 1}
                v_c = {(jo, k): 1}
                w_c = {(k, io): 1, (k, io+1): 1}
                candidates.append((vec, u_c, v_c, w_c))
    
    # Type 5: 1x2x2 blocks
    for i in range(n):
        for jo in range(m - 1):
            for ko in range(p - 1):
                # U[i,j0]=1, U[i,j1]=1; V[j0,k0]=1, V[j1,k1]=1; W[k0,i]=1
                vec = np.zeros(n * m * p)
                vec[i * m * p + jo * p + ko] = 1.0
                vec[i * m * p + (jo+1) * p + (ko+1)] = 1.0
                u_c = {(i, jo): 1, (i, jo+1): 1}
                v_c = {(jo, ko): 1, (jo+1, ko+1): 1}
                w_c = {(ko, i): 1}
                candidates.append((vec, u_c, v_c, w_c))
                
                # U[i,j0]=1, U[i,j1]=1; V[j0,k0]=1, V[j1,k1]=1; W[k1,i]=1
                vec = np.zeros(n * m * p)
                vec[i * m * p + jo * p + ko] = 1.0
                vec[i * m * p + (jo+1) * p + (ko+1)] = 1.0
                u_c = {(i, jo): 1, (i, jo+1): 1}
                v_c = {(jo, ko): 1, (jo+1, ko+1): 1}
                w_c = {(ko+1, i): 1}
                candidates.append((vec, u_c, v_c, w_c))
                
                # U[i,j0]=1, U[i,j1]=1; V[j0,k0]=1; W[k0,i]=1, W[k1,i]=1
                vec = np.zeros(n * m * p)
                vec[i * m * p + jo * p + ko] = 1.0
                vec[i * m * p + (jo+1) * p + (ko+1)] = 1.0
                u_c = {(i, jo): 1, (i, jo+1): 1}
                v_c = {(jo, ko): 1}
                w_c = {(ko, i): 1, (ko+1, i): 1}
                candidates.append((vec, u_c, v_c, w_c))
    
    # Type 6: 2x1x2 blocks
    for io in range(n - 1):
        for j in range(m):
            for ko in range(p - 1):
                # U[i0,j]=1, U[i1,j]=1; V[j,k0]=1, V[j,k1]=1; W[k0,i0]=1
                vec = np.zeros(n * m * p)
                vec[io * m * p + j * p + ko] = 1.0
                vec[(io+1) * m * p + j * p + (ko+1)] = 1.0
                u_c = {(io, j): 1, (io+1, j): 1}
                v_c = {(j, ko): 1, (j, ko+1): 1}
                w_c = {(ko, io): 1}
                candidates.append((vec, u_c, v_c, w_c))
                
                # U[i0,j]=1, U[i1,j]=1; V[j,k0]=1, V[j,k1]=1; W[k1,i1]=1
                vec = np.zeros(n * m * p)
                vec[io * m * p + j * p + ko] = 1.0
                vec[(io+1) * m * p + j * p + (ko+1)] = 1.0
                u_c = {(io, j): 1, (io+1, j): 1}
                v_c = {(j, ko): 1, (j, ko+1): 1}
                w_c = {(ko+1, io+1): 1}
                candidates.append((vec, u_c, v_c, w_c))
                
                # U[i0,j]=1, U[i1,j]=1; V[j,k0]=1; W[k0,i0]=1, W[k1,i1]=1
                vec = np.zeros(n * m * p)
                vec[io * m * p + j * p + ko] = 1.0
                vec[(io+1) * m * p + j * p + (ko+1)] = 1.0
                u_c = {(io, j): 1, (io+1, j): 1}
                v_c = {(j, ko): 1}
                w_c = {(ko, io): 1, (ko+1, io+1): 1}
                candidates.append((vec, u_c, v_c, w_c))
    
    # Precompute norms and build candidate matrix for fast greedy
    num_cands = len(candidates)
    cand_vecs = np.array([c[0] for c in candidates])  # (num_cands, 60)
    cand_norms_sq = np.sum(cand_vecs ** 2, axis=1)  # (num_cands,)
    # Avoid division by zero
    cand_norms_sq_safe = np.where(cand_norms_sq > 1e-15, cand_norms_sq, 1.0)
    
    # Precompute Gram matrix between candidates (optional, for rank reduction)
    # G[i,j] = dot(cand_i, cand_j)
    G = cand_vecs @ cand_vecs.T  # (num_cands, num_cands)
    
    T_target = np.ones(n * m * p)
    residual = T_target.copy()
    
    selected_indices = []
    selected_coeffs = []
    
    max_iter = 150
    for _ in range(max_iter):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Compute dot products with residual for all candidates at once
        dots = cand_vecs @ residual  # (num_cands,)
        reductions = dots ** 2 / cand_norms_sq_safe  # (num_cands,)
        
        # Find best candidate
        best_idx = int(np.argmax(reductions))
        best_reduction = reductions[best_idx]
        
        if best_reduction < 1e-10:
            break
        
        best_coeff = dots[best_idx] / cand_norms_sq_safe[best_idx]
        
        selected_indices.append(best_idx)
        selected_coeffs.append(best_coeff)
        
        residual -= best_coeff * cand_vecs[best_idx]
        
        if np.linalg.norm(residual) < 1e-9:
            break
    
    # Rank reduction pass: try to merge linearly dependent selected terms
    if len(selected_indices) > 1 and np.linalg.norm(residual) < 1e-9:
        # Build the sub-matrix of selected candidates
        sel_vecs = cand_vecs[selected_indices]  # (rank, 60)
        sel_coeffs_arr = np.array(selected_coeffs)
        
        # Check for linear dependencies among selected vectors
        # If sel_vecs has rank < len(selected_indices), we can merge
        # Use SVD to find the null space
        try:
            U_svd, s_vals, Vt_svd = np.linalg.svd(sel_vecs, full_matrices=False)
            rank_sel = np.sum(s_vals > 1e-10)
            
            if rank_sel < len(selected_indices):
                # There are dependencies. Try to find a sparser representation.
                # The null space of sel_vecs (as rows) gives coefficients that sum to zero.
                # We can use this to eliminate some terms.
                # For now, just note that we might be able to reduce.
                # A simple approach: iteratively remove the term with smallest |coeff|
                # and re-solve for the remaining.
                
                # Try removing each term and checking if the rest can still span T
                for try_remove in range(len(selected_indices) - 1, -1, -1):
                    if time.time() - start_time > DEADLINE - 10:
                        break
                    
                    remaining_indices = [i for i in range(len(selected_indices)) if i != try_remove]
                    if len(remaining_indices) == 0:
                        break
                    
                    rem_vecs = sel_vecs[remaining_indices]
                    
                    # Solve rem_vecs.T @ x = T_target (least squares)
                    # Actually we want sum_r x_r * rem_vecs[r] = T_target
                    # So rem_vecs.T @ x = T_target where x is (len(remaining),)
                    # This is an overdetermined system in general.
                    # Use least squares: x = pinv(rem_vecs.T) @ T_target
                    try:
                        x_sol = np.linalg.lstsq(rem_vecs.T, T_target, rcond=None)[0]
                        reconstruction = rem_vecs @ x_sol
                        if np.linalg.norm(reconstruction - T_target) < 1e-8:
                            # Success! We can drop term try_remove
                            new_selected_indices = [selected_indices[i] for i in remaining_indices]
                            new_selected_coeffs = x_sol.tolist()
                            selected_indices = new_selected_indices
                            selected_coeffs = new_selected_coeffs
                            # Update residual check
                            residual_new = T_target - np.array([cand_vecs[i] for i in selected_indices]) @ np.array(selected_coeffs)
                            if np.linalg.norm(residual_new) < 1e-8:
                                residual = residual_new
                    except:
                        pass
        except:
            pass
    
    # Check if exact
    if len(selected_indices) > 0 and np.linalg.norm(residual) < 1e-8:
        rank_found = len(selected_indices)
        
        # Construct U, V, W
        U = np.zeros((n * m, rank_found))
        V = np.zeros((m * p, rank_found))
        W = np.zeros((p * n, rank_found))
        
        for r, idx in enumerate(selected_indices):
            coeff = selected_coeffs[r]
            vec, u_c, v_c, w_c = candidates[idx]
            
            for (i, j), val in u_c.items():
                U[i * m + j, r] = val * coeff
            for (j, k), val in v_c.items():
                V[j * p + k, r] = val
            for (k, i), val in w_c.items():
                W[k * n + i, r] = val
                
        # Verify exactness with random test
        np.random.seed(42)
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(rank_found):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                C_calc[i, k] = val
                
        if np.allclose(C_test, C_calc, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Return the known 54-rank solution
    n, m, p = DIMENSIONS
    rank_new = 54
    U_new = np.zeros((n * m, rank_new))
    V_new = np.zeros((m * p, rank_new))
    W_new = np.zeros((p * n, rank_new))
    
    current_rank = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal current_rank
        # r=0: M1
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=1: M2
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = -1
        current_rank += 1
        # r=2: M3
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=3: M4
        U_new[(i_off+1)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = -1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=4: M5
        U_new[(i_off)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = -1
        W_new[(k_off+1)*n + (i_off), current_rank] = 1
        current_rank += 1
        # r=5: M6
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        U_new[(i_off)*m + (j_off), current_rank] = -1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        V_new[(j_off)*p + (k_off+1), current_rank] = 1
        W_new[(k_off+1)*n + (i_off+1), current_rank] = 1
        current_rank += 1
        # r=6: M7
        U_new[(i_off)*m + (j_off+1), current_rank] = 1
        U_new[(i_off+1)*m + (j_off+1), current_rank] = -1
        V_new[(j_off+1)*p + (k_off), current_rank] = 1
        V_new[(j_off+1)*p + (k_off+1), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1

    def add_standard_block(i_off, j_off, k_off):
        nonlocal current_rank
        U_new[(i_off)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off), current_rank] = 1
        current_rank += 1
        U_new[(i_off+1)*m + (j_off), current_rank] = 1
        V_new[(j_off)*p + (k_off), current_rank] = 1
        W_new[(k_off)*n + (i_off+1), current_rank] = 1
        current_rank += 1

    # Part 1: A[:, 0:2] * B[0:2, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Part 2a: A[:, 2:3] * B[2:3, :]
    for k in range(6):
        add_standard_block(0, 2, k)
        
    # Part 2b: A[:, 3:5] * B[3:5, :]
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
