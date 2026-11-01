import time
import numpy as np

DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60
    
    # Build a large pool of candidate rank-1 tensors
    candidates = []  # list of (u_vec, v_vec, w_vec, contrib_vec)
    
    def add_candidate(u_sparse, v_sparse, w_sparse):
        """Add a candidate defined by sparse dicts."""
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        for (i, j), val in u_sparse.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_sparse.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_sparse.items():
            w_vec[k * n + i] = val
        
        # Compute contribution vector
        contrib = np.zeros(total_entries)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i * m * p + j * p + k] = u_vec[i*m+j] * v_vec[j*p+k] * w_vec[k*n+i]
        
        candidates.append((u_vec, v_vec, w_vec, contrib))
    
    # Type 1: Standard unit tensors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_candidate({(i,j): 1}, {(j,k): 1}, {(k,i): 1})
    
    # Type 2: Strassen 2x2x2 blocks
    def strassen_terms(io, jo, ko):
        terms = []
        terms.append(({(io,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1}, 
                      {(ko,io+1): 1, (ko+1,io+1): -1}))
        terms.append(({(io,jo): 1}, 
                      {(jo,ko+1): 1, (jo+1,ko+1): -1}, 
                      {(ko+1,io): 1, (ko+1,io+1): 1}))
        terms.append(({(io+1,jo+1): 1}, 
                      {(jo+1,ko): 1, (jo,ko): -1}, 
                      {(ko,io): 1, (ko,io+1): 1}))
        terms.append(({(io,jo): 1, (io,jo+1): 1}, 
                      {(jo+1,ko+1): 1}, 
                      {(ko,io): -1, (ko+1,io): 1}))
        terms.append(({(io+1,jo): 1, (io,jo): -1}, 
                      {(jo,ko): 1, (jo,ko+1): 1}, 
                      {(ko+1,io+1): 1}))
        terms.append(({(io,jo+1): 1, (io+1,jo+1): -1}, 
                      {(jo+1,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1}))
        return terms
    
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
    
    # Type 3: Additional structured candidates with wider blocks
    # 2x3x2 blocks: combine Strassen with an extra column
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 1):
                # Standard terms for the 2x3x2 region
                for j_off in range(3):
                    for k_off in range(2):
                        for i_off in range(2):
                            add_candidate(
                                {(io+i_off, jo+j_off): 1},
                                {(jo+j_off, ko+k_off): 1},
                                {(ko+k_off, io+i_off): 1}
                            )
                # Strassen-like on first 2 columns, standard on 3rd
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                # Cross terms involving all 3 columns
                add_candidate({(io,jo): 1, (io+1,jo+2): 1},
                              {(jo,ko): 1, (jo+1,ko+1): 1, (jo+2,ko): 1},
                              {(ko,io): 1, (ko+1,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): -1},
                              {(jo,ko): 1, (jo+1,ko+1): -1, (jo+2,ko): 1},
                              {(ko,io): 1, (ko+1,io+1): 1})
    
    # Type 4: 2x2x3 blocks
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 2):
                for j_off in range(2):
                    for k_off in range(3):
                        for i_off in range(2):
                            add_candidate(
                                {(io+i_off, jo+j_off): 1},
                                {(jo+j_off, ko+k_off): 1},
                                {(ko+k_off, io+i_off): 1}
                            )
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
                # Cross terms
                add_candidate({(io,jo): 1, (io+1,jo+1): 1},
                              {(jo,ko): 1, (jo+1,ko+1): 1, (jo+1,ko+2): 1},
                              {(ko,io): 1, (ko+1,io+1): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+1): -1},
                              {(jo,ko): 1, (jo+1,ko+1): -1, (jo+1,ko+2): 1},
                              {(ko,io): 1, (ko+1,io+1): 1, (ko+2,io+1): 1})
    
    # Type 5: 2x3x3 blocks - more diverse patterns
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                # All standard
                for j_off in range(3):
                    for k_off in range(3):
                        for i_off in range(2):
                            add_candidate(
                                {(io+i_off, jo+j_off): 1},
                                {(jo+j_off, ko+k_off): 1},
                                {(ko+k_off, io+i_off): 1}
                            )
                # Strassen on various 2x2x2 sub-blocks within 2x3x3
                add_candidate({(io,jo): 1, (io+1,jo+1): 1},
                              {(jo,ko): 1, (jo+1,ko+1): 1},
                              {(ko,io): 1, (ko+1,io+1): 1})
                add_candidate({(io,jo+1): 1, (io+1,jo+2): 1},
                              {(jo+1,ko): 1, (jo+2,ko+1): 1},
                              {(ko,io): 1, (ko+1,io+1): 1})
                add_candidate({(io,jo): 1, (io+1,jo+2): 1},
                              {(jo,ko): 1, (jo+2,ko+2): 1},
                              {(ko,io): 1, (ko+2,io+1): 1})
                # Mixed patterns
                add_candidate({(io,jo): 1, (io,jo+1): 1, (io+1,jo+2): 1},
                              {(jo,ko): 1, (jo+1,ko+1): 1, (jo+2,ko+2): 1},
                              {(ko,io): 1, (ko+1,io): 1, (ko+2,io+1): 1})
                add_candidate({(io,jo): 1, (io,jo+1): -1, (io+1,jo+2): 1},
                              {(jo,ko): 1, (jo+1,ko+1): -1, (jo+2,ko+2): 1},
                              {(ko,io): 1, (ko+1,io): 1, (ko+2,io+1): 1})
    
    # Now use greedy to find a good decomposition
    T_target = np.ones(total_entries)
    residual = T_target.copy()
    
    selected_indices = []
    selected_coeffs = []
    
    num_candidates = len(candidates)
    
    # Precompute norms of all candidates
    norms = np.zeros(num_candidates)
    contribs = np.array([c[3] for c in candidates])
    for idx in range(num_candidates):
        norms[idx] = np.dot(contribs[idx], contribs[idx])
    
    max_iter = 100
    for _ in range(max_iter):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        best_idx = -1
        best_reduction = 0
        best_coeff = 0
        
        for idx in range(num_candidates):
            if norms[idx] < 1e-12:
                continue
            dot_rc = np.dot(residual, contribs[idx])
            coeff = dot_rc / norms[idx]
            reduction = (dot_rc ** 2) / norms[idx]
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_idx = idx
                best_coeff = coeff
        
        if best_idx == -1 or best_reduction < 1e-9:
            break
        
        selected_indices.append(best_idx)
        selected_coeffs.append(best_coeff)
        residual -= best_coeff * contribs[best_idx]
        
        if np.linalg.norm(residual) < 1e-9:
            break
    
    # Check if we got an exact solution
    if np.linalg.norm(residual) < 1e-9:
        # Try to reduce rank by finding linear dependencies
        rank_found = len(selected_indices)
        
        if rank_found > 1:
            # Build matrix of contribution vectors for selected terms
            C_sel = np.array([candidates[idx][3] for idx in selected_indices])
            # Find rank via SVD
            try:
                U_svd, s_vals, Vt_svd = np.linalg.svd(C_sel, full_matrices=False)
                tol = 1e-8
                effective_rank = np.sum(s_vals > tol)
                
                if effective_rank < rank_found:
                    # We have dependencies - reconstruct minimal set
                    # Keep only the first effective_rank independent vectors
                    # This is a simplification; a proper approach would use null space
                    # For now, just check if we can verify with fewer terms
                    pass
            except:
                pass
        
        U = np.zeros((n * m, rank_found))
        V = np.zeros((m * p, rank_found))
        W = np.zeros((p * n, rank_found))
        
        for r, idx in enumerate(selected_indices):
            coeff = selected_coeffs[r]
            u_vec, v_vec, w_vec, _ = candidates[idx]
            U[:, r] = u_vec * coeff
            V[:, r] = v_vec
            W[:, r] = w_vec
        
        # Verify
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_test = A_test @ B_test
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0
                for j in range(m):
                    for r in range(rank_found):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                C_calc[i, k] = val
        
        if np.allclose(C_test, C_calc, atol=1e-5):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return a verified solution
    n, m, p = 2, 5, 6
    
    # Build solution using Strassen blocks + standard terms
    # Tiling: split B into columns [0:2], [2:3], [3:5]
    # [0:2] uses 3 Strassen blocks (k=0,2,4) -> 21 terms
    # [2:3] uses standard 2x1x6 -> 12 terms  
    # [3:5] uses 3 Strassen blocks (k=0,2,4) -> 21 terms
    # Total: 54 terms
    
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

    def add_standard_term(i, j, k):
        nonlocal current_rank
        U_new[i*m + j, current_rank] = 1
        V_new[j*p + k, current_rank] = 1
        W_new[k*n + i, current_rank] = 1
        current_rank += 1

    # Part 1: A[:, 0:2] * B[0:2, :]
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Part 2: A[:, 2:3] * B[2:3, :]
    for i in range(n):
        for k in range(p):
            add_standard_term(i, 2, k)
    
    # Part 3: A[:, 3:5] * B[3:5, :]
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    # Verify the fallback
    A_test = np.random.randint(-10, 10, (n, m))
    B_test = np.random.randint(-10, 10, (m, p))
    C_test = A_test @ B_test
    
    C_calc = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0
            for j in range(m):
                for r in range(current_rank):
                    val += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
            C_calc[i, k] = val
    
    if not np.allclose(C_test, C_calc, atol=1e-5):
        # Emergency fallback: standard multiplication
        rank_std = n * m * p
        U_std = np.zeros((n * m, rank_std))
        V_std = np.zeros((m * p, rank_std))
        W_std = np.zeros((p * n, rank_std))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_std[i*m+j, r] = 1
                    V_std[j*p+k, r] = 1
                    W_std[k*n+i, r] = 1
                    r += 1
        return {"U": U_std.tolist(), "V": V_std.tolist(), "W": W_std.tolist()}
    
    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
