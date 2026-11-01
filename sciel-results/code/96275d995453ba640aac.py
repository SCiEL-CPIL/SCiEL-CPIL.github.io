import time
import numpy as np
from itertools import product

DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60

    # Build a rich pool of candidate rank-1 tensors
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
    
    # Type 1: Standard unit tensors (60 candidates)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_candidate({(i,j): 1}, {(j,k): 1}, {(k,i): 1})
    
    # Type 2: Strassen 2x2x2 blocks (7 terms per block)
    def strassen_terms(io, jo, ko):
        terms = []
        # M1: (A00+A11)(B00+B11)
        terms.append(({(io,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1, (ko+1,io+1): 1}))
        # M2: (A10+A11)(B00)
        terms.append(({(io+1,jo): 1, (io+1,jo+1): 1}, 
                      {(jo,ko): 1}, 
                      {(ko,io+1): 1, (ko+1,io+1): -1}))
        # M3: A00(B01-B11)
        terms.append(({(io,jo): 1}, 
                      {(jo,ko+1): 1, (jo+1,ko+1): -1}, 
                      {(ko+1,io): 1, (ko+1,io+1): 1}))
        # M4: A11(B10-B00)
        terms.append(({(io+1,jo+1): 1}, 
                      {(jo+1,ko): 1, (jo,ko): -1}, 
                      {(ko,io): 1, (ko,io+1): 1}))
        # M5: (A00+A01)(B11)
        terms.append(({(io,jo): 1, (io,jo+1): 1}, 
                      {(jo+1,ko+1): 1}, 
                      {(ko,io): -1, (ko+1,io): 1}))
        # M6: (A10-A00)(B00+B01)
        terms.append(({(io+1,jo): 1, (io,jo): -1}, 
                      {(jo,ko): 1, (jo,ko+1): 1}, 
                      {(ko+1,io+1): 1}))
        # M7: (A01-A11)(B10+B11)
        terms.append(({(io,jo+1): 1, (io+1,jo+1): -1}, 
                      {(jo+1,ko): 1, (jo+1,ko+1): 1}, 
                      {(ko,io): 1}))
        return terms
    
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
    
    # Type 3: Additional structured candidates for 2x3x3 blocks
    # These help cover cases where Strassen alone isn't optimal
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                # Generate mixed candidates involving 3 j-indices and 3 k-indices
                # Pattern: combine Strassen on first 2 j's with standard on 3rd j
                # and cross terms
                
                # Standard terms for j=jo+2
                for i_off in range(2):
                    for k_off in range(3):
                        add_candidate(
                            {(io+i_off, jo+2): 1},
                            {(jo+2, ko+k_off): 1},
                            {(ko+k_off, io+i_off): 1}
                        )
                
                # Cross terms: A[i,jo] * B[jo+2,k] patterns
                for i_off in range(2):
                    for k_off in range(3):
                        add_candidate(
                            {(io+i_off, jo): 1, (io+i_off, jo+2): 1},
                            {(jo, ko+k_off): 1, (jo+2, ko+k_off): -1},
                            {(ko+k_off, io+i_off): 1}
                        )
                
                # More diverse patterns
                for i_off in range(2):
                    for j_off in range(3):
                        for k_off in range(3):
                            # Skip duplicates already added
                            if j_off == 0 and k_off == 0:
                                continue
                            add_candidate(
                                {(io+i_off, jo+j_off): 1},
                                {(jo+j_off, ko+k_off): 1},
                                {(ko+k_off, io+i_off): 1}
                            )
    
    # Type 4: Half-integer candidates (may help with exact decompositions)
    # These are Strassen-like but with 1/2 coefficients
    def strassen_half_terms(io, jo, ko):
        terms = []
        # Modified Strassen with half-integers
        terms.append(({(io,jo): 0.5, (io+1,jo+1): 0.5}, 
                      {(jo,ko): 0.5, (jo+1,ko+1): 0.5}, 
                      {(ko,io): 4, (ko+1,io+1): 4}))
        terms.append(({(io+1,jo): 0.5, (io+1,jo+1): 0.5}, 
                      {(jo,ko): 0.5}, 
                      {(ko,io+1): 4, (ko+1,io+1): -4}))
        terms.append(({(io,jo): 0.5}, 
                      {(jo,ko+1): 0.5, (jo+1,ko+1): -0.5}, 
                      {(ko+1,io): 4, (ko+1,io+1): 4}))
        terms.append(({(io+1,jo+1): 0.5}, 
                      {(jo+1,ko): 0.5, (jo,ko): -0.5}, 
                      {(ko,io): 4, (ko,io+1): 4}))
        terms.append(({(io,jo): 0.5, (io,jo+1): 0.5}, 
                      {(jo+1,ko+1): 0.5}, 
                      {(ko,io): -4, (ko+1,io): 4}))
        terms.append(({(io+1,jo): 0.5, (io,jo): -0.5}, 
                      {(jo,ko): 0.5, (jo,ko+1): 0.5}, 
                      {(ko+1,io+1): 4}))
        terms.append(({(io,jo+1): 0.5, (io+1,jo+1): -0.5}, 
                      {(jo+1,ko): 0.5, (jo+1,ko+1): 0.5}, 
                      {(ko,io): 4}))
        return terms
    
    for io in range(n - 1):
        for jo in range(m - 1):
            for ko in range(p - 1):
                for u_c, v_c, w_c in strassen_half_terms(io, jo, ko):
                    add_candidate(u_c, v_c, w_c)
    
    # Now use greedy to find a good decomposition
    T_target = np.ones(total_entries)
    residual = T_target.copy()
    
    selected_indices = []
    selected_coeffs = []
    
    num_candidates = len(candidates)
    
    # Precompute norms of all candidates
    norms = np.zeros(num_candidates)
    for idx in range(num_candidates):
        vec = candidates[idx][3]
        norms[idx] = np.dot(vec, vec)
    
    # Greedy selection with multiple passes
    max_iter = 200
    for iteration in range(max_iter):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        best_idx = -1
        best_reduction = 0
        best_coeff = 0
        
        for idx in range(num_candidates):
            if norms[idx] < 1e-12:
                continue
            dot_rc = np.dot(residual, candidates[idx][3])
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
        residual -= best_coeff * candidates[best_idx][3]
        
        if np.linalg.norm(residual) < 1e-9:
            break
    
    # Check if we got an exact solution
    if np.linalg.norm(residual) < 1e-9:
        rank_found = len(selected_indices)
        
        U = np.zeros((n * m, rank_found))
        V = np.zeros((m * p, rank_found))
        W = np.zeros((p * n, rank_found))
        
        for r, idx in enumerate(selected_indices):
            coeff = selected_coeffs[r]
            u_vec, v_vec, w_vec, _ = candidates[idx]
            U[:, r] = u_vec * coeff
            V[:, r] = v_vec
            W[:, r] = w_vec
        
        # Verify with multiple test cases
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
    
    # Fallback: return a known good solution
    # Use a block decomposition that's known to work
    n, m, p = 2, 5, 6
    
    # Strategy: Use Strassen for 2x2x2 blocks where possible,
    # and standard multiplication elsewhere
    # 
    # For 2x5x6, we can partition B into columns:
    # - Columns 0-1: Use Strassen (7 terms) for each row pair
    # - Columns 2-3: Use Strassen (7 terms) for each row pair  
    # - Column 4: Standard (2 terms per row)
    # - Column 5: Standard (2 terms per row)
    #
    # But we need to be more careful about how we partition.
    
    # Let's use a systematic approach:
    # Partition the 5 columns into groups of size 2, 2, 1
    # For each group of 2 columns, use Strassen (7 terms)
    # For single columns, use standard (2 terms)
    
    rank_new = 0
    U_new = []
    V_new = []
    W_new = []
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal rank_new
        u_vec = np.zeros(n * m)
        v_vec = np.zeros(m * p)
        w_vec = np.zeros(p * n)
        
        for (i, j), val in u_dict.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_dict.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_dict.items():
            w_vec[k * n + i] = val
        
        U_new.append(u_vec)
        V_new.append(v_vec)
        W_new.append(w_vec)
        rank_new += 1
    
    # Process column pairs (0,1), (2,3) with Strassen, and columns 4,5 with standard
    for col_start in [0, 2]:
        # Strassen for columns col_start, col_start+1
        for i in range(n):
            # We need to handle each row separately since n=2
            # Actually, Strassen works on 2x2 blocks, so we use it directly
            pass
    
    # Simpler fallback: just use standard multiplication
    # This gives rank = 2*5*6 = 60, but we can do better
    
    # Use a hybrid: Strassen for 2x2x2 blocks
    # For 2x5x6, we have 1 x 4 x 5 = 20 possible 2x2x2 blocks
    # But they overlap, so we need to be careful
    
    # Let's just return the 54-rank solution from the original program
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
