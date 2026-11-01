import time
import numpy as np

DEADLINE = 115.0

def construct():
    start_time = time.time()
    n, m, p = 2, 5, 6
    total_entries = n * m * p  # 60
    
    # Build a large pool of candidate rank-1 tensors
    # A rank-1 tensor is (u_vec, v_vec, w_vec) where:
    #   u_vec has length n*m, v_vec has length m*p, w_vec has length p*n
    # and the contribution to the (i,j,k) entry is u[i*m+j] * v[j*p+k] * w[k*n+i]
    
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
    
    # Type 3: Rank-11 bilinear algorithm for 2x3x3 sub-blocks
    # We can apply a rank-11 algorithm to 2x3x3 blocks
    # For 2x3x3: rows i=io,io+1; cols j=jo..jo+2; k=ko..ko+2
    # The rank of the 2x3x3 multiplication tensor is 11 (not 18)
    # This is a known result from algebraic complexity theory
    
    def rank11_2x3x3_terms(io, jo, ko):
        """
        Return 11 rank-1 terms that decompose the 2x3x3 matrix multiplication tensor.
        This uses a construction based on the fact that the rank of M_{2,3,3} is 11.
        """
        terms = []
        # i indices: io, io+1
        # j indices: jo, jo+1, jo+2
        # k indices: ko, ko+1, ko+2
        
        # Construction: Use the rank-7 for 2x2x2 on the first two j's,
        # plus additional terms for the third j and interactions.
        # Actually, let me use a known construction.
        
        # A simpler approach: Use rank 7 for 2x2x2 on (i, j=jo..jo+1, k=ko..ko+2)
        # But that doesn't directly work since Strassen is for 2x2x2.
        
        # Let me use a different strategy:
        # The 2x3x3 tensor can be seen as 2x2x3 + 2x1x3
        # 2x2x3 has rank 11? No, let me think differently.
        
        # Actually, let me just use the known fact that we can get rank 11
        # by combining Strassen-like terms with the extra column.
        
        # For now, let me generate candidates that involve all three j's:
        
        # Term 1: (A00+A11)(B00+B11+B22)
        u1 = {(io,jo): 1, (io+1,jo+2): 1}
        v1 = {(jo,ko): 1, (jo+1,ko+1): 1, (jo+2,ko+2): 1}
        w1 = {(ko,io): 1, (ko+1,io+1): 1, (ko+2,io+1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: (A00+A11)(B00-B11-B22)
        u2 = {(io,jo): 1, (io+1,jo+2): 1}
        v2 = {(jo,ko): 1, (jo+1,ko+1): -1, (jo+2,ko+2): -1}
        w2 = {(ko,io): 1, (ko+1,io+1): 1, (ko+2,io+1): 1}
        terms.append((u2, v2, w2))
        
        # Term 3: (A01+A12)(B00+B11)
        u3 = {(io,jo+1): 1, (io+1,jo+2): 1}
        v3 = {(jo,ko): 1, (jo+1,ko+1): 1}
        w3 = {(ko,io): 1, (ko+1,io+1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: (A01-A12)(B00-B11)
        u4 = {(io,jo+1): 1, (io+1,jo+2): -1}
        v4 = {(jo,ko): 1, (jo+1,ko+1): -1}
        w4 = {(ko,io): 1, (ko+1,io+1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: A00*(B00+B11)
        u5 = {(io,jo): 1}
        v5 = {(jo,ko): 1, (jo+1,ko+1): 1}
        w5 = {(ko,io): 1, (ko+1,io): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: A00*(B00-B11)
        u6 = {(io,jo): 1}
        v6 = {(jo,ko): 1, (jo+1,ko+1): -1}
        w6 = {(ko,io): 1, (ko+1,io): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: A11*(B00+B11)
        u7 = {(io+1,jo+1): 1}
        v7 = {(jo,ko): 1, (jo+1,ko+1): 1}
        w7 = {(ko,io+1): 1, (ko+1,io+1): 1}
        terms.append((u7, v7, w7))
        
        # Term 8: A11*(B00-B11)
        u8 = {(io+1,jo+1): 1}
        v8 = {(jo,ko): 1, (jo+1,ko+1): -1}
        w8 = {(ko,io+1): 1, (ko+1,io+1): 1}
        terms.append((u8, v8, w8))
        
        # Term 9: A02*B22
        u9 = {(io,jo+2): 1}
        v9 = {(jo+2,ko+2): 1}
        w9 = {(ko+2,io): 1}
        terms.append((u9, v9, w9))
        
        # Term 10: A12*B22
        u10 = {(io+1,jo+2): 1}
        v10 = {(jo+2,ko+2): 1}
        w10 = {(ko+2,io+1): 1}
        terms.append((u10, v10, w10))
        
        # Term 11: Cross terms
        u11 = {(io,jo+1): 1, (io+1,jo): -1}
        v11 = {(jo,ko+2): 1, (jo+1,ko+1): -1}
        w11 = {(ko+2,io): 1, (ko+1,io+1): 1}
        terms.append((u11, v11, w11))
        
        return terms
    
    # Actually, the above construction is not verified to give rank 11.
    # Let me use a simpler approach: just generate more diverse candidates.
    
    # Type 3: Mixed patterns with 3 columns
    for io in range(n - 1):
        for jo in range(m - 2):
            for ko in range(p - 2):
                # Generate some structured candidates for 2x3x3
                # Simple: use standard terms for all 9 positions
                for j_off in range(3):
                    for k_off in range(3):
                        for i_off in range(2):
                            add_candidate(
                                {(io+i_off, jo+j_off): 1},
                                {(jo+j_off, ko+k_off): 1},
                                {(ko+k_off, io+i_off): 1}
                            )
    
    # Type 4: Additional Strassen-like terms with 3-wide blocks
    # These are heuristic candidates that might help
    
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
    
    # Fallback: return the known 54-rank solution
    n, m, p = 2, 5, 6
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
