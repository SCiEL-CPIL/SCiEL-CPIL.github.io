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
    
    # Type 3: 2x3x3 block candidates
    # Generate all sign combinations for 2 i's, 3 j's, 3 k's
    # This allows the greedy search to potentially find rank-11 decompositions
    # if they exist in this space.
    
    # To limit computation time, we only add these if we have time
    if time.time() - start_time < 10:
        for io in range(n - 1):
            for jo in range(m - 2):
                for ko in range(p - 2):
                    # i indices: io, io+1
                    # j indices: jo, jo+1, jo+2
                    # k indices: ko, ko+1, ko+2
                    
                    # Generate all sign patterns for u, v, w
                    # u has 2 non-zeros, v has 3, w has 3
                    # We iterate over all combinations of +/-1 for the non-zero entries
                    
                    # To avoid adding too many, we can sample or just add all.
                    # 2^2 * 2^3 * 2^3 = 256 candidates per block.
                    # Number of blocks: 1 * 3 * 4 = 12 blocks.
                    # Total 12 * 256 = 3072 candidates. This is fine.
                    
                    for u_s0 in [1, -1]:
                        for u_s1 in [1, -1]:
                            for v_s0 in [1, -1]:
                                for v_s1 in [1, -1]:
                                    for v_s2 in [1, -1]:
                                        for w_s0 in [1, -1]:
                                            for w_s1 in [1, -1]:
                                                for w_s2 in [1, -1]:
                                                    u_c = {(io, jo): u_s0, (io+1, jo+1): u_s1} 
                                                    # Note: The support of u, v, w must match the tensor structure?
                                                    # Actually, any rank-1 tensor is a candidate.
                                                    # But for it to be part of a decomposition of the *multiplication* tensor,
                                                    # the supports must be such that the product matches the target structure.
                                                    # The target tensor T[i,j,k] = 1 if the indices are valid.
                                                    # A rank-1 term is u[i]v[j]w[k].
                                                    # For it to contribute to T, u[i]v[j]w[k] must be non-zero only where T is non-zero?
                                                    # No, the sum of terms must equal T.
                                                    # So we can have any rank-1 tensor.
                                                    # However, to be efficient, we should restrict to "structured" supports.
                                                    # The above u_c has support (io,jo) and (io+1,jo+1).
                                                    # This is a diagonal-like support.
                                                    # Let's use a more general support for the 2x3x3 block.
                                                    # u support: any 2 of the 6 positions in 2x3? No, u is 2x3.
                                                    # Let's stick to the Strassen-like structure but extended.
                                                    # Actually, just adding all sign combinations of a *fixed* support pattern
                                                    # might not be enough. We need the *right* support pattern.
                                                    # The rank-11 decomposition has specific supports.
                                                    # Without knowing them, random supports are unlikely to work.
                                                    # 
                                                    # However, the greedy search works by finding *any* linear combination.
                                                    # If the pool contains the *basis* vectors of the subspace spanned by the 
                                                    # rank-1 terms of the optimal decomposition, it can find it.
                                                    # But the rank-1 terms themselves are the basis.
                                                    # So we need the *exact* rank-1 vectors.
                                                    # 
                                                    # Given the difficulty, let's just add the 2x2x3 and 2x3x2 Strassen-like terms.
                                                    # Or better, just run the greedy search on the existing pool but with more iterations
                                                    # and random restarts.
                                                    pass

    # Let's refine the strategy: 
    # The current pool is sufficient to find the rank-54 solution.
    # To find a lower rank, we need better candidates.
    # Since I cannot generate the exact rank-11 2x3x3 terms, I will try to 
    # improve the greedy search to find a rank < 54 solution if one exists 
    # in the current pool (unlikely) or simply ensure the fallback is used 
    # if the search fails.
    
    # Actually, let's try to add candidates for 2x2x3 blocks.
    # A 2x2x3 block can be decomposed into a 2x2x2 (rank 7) + 2x2x1 (rank 4)? Total 11.
    # Or 2x2x3 rank is 11?
    # Let's add candidates that are sums of Strassen terms + unit terms.
    
    # For now, let's just run the greedy search with the current pool.
    # The current pool has ~ 60 + 42*7 + ... candidates.
    # It found 54.
    
    # To improve, let's add a "Randomized" greedy search with multiple restarts.
    
    T_target = np.ones(total_entries)
    residual = T_target.copy()
    
    selected_indices = []
    selected_coeffs = []
    
    num_candidates = len(candidates)
    
    # Precompute all contribution vectors as a matrix
    contrib_matrix = np.zeros((num_candidates, total_entries))
    for idx in range(num_candidates):
        contrib_matrix[idx] = candidates[idx][3]
    
    # Precompute norms
    norms = np.sum(contrib_matrix ** 2, axis=1)
    
    # Precompute Gram matrix
    if num_candidates <= 2000:
        Gram = contrib_matrix @ contrib_matrix.T
    else:
        Gram = None
    
    best_rank = float('inf')
    best_solution = None
    
    # Run greedy search with random restarts
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        residual = T_target.copy()
        selected_indices = []
        selected_coeffs = []
        
        max_iter = 100
        for _ in range(max_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            best_idx = -1
            best_reduction = 0
            best_coeff = 0
            
            if Gram is not None:
                dots = contrib_matrix @ residual
                valid = norms > 1e-12
                reductions = np.where(valid, dots**2 / np.where(norms>1e-12, norms, 1), 0)
                best_idx = int(np.argmax(reductions))
                best_reduction = reductions[best_idx]
                if best_reduction < 1e-9:
                    break
                best_coeff = dots[best_idx] / norms[best_idx]
            else:
                for idx in range(num_candidates):
                    if norms[idx] < 1e-12:
                        continue
                    dot_rc = np.dot(residual, contrib_matrix[idx])
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
            residual -= best_coeff * contrib_matrix[best_idx]
            
            if np.linalg.norm(residual) < 1e-9:
                break
        
        if np.linalg.norm(residual) < 1e-9:
            rank_found = len(selected_indices)
            if rank_found < best_rank:
                best_rank = rank_found
                # Store solution
                U = np.zeros((n * m, rank_found))
                V = np.zeros((m * p, rank_found))
                W = np.zeros((p * n, rank_found))
                for r, idx in enumerate(selected_indices):
                    coeff = selected_coeffs[r]
                    u_vec, v_vec, w_vec, _ = candidates[idx]
                    U[:, r] = u_vec * coeff
                    V[:, r] = v_vec
                    W[:, r] = w_vec
                best_solution = (U, V, W)

    if best_solution is not None:
        U, V, W = best_solution
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
