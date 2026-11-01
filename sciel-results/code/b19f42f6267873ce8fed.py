"""Kissing number in dimension 11: constructive algebraic approach via E_11 root lattice."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    # The E_11 root lattice has 792 shortest vectors (squared norm 2).
    # These are the kissing vectors of the E_11 lattice.
    # We need to construct them explicitly.
    
    # The E_n lattice (n >= 5) can be constructed as follows:
    # E_n = { x in R^n : x_i in Z for all i, sum x_i even }
    #       union
    #       { x in R^n : x_i in Z + 1/2 for all i, sum x_i even }
    
    # The shortest vectors have squared norm 2.
    # In the first component (integer part with even sum):
    #   Vectors with two entries of +/-1 and rest 0, with even sum.
    #   Two entries of +1: sum=2 (even) -> valid
    #   Two entries of -1: sum=-2 (even) -> valid
    #   One +1 and one -1: sum=0 (even) -> valid
    #   So all vectors with exactly two non-zero entries, each +/-1, are valid.
    #   Count: C(11,2) * 4 = 55 * 4 = 220
    
    # In the second component (half-integer part with even sum):
    #   x_i in Z + 1/2 for all i, sum x_i even
    #   Since each x_i = k_i + 1/2 where k_i is integer,
    #   sum x_i = sum k_i + 11/2 = sum k_i + 5.5
    #   For sum x_i to be even (integer), we need sum k_i + 5.5 to be even integer
    #   But 5.5 is not an integer, so sum k_i + 5.5 can never be an integer.
    #   Wait, that means the second component contributes nothing?
    
    # Let me reconsider. The standard definition:
    # E_n = { x in R^n : x_i in Z, sum x_i even } union { x in R^n : x_i in Z+1/2, sum x_i even }
    # 
    # For the second part: x_i = m_i + 1/2 where m_i in Z
    # sum x_i = sum(m_i) + n/2
    # For n=11: sum x_i = sum(m_i) + 11/2
    # This is never an integer, so it can never be "even" in the integer sense.
    # 
    # Hmm, but the E_11 lattice should have 792 roots. Let me reconsider.
    
    # Actually, I think I need to be more careful. Let me use a different construction.
    # 
    # Alternative: The E_n lattice can also be described using the D_n lattice.
    # D_n = { x in Z^n : sum x_i even }
    # E_n for n >= 6 can be constructed from D_{n-1} or D_{n+1}.
    #
    # Actually, let me just directly generate all candidate vectors with small entries
    # and filter for the right properties.
    
    # Let's take a different approach. The kissing number of E_11 is known to be 792.
    # The roots of E_11 are:
    # 1. Vectors of type (±1, ±1, 0, ..., 0): 2 * C(11,2) * 2 = wait...
    #    Two positions chosen, each can be +1 or -1: C(11,2) * 4 = 220
    # 2. Vectors of type (±1/2, ±1/2, ..., ±1/2) with 11 entries, all non-zero:
    #    2^11 = 2048, but we need even sum constraint.
    #    Sum = (1/2) * (sum of signs)
    #    For sum to be even integer... (1/2)*(integer) is even integer only if the integer is 4k.
    #    Sum of 11 signs, each ±1: ranges from -11 to 11 in steps of 2.
    #    (1/2) * sum is an integer when sum is even, i.e., when the number of +1s is even or odd?
    #    Sum of 11 signs: if k are +1 and 11-k are -1, sum = k - (11-k) = 2k - 11
    #    2k - 11 is always odd (since 2k is even, 2k-11 is odd)
    #    So (1/2)*(2k-11) is never an integer.
    #    
    # This means the half-integer component contributes no roots for odd n? That seems wrong.
    
    # Let me look at this differently. Perhaps I should just generate all vectors with
    # entries in {-1, 0, 1} or {-1/2, 0, 1/2} or mixed, with squared norm 2, and check
    # which ones form a valid kissing configuration.
    
    # Actually, for the purpose of this task, I don't need the exact E_11 roots.
    # I need ANY set of integer vectors in R^11 where min pairwise distance >= max norm.
    # 
    # The score is k (number of vectors). So I want to maximize k.
    #
    # Let me use a hybrid approach:
    # 1. Generate W1 vectors: (±2, 0, ..., 0) - 22 vectors, norm 2
    # 2. Generate W2 vectors: (±1, ±1, 0, ...) - 220 vectors, norm sqrt(2)
    # 3. Generate W4 vectors: (±1, ±1, ±1, ±1, 0, ...) - 5280 vectors, norm 2
    #
    # Then find a large independent set using the conflict graph approach,
    # but with a better initialization and local search.
    
    # Generate W1 vectors
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    vectors_v1 = np.array(v1_list, dtype=np.int64)
    N1 = len(vectors_v1)
    
    # Generate W2 vectors
    v2_list = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_list.append(v)
    vectors_v2 = np.array(v2_list, dtype=np.int64)
    N2 = len(vectors_v2)
    
    # Generate W4 vectors
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    vectors_v4 = np.array(v4_list, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Combine all vectors
    all_vectors = np.vstack([vectors_v1, vectors_v2, vectors_v4])
    N_total = len(all_vectors)
    
    # Compute norms
    norms = np.linalg.norm(all_vectors, axis=1)
    max_norm = np.max(norms)
    
    # Compute conflict matrix: conflict if distance < max_norm
    dots = all_vectors @ all_vectors.T
    dists_sq = norms[:, None]**2 + norms[None, :]**2 - 2 * dots
    np.fill_diagonal(dists_sq, np.inf)
    
    conflicts = dists_sq < max_norm**2 - 1e-9
    np.fill_diagonal(conflicts, False)
    
    # Build adjacency lists
    adj = [np.where(conflicts[i])[0].tolist() for i in range(N_total)]
    adj_sizes = np.array([len(a) for a in adj], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    def randomized_greedy():
        """Randomized greedy: pick vertices in random order, add if no conflict."""
        in_set = np.zeros(N_total, dtype=bool)
        order = rng.permutation(N_total)
        for v in order:
            if not in_set[v]:
                if not np.any(conflicts[v] & in_set):
                    in_set[v] = True
        return in_set
    
    def local_search(in_set, max_time=5.0):
        """Local search with Add/Remove moves using conflict counts."""
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # conflict_count[i] = number of neighbors of i in in_set
        conflict_count = np.zeros(N_total, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[adj[i]] += 1
        
        ls_deadline = time.time() + max_time
        T = 1.0
        cooling = 0.995
        min_T = 0.05
        
        while time.time() < ls_deadline:
            # Try Add Move
            not_in = np.where(~in_set)[0]
            if len(not_in) > 0:
                cc = conflict_count[not_in]
                k = min(10, len(not_in))
                idxs = np.argsort(cc)[:k]
                v = not_in[idxs[rng.integers(k)]]
                
                k_conf = conflict_count[v]
                delta = 1 - k_conf
                
                if delta > 0 or rng.random() < np.exp(delta / T):
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            conflict_count[adj[u]] -= 1
                    
                    in_set[v] = True
                    current_size += delta
                    conflict_count[adj[v]] += 1
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
            
            # Try Remove Move with low probability
            in_set_indices = np.where(in_set)[0]
            if len(in_set_indices) > 0 and rng.random() < 0.05:
                u = in_set_indices[rng.integers(len(in_set_indices))]
                in_set[u] = False
                current_size -= 1
                conflict_count[adj[u]] -= 1
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        # Final polish: add all possible vertices with zero conflicts
        improved = True
        while improved:
            improved = False
            counts = np.zeros(N_total, dtype=np.int32)
            for i in np.where(best_in_set)[0]:
                counts[adj[i]] += 1
            
            candidates = np.where((counts == 0) & (~best_in_set))[0]
            for v in candidates:
                if counts[v] == 0 and not best_in_set[v]:
                    best_in_set[v] = True
                    improved = True
                    for u in adj[v]:
                        counts[u] += 1
        
        best_size = best_in_set.sum()
        return best_in_set, best_size
    
    best_size = 0
    best_in_set = np.zeros(N_total, dtype=bool)
    
    n_restarts = 0
    while time.time() < deadline - 5:
        n_restarts += 1
        
        # Initialize with randomized greedy
        in_set = randomized_greedy()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # Local Search
        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break
        
        ls_time = min(remaining * 0.7, 10.0)
        ls_result, ls_size = local_search(in_set, max_time=ls_time)
        
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
        
        # Polish the best solution
        remaining = deadline - 2 - time.time()
        if remaining > 5 and n_restarts % 5 == 0:
            polish_time = min(remaining * 0.8, 15.0)
            p_result, p_size = local_search(best_in_set, max_time=polish_time)
            if p_size > best_size:
                best_size = p_size
                best_in_set = p_result.copy()
    
    # Extract final vectors
    final_indices = np.where(best_in_set)[0]
    final_vectors = all_vectors[final_indices]
    
    return final_vectors

# EVOLVE-BLOCK-END
