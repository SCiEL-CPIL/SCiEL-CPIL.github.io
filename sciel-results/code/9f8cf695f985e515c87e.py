"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute conflict matrix: conflict if dot > 2
    # For W4 vectors, dot product can be -4,-2,0,2,4
    # If dot > 2, i.e., dot == 4, they are identical (same vector)
    # Wait, distinct vectors can have dot=4 only if they are the same.
    # But we are selecting a subset of DISTINCT vectors.
    # The condition is min pairwise distance >= max norm.
    # Norm of W4 is sqrt(4) = 2.
    # Distance squared = ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 4 + 4 - 2*v_i.v_j = 8 - 2*v_i.v_j.
    # We need distance >= 2, so distance^2 >= 4.
    # 8 - 2*v_i.v_j >= 4 => 2*v_i.v_j <= 4 => v_i.v_j <= 2.
    # So conflict if v_i.v_j > 2. Since max dot is 4 (identical), and next is 2.
    # So distinct W4 vectors with dot=2 are NOT conflicts.
    # Only identical vectors have dot=4. But we don't have duplicates in our candidate list?
    # Yes, the list contains unique vectors.
    # So for W4 vectors, NO two distinct vectors conflict?
    # Let's check.
    # v1 = (1,1,1,1,0,0,0,0,0,0,0)
    # v2 = (1,1,1,-1,0,0,0,0,0,0,0) -> dot = 1+1+1-1 = 2. OK.
    # v3 = (1,1,-1,-1,0,0,0,0,0,0,0) -> dot with v1 = 1+1-1-1=0. OK.
    # v4 = (1,-1,-1,-1,0,0,0,0,0,0,0) -> dot with v1 = 1-1-1-1 = -2. OK.
    # Is it possible to have dot=4 for distinct vectors? No, only if identical.
    # So ALL W4 vectors are mutually compatible?
    # If so, we can take ALL of them.
    # Number of W4 vectors: C(11,4) * 16 = 330 * 16 = 5280.
    # Plus 22 W1 vectors.
    # Total 5302.
    
    # Let's verify this logic.
    # If we take all W4 vectors, min distance is 2. Max norm is 2. Condition holds.
    # Why did the inspiration use SA?
    # Maybe I am missing something.
    # "min pairwise distance is at least the maximum vector norm"
    # If all norms are 2, min dist must be >= 2.
    # If all pairs have dist >= 2, we are good.
    # For W4 vectors, min dist is 2 (when dot=2).
    # So yes, all W4 vectors are compatible.
    
    # Why did the inspiration score 2.64?
    # Score = k / seed. Seed = 220.
    # 2.64 * 220 = 580.8.
    # My count is 5280 + 22 = 5302.
    # 5302 / 220 = 24.1.
    # This is much higher than 2.64.
    # Did I make a mistake?
    
    # Let's re-read the task.
    # "Construct nonzero integer vectors in R^11 whose minimum pairwise distance is at least the maximum vector norm."
    # "Normalizing each vector to norm 2 certifies a kissing configuration of equal-radius spheres."
    
    # Maybe the inspiration found a different set?
    # Or maybe my logic about dot products is wrong?
    # v_i . v_j = sum(v_ik * v_jk).
    # If supports are disjoint, dot=0.
    # If supports overlap in 1 element: dot = +/- 1.
    # If supports overlap in 2 elements: dot = -2, 0, 2.
    # If supports overlap in 3 elements: dot = -3, -1, 1, 3.
    # If supports overlap in 4 elements: dot = -4, -2, 0, 2, 4.
    
    # Conflict if dot > 2.
    # So dot=3 or 4 is a conflict.
    # Dot=3 is possible if supports overlap in 3 elements and signs match in all 3.
    # Example:
    # v1: 1,1,1,1 (positions 0,1,2,3)
    # v2: 1,1,1,0 (positions 0,1,2,4) -> overlap 0,1,2. Signs 1,1,1. Dot = 1+1+1 = 3.
    # Distance squared = 8 - 2*3 = 2. Distance = sqrt(2) < 2.
    # So YES, there are conflicts!
    # My previous check only considered overlap in 4 or 2.
    # Overlap in 3 with matching signs gives dot=3.
    # Overlap in 3 with one mismatch gives dot=1.
    # Overlap in 3 with two mismatches gives dot=-1.
    # Overlap in 3 with three mismatches gives dot=-3.
    
    # So we need to find an independent set in the graph where edges connect vectors with dot > 2.
    # This is a hard problem. The inspiration used SA to solve it.
    
    # So I should use the inspiration's approach.
    # The current program (score 1.0) just returns W2 vectors.
    # The inspiration (score 2.64) uses W4 + W1 and SA.
    
    # I will combine the robust SA from inspiration 1 with the clear structure of the current program (though current is simple).
    # Actually, the instruction says "Combine the strategy of the current program with the strongest ideas of the inspirations".
    # Current program strategy: Simple construction, no search.
    # Inspiration strategy: Complex search (SA) on a large candidate set.
    # The "strongest idea" is definitely the SA search on W4 vectors.
    # The "structural idea" of the current program is... just being a program?
    # Maybe I should stick to the W4+W1 approach and optimize the SA.
    
    # Let's refine the SA from Inspiration 1.
    # Inspiration 1 uses:
    # 1. Greedy Min-Degree Init
    # 2. SA Local Search
    # 3. Polish
    # Multiple restarts.
    
    # I will implement this carefully.
    
    # Compute conflict matrix
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists
    # For efficiency, store as lists of arrays or just use the matrix in SA.
    # N4 = 5280. Matrix is 5280x5280. 27 million entries. Fits in memory.
    
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing local search."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
                
            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)
            
            min_cands = not_in_set[counts == min_count]
            v = min_cands[rng.integers(len(min_cands))]
            
            k = min_count
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < np.exp(delta / T)
                
            if accept:
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 50
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Min-Degree Init
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 2. SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.6, 15.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # 3. Second SA pass on best found so far (polish)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 10.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
