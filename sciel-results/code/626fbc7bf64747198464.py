import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    n = 12
    m = 40
    N = 1 << n
    
    # Precompute subset relationships for all possible masks?
    # N = 4096. N^2 = ~16M operations. This is fast enough in Python if optimized, 
    # but we can do better.
    # We need to count comparable pairs.
    # A pair (A, B) is comparable if A < B or B < A.
    # Using bitmasks: A is strict subset of B if (A & B) == A and A != B.
    
    # Strategy:
    # 1. Generate a good initial family.
    #    Random sets of size 0..11 might not be optimal.
    #    Comparability is highest when sizes are close.
    #    If we pick all sets of size k, none are comparable (same size).
    #    If we pick sizes k and k+1, a set of size k is subset of a set of size k+1
    #    only if it is contained in it.
    #    Actually, the maximum number of comparable pairs is achieved by a chain?
    #    No, we have fixed size m=40.
    #    A chain of length 40 has 40*39/2 = 780 comparable pairs.
    #    Can we do better?
    #    If we have two chains, say chain A and chain B.
    #    Pairs within A: 780. Pairs within B: 780.
    #    Cross pairs: depends on structure.
    #    If we take a "grid" of sets, e.g., all subsets of a specific 6-set?
    #    Size 6 set has 2^6=64 subsets. We can pick 40 of them.
    #    In a Boolean lattice B_6, the number of comparable pairs among 40 elements?
    #    The total comparable pairs in B_6 is sum_{k=0}^5 (C(6,k) * C(6-k, 1)) ... no.
    #    Total comparable pairs in B_n is 3^n - 2^n.
    #    For n=6: 3^6 - 2^6 = 729 - 64 = 665.
    #    Wait, 3^n counts pairs (A,B) where A subset B (including equal).
    #    Strict subsets: 3^n - 2^n.
    #    For n=12, max comparable pairs in full lattice is 3^12 - 2^12 = 531441 - 4096 = 527345.
    #    But we only pick 40.
    #    
    #    Let's check the chain strategy again.
    #    A chain of 40 sets: S_1 < S_2 < ... < S_40.
    #    Every pair (i, j) with i<j is comparable.
    #    Count = 40*39/2 = 780.
    #    
    #    Can we exceed 780?
    #    Consider a "broad" structure.
    #    If we pick 20 sets of size 6 and 20 sets of size 7.
    #    No comparability within same size.
    #    Comparability only between size 6 and size 7.
    #    Max edges in bipartite graph between C(12,6) and C(12,7) is limited by containment.
    #    Each size 6 set is contained in (12-6) = 6 size 7 sets.
    #    Each size 7 set contains 7 size 6 sets.
    #    If we pick 20 size 6 and 20 size 7 optimally, how many edges?
    #    This is a bipartite matching problem.
    #    It's unlikely to reach 780 with just two layers because the graph is sparse.
    #    20 * 6 = 120 potential edges max if we pick neighbors, but overlaps reduce it.
    #    So a chain is likely better or close to it.
    #    
    #    What about a "star" or "pyramid"?
    #    Take the empty set {} (1 set).
    #    Take all singletons (12 sets).
    #    Take all pairs containing a specific element, say 0. (11 sets).
    #    Take all triples containing 0,1. (10 sets).
    #    ...
    #    This forms a chain of "cones".
    #    Let's try to construct a family that is a sub-lattice or close to a chain.
    #    
    #    Actually, the problem is maximizing the number of comparable pairs.
    #    This is equivalent to finding a subset of the poset P with max internal edges.
    #    This is NP-hard in general, but for small m=40, local search is good.
    #    
    #    Let's use Simulated Annealing with a bitmask representation.
    #    State: list of 40 masks.
    #    Score: count of comparable pairs.
    #    Move: replace one mask with another random mask, or swap two masks.
    #    
    #    To speed up scoring, we can precompute for each mask in the current set,
    #    how many other sets it is comparable with.
    #    But updating this after a swap is O(m) if we recompute all, or O(1) if we maintain counts?
    #    If we change mask i to new_mask:
    #    - Remove old contributions of i with all j != i.
    #    - Add new contributions of i with all j != i.
    #    - This is O(m) per move.
    #    m=40. 40 operations per move.
    #    We can do ~100,000 moves in 1 second? 40 * 100,000 = 4M ops. Yes.
    #    
    #    Optimization:
    #    Precompute `is_subset[i][j]` for all i,j in 0..4095?
    #    4096 * 4096 booleans is 16M bytes. Acceptable.
    #    Then `is_subset[mask1][mask2]` is O(1) array lookup.
    #    Actually, Python list of lists is slow. Use a flat list or numpy?
    #    Numpy is allowed.
    #    `subset_matrix` of shape (4096, 4096) bool.
    #    Memory: 4096*4096 * 1 byte = 16 MB. Fine.
    #    
    #    Let's build the subset matrix.
    
    import numpy as np
    
    # Build subset matrix
    # subset_matrix[a, b] is True if a is strict subset of b
    # a < b iff (a & b) == a and a != b
    # This is O(N^2) to build. N=4096. 16M ops. ~1-2 seconds in Python?
    # Might be too slow if done naively.
    # Vectorize with numpy?
    # a_exp = np.arange(N)[:, None]
    # b_exp = np.arange(N)[None, :]
    # subset = (a_exp & b_exp) == a_exp
    # strict = subset & (a_exp != b_exp)
    # This creates 4096x4096 arrays. 16M elements.
    # Numpy handles this fast.
    
    a_exp = np.arange(N)[:, None]
    b_exp = np.arange(N)[None, :]
    subset_mask = (a_exp & b_exp) == a_exp
    strict_subset = subset_mask & (a_exp != b_exp)
    
    # strict_subset[i, j] is True if i is strict subset of j.
    # Comparability of i, j is strict_subset[i, j] or strict_subset[j, i].
    # Since we count unordered pairs, we just sum strict_subset[i,j] for i<j?
    # No, strict_subset[i,j] covers i<j. strict_subset[j,i] covers j<i.
    # They are mutually exclusive for distinct i,j.
    # So total comparable pairs = sum_{i<j} (strict_subset[i,j] + strict_subset[j,i])
    # Or simply: for each pair i,j, check if comparable.
    
    # Let's initialize with a random set of masks.
    # Better: initialize with a chain to have a high starting score.
    # Chain: 0, 1, 3, 7, 15...
    # We need 40 distinct masks.
    # Max chain length in B_12 is 13 (size 0 to 12).
    # So we can't have a single chain of 40.
    # We need multiple chains or a "thick" chain.
    
    # Let's start with a random initialization and let SA find the structure.
    # Or, initialize with 40 random masks, then run SA.
    
    # To make SA faster, we can precompute the "comparability count" for each mask against the current set?
    # No, that changes as the set changes.
    
    # Let's use the numpy matrix for fast lookups.
    # current_masks: list of 40 ints.
    # score: int.
    
    # Initialize
    current_masks = [random.randint(0, N-1) for _ in range(m)]
    # Ensure distinct
    while len(set(current_masks)) < m:
        # Replace duplicates
        for i in range(m):
            if current_masks.count(current_masks[i]) > 1:
                current_masks[i] = random.randint(0, N-1)
    
    def calculate_score(masks):
        # masks is a list of ints
        # Convert to numpy array for vectorized comparison?
        # We need sum of strict_subset[m_i, m_j] for i < j.
        # This is effectively the number of edges in the comparability graph induced by masks.
        # Using the precomputed matrix:
        # indices = np.array(masks)
        # submatrix = strict_subset[np.ix_(indices, indices)]
        # count = np.sum(submatrix)
        # Since strict_subset is asymmetric (strict), submatrix[i,j] is 1 if i<j.
        # The sum of the upper triangle is the count.
        # np.sum(submatrix) counts all i,j where i is subset of j.
        # Since i!=j is enforced, and subset is strict, this is exactly the number of comparable pairs.
        # Wait, if i is subset of j, then strict_subset[i,j] is True.
        # If j is subset of i, strict_subset[j,i] is True.
        # They cannot both be true.
        # So sum of all elements in the submatrix (excluding diagonal which is 0) is the count.
        idx = np.array(masks)
        sub = strict_subset[np.ix_(idx, idx)]
        return int(np.sum(sub))
    
    initial_score = calculate_score(current_masks)
    best_masks = current_masks[:]
    best_score = initial_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    alpha = 0.999
    iterations = 200000
    
    # Precompute random masks? No, just generate on the fly.
    # To speed up, we can precompute a list of all masks? No, N=4096 is small.
    
    # Optimization:
    # Instead of recalculating full score O(m^2) or O(m) with numpy overhead,
    # we can do incremental updates.
    # But numpy overhead for small arrays might be high.
    # Let's stick to numpy for simplicity first, check time.
    # 200k iterations * (numpy ix_ + sum) might be slow.
    # np.ix_ creates fancy indexing.
    # Alternative:
    # Maintain a count array?
    
    # Let's try a different approach for speed:
    # Since m=40 is small, pure Python with bit operations might be faster than numpy overhead for small arrays?
    # Or use a precomputed list of comparable partners for each mask in the current set?
    
    # Let's use the incremental update method with pure Python and the precomputed strict_subset as a list of lists?
    # No, list of lists of bools is memory heavy and slow to access.
    # Use a flat list of ints (0/1) for strict_subset?
    # strict_flat[i * N + j]
    
    # Let's convert strict_subset to a flat numpy array or list.
    # Actually, `strict_subset` is already a numpy array.
    # Accessing `strict_subset[i, j]` is fast in numpy.
    
    # Let's optimize the score calculation in SA:
    # When we swap mask at index i with new_mask:
    # 1. Remove old mask m_old from consideration.
    # 2. Add new mask m_new.
    # Delta score = 
    #   - (comparability of m_old with all other masks)
    #   + (comparability of m_new with all other masks)
    #   Note: comparability of m_old with m_old is 0.
    #   We must exclude the pair (i, i) which is 0 anyway.
    #   So we sum over j != i.
    
    def get_comparability_count(mask, other_masks, exclude_index):
        # Returns number of masks in other_masks (excluding exclude_index) that are comparable with mask
        count = 0
        for j, m in enumerate(other_masks):
            if j == exclude_index:
                continue
            if strict_subset[mask, m] or strict_subset[m, mask]:
                count += 1
        return count
    
    # This is O(m) per evaluation.
    # In SA loop:
    #   pick i, j (swap) or i (replace).
    #   Let's do replace for simplicity first.
    #   old_m = masks[i]
    #   new_m = random
    #   if new_m in masks: continue
    #   delta = get_comparability_count(new_m, masks, i) - get_comparability_count(old_m, masks, i)
    #   if delta > 0 or random < exp(delta/T): accept
    
    # This is 2 * O(m) = O(m) per step.
    # 200,000 steps * 40 ops = 8,000,000 ops.
    # Python can do ~10-50M simple ops/sec.
    # This should fit in 1-2 seconds.
    
    # Let's refine the SA loop.
    
    masks = current_masks[:]
    score = initial_score
    
    # Precompute list of indices for random selection
    indices = list(range(m))
    
    for step in range(iterations):
        if time.time() > deadline - 5:
            break
            
        # Pick a random index to replace
        i = random.choice(indices)
        old_m = masks[i]
        
        # Pick a new random mask
        # To increase diversity, maybe bias towards masks of similar size?
        # Or just uniform.
        new_m = random.randint(0, N - 1)
        
        # Check if new_m is already in masks
        # O(m) check.
        if new_m in masks:
            continue
            
        # Calculate delta
        # We need to count comparabilities of old_m and new_m with the rest of the set (excluding i)
        # We can optimize this by maintaining a count of comparabilities for each mask in the set?
        # Let `comp_count[k]` be the number of masks in `masks` (excluding k) that are comparable with masks[k].
        # Then score = sum(comp_count) / 2.
        # If we replace i:
        #   Remove old_m:
        #     For all j != i, if comparable(old_m, masks[j]), comp_count[j] decreases by 1.
        #     comp_count[i] is removed.
        #   Add new_m:
        #     For all j != i, if comparable(new_m, masks[j]), comp_count[j] increases by 1.
        #     comp_count[i] is set to count of comparable(new_m, masks[j]) for j != i.
        
        # This update is O(m).
        # And we can maintain `score` incrementally.
        
        # Let's implement this.
        
        # First, let's initialize comp_count for the current masks
        # This is O(m^2) once.
        
        # Actually, let's just do the O(m) delta calculation inside the loop, it's simpler.
        # 8M ops is fine.
        
        delta = 0
        # Count comparabilities of new_m with others
        for j in range(m):
            if j == i:
                continue
            if strict_subset[new_m, masks[j]] or strict_subset[masks[j], new_m]:
                delta += 1
        # Subtract comparabilities of old_m with others
        for j in range(m):
            if j == i:
                continue
            if strict_subset[old_m, masks[j]] or strict_subset[masks[j], old_m]:
                delta -= 1
                
        if delta > 0:
            # Accept
            masks[i] = new_m
            score += delta
            if score > best_score:
                best_score = score
                best_masks = masks[:]
        else:
            # Accept with probability exp(delta/T)
            if random.random() < np.exp(delta / T):
                masks[i] = new_m
                score += delta
                
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Return the best found
    # Convert masks to lists of ints
    result = []
    for mask in best_masks:
        # Ensure distinct integers 0..11
        # mask is int.
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
        
    # Verify constraints
    # 40 distinct subsets
    # Types: list of ints
    
    return result

# EVOLVE-BLOCK-END
