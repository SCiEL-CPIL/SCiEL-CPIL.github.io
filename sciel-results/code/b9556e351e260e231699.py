import itertools
import random
import math

def construct():
    n = 12
    m = 40
    
    # 1. Generate all subsets
    all_subsets = []
    for r in range(n+1):
        for combo in itertools.combinations(range(n), r):
            all_subsets.append(frozenset(combo))
            
    num_subsets = len(all_subsets) # 4096
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # 2. Precompute comparability masks
    # comp_masks[i] is an integer with bit j set if all_subsets[i] is a strict subset of all_subsets[j]
    # Note: We need strict subset. A < B.
    # If we just check A < B, we miss B < A.
    # Let's define comparable_mask[i] such that bit j is 1 if i and j are comparable (one is strict subset of other).
    # Since strict subset is antisymmetric, i < j or j < i.
    
    comp_masks = [0] * num_subsets
    for i in range(num_subsets):
        si = all_subsets[i]
        mask = 0
        for j in range(num_subsets):
            if i == j:
                continue
            sj = all_subsets[j]
            if si < sj or sj < si:
                mask |= (1 << j)
        comp_masks[i] = mask
        
    def popcount(x):
        return x.bit_count()
        
    # 3. Helper to calculate score of a family given by a mask
    # Score = number of unordered pairs {i, j} in family such that i and j are comparable.
    # Sum over i in family of (number of j in family comparable to i) / 2.
    def get_score_from_mask(mask):
        score = 0
        # Iterate through set bits
        m_copy = mask
        while m_copy:
            # Get lowest set bit
            lsb = m_copy & -m_copy
            i = lsb.bit_length() - 1
            score += popcount(comp_masks[i] & mask)
            m_copy ^= lsb
        return score // 2

    # 4. Greedy Initialization
    # Start with a random valid family or a simple greedy one.
    # Let's do a fast greedy: pick set with max comparability to existing, then next, etc.
    family_mask = 0
    family_indices = []
    
    # Start with empty set and full set as they are comparable to many things?
    # Empty set is subset of all. Full set is superset of all.
    # They are comparable to everyone.
    init_candidates = [0, (1 << n) - 1] # Indices for empty and full set
    # Wait, index 0 is empty? 
    # combinations range(0) -> () -> frozenset() -> index 0. Yes.
    # Index for full set? 
    # It's the last one generated? 
    # Let's just find them.
    empty_idx = subset_to_idx[frozenset()]
    full_idx = subset_to_idx[frozenset(range(n))]
    
    family_indices = [empty_idx, full_idx]
    family_mask = (1 << empty_idx) | (1 << full_idx)
    
    # Greedy fill
    available = [i for i in range(num_subsets) if i not in (empty_idx, full_idx)]
    random.shuffle(available)
    
    for _ in range(m - 2):
        best_idx = -1
        best_gain = -1
        # To speed up, we might not want to check all 4000 every time if we can vectorize, 
        # but 40 * 4000 * popcount is ~160k ops, which is fast in Python if popcount is C-level.
        for idx in available:
            # Gain is number of comparable sets in current family
            gain = popcount(comp_masks[idx] & family_mask)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx != -1:
            family_indices.append(best_idx)
            family_mask |= (1 << best_idx)
            available.remove(best_idx) # This is O(N), total O(N^2) for init, acceptable for N=4096, M=40
            
    current_score = get_score_from_mask(family_mask)
    best_score = current_score
    best_family_mask = family_mask

    # 5. Simulated Annealing
    # Parameters
    T_start = 100.0
    T_end = 0.1
    alpha = 0.999 # Cooling rate
    max_iterations = 50000 # Adjust based on time limit
    
    # Estimate time per iteration.
    # One iteration: 
    # 1. Pick random i in family (O(1) if we maintain list, but we have mask. We can maintain list too).
    # Let's maintain a list of indices for the family to pick random swaps easily.
    current_family_list = family_indices[:]
    
    # Re-initialize list to match mask
    current_family_list = []
    m_copy = family_mask
    while m_copy:
        lsb = m_copy & -m_copy
        i = lsb.bit_length() - 1
        current_family_list.append(i)
        m_copy ^= lsb
        
    T = T_start
    best_family_list = current_family_list[:]
    
    # Precompute a list of all indices for random sampling
    all_indices = list(range(num_subsets))
    
    import time
    start_time = time.time()
    time_limit = 55.0 # Leave 5s for overhead
    
    while T > T_end and (time.time() - start_time) < time_limit:
        # Pick a random position in the family to swap
        pos = random.randint(0, m - 1)
        old_idx = current_family_list[pos]
        
        # Pick a random new index not in family
        # To avoid O(N) search for "not in family", we can just pick random and check.
        # With 4096 total and 40 in family, probability of collision is ~1%.
        while True:
            new_idx = random.choice(all_indices)
            if not (family_mask & (1 << new_idx)):
                break
                
        # Calculate delta score
        # Remove old_idx, add new_idx
        # Score change = (comparables of new_idx with others) - (comparables of old_idx with others)
        # Others = family_mask without old_idx
        
        others_mask = family_mask ^ (1 << old_idx)
        
        # Comparables of old_idx with others
        old_comps = popcount(comp_masks[old_idx] & others_mask)
        
        # Comparables of new_idx with others
        new_comps = popcount(comp_masks[new_idx] & others_mask)
        
        delta = new_comps - old_comps
        
        # Acceptance criterion
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept
            family_mask ^= (1 << old_idx)
            family_mask |= (1 << new_idx)
            current_family_list[pos] = new_idx
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family_mask = family_mask
                best_family_list = current_family_list[:]
        
        # Cool
        T *= alpha
        
    # Convert best family mask to result
    result = []
    m_copy = best_family_mask
    while m_copy:
        lsb = m_copy & -m_copy
        i = lsb.bit_length() - 1
        result.append(sorted(list(all_subsets[i])))
        m_copy ^= lsb
        
    # Ensure exactly 40 distinct subsets
    # The SA logic maintains distinctness by checking mask.
    # The size is fixed at m.
    
    return result
