"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    """
    Count the number of unordered pairs (i, j) with i < j such that 
    sets[i] is a strict subset of sets[j] or sets[j] is a strict subset of sets[i].
    Uses bitmask representation for efficiency.
    """
    n = P['n']
    m = len(sets)
    # Convert lists to bitmasks
    masks = [0] * m
    for i in range(m):
        mask = 0
        for v in sets[i]:
            if 0 <= v < n:
                mask |= (1 << v)
        masks[i] = mask
    
    count = 0
    for i in range(m):
        mi = masks[i]
        for j in range(i + 1, m):
            mj = masks[j]
            inter = mi & mj
            if inter == mi and mi != mj:
                count += 1
            elif inter == mj and mi != mj:
                count += 1
    return count

def construct():
    n = P['n']
    m = P['m']
    
    # Deadline guard
    start_time = time.time()
    max_time = 85.0  # 5 seconds safety margin

    # Pre-compute all 2^n subsets as masks and their popcounts
    all_masks = list(range(1 << n))
    # Sort by size to help structure? Or just iterate.
    
    # Strategy: Greedy Construction
    # We want to select 40 distinct subsets that maximize comparability.
    # A set A is comparable to B if A subset B or B subset A.
    # This is equivalent to saying A and B are comparable in the subset lattice.
    
    # Greedy approach:
    # 1. Start with an empty list of selected sets.
    # 2. At each step, pick the unselected set S that maximizes the number of 
    #    comparable pairs it forms with the already selected sets.
    #    Tie-breaker: pick the one with size closer to n/2? Or random?
    #    Actually, to maximize total comparability, we want "hubs".
    #    Small sets are subsets of many larger sets. Large sets are supersets of many smaller sets.
    #    Sets of size n/2 are comparable with fewer sets (only those strictly containing or contained in them).
    #    Specifically, a set of size k is comparable with:
    #    - All subsets of size < k (it is a superset)
    #    - All supersets of size > k (it is a subset)
    #    - None of the sets of size k (unless equal, which is not allowed)
    #    
    #    So, to maximize comparability, we should pick sets of extreme sizes?
    #    If we pick the empty set, it is a subset of ALL other 4095 sets.
    #    If we pick the full set, it is a superset of ALL other 4095 sets.
    #    If we pick a set of size 1, it is a subset of all sets containing it (2^11 - 1 = 2047) and a superset of the empty set (1). Total comparable: 2048.
    #    If we pick a set of size 6, it is a subset of C(12,7)+...+C(12,12) and superset of C(12,0)+...+C(12,5).
    #    C(12,6) is the middle. The number of comparable sets for a size-k set is 2^12 - 1 - C(12,k).
    #    This is maximized when C(12,k) is minimized, i.e., k=0 or k=12.
    #    C(12,0)=1, C(12,12)=1. Max comparability = 4095 - 1 = 4094.
    #    k=1: C(12,1)=12. Max comparability = 4095 - 12 = 4083.
    #    k=11: C(12,11)=12. Max comparability = 4095 - 12 = 4083.
    #    k=6: C(12,6)=924. Max comparability = 4095 - 924 = 3171.
    
    # So, intuitively, we should pick many sets of size 0, 1, 11, 12 to maximize pairwise comparability.
    # However, we are limited to 40 sets.
    # If we pick the empty set, it is comparable to all other 39 sets.
    # If we pick the full set, it is comparable to all other 39 sets.
    # If we pick 38 sets of size 1?
    # Two distinct sets of size 1 are NOT comparable (unless one is subset of other, but they are distinct singletons, so no).
    # So a set of size 1 is comparable with:
    # - Empty set (if present)
    # - Full set (if present)
    # - Any set of size >= 2 that contains it.
    # - Any set of size 0 (empty).
    
    # Let's try a greedy algorithm that picks the best next set based on current selection.
    
    selected_masks = []
    selected_set_tuples = []
    
    # We can pre-compute comparability for all pairs? 4096*4096 is 16M pairs. 
    # 16M checks is fast in Python? Maybe 1-2 seconds.
    # Or we can compute on the fly.
    
    # Let's use a priority queue or just scan all unselected masks each step.
    # 40 steps * 4096 candidates * (check against 40 selected) = 40 * 4096 * 40 = 6.5M operations.
    # This is very fast.
    
    unselected = set(all_masks)
    
    # To break ties or guide the search, we might want to start with a specific set?
    # Or just pick the one with max potential?
    # Initially, all sets have the same potential (comparable with 0 selected).
    # Let's pick the empty set first? Or full set?
    # Actually, the choice of the first set doesn't matter for the count (0 pairs).
    # But it affects the second set.
    # Let's pick the empty set (mask 0) first. It is comparable with everything.
    
    # To add some diversity in the search space in case greedy gets stuck in a local optimum 
    # (though for this structure, greedy is often strong), we can run multiple restarts 
    # with different random tie-breaking or initial seeds.
    
    best_score = -1
    best_sets = []
    
    # Run a few iterations of greedy with random tie-breaking to find a good structure
    # Since the landscape is smooth (comparability is monotonic-ish), one greedy pass might be enough,
    # but randomizing the order of evaluation or tie-breaking can lead to different local optima.
    
    # Actually, let's just do a pure greedy with a specific tie-breaking:
    # Tie-break by preferring smaller sets? Or larger?
    # Or random.
    
    # Let's try 50 random greedy runs to be safe, and keep the best.
    # Each run is ~6.5M ops. 50 runs = 325M ops. Might be too slow in Python (10-20s?).
    # Let's optimize the inner loop.
    
    # Pre-compute for each mask, the list of masks it is comparable to?
    # No, memory is 4096 * avg_comparables. Avg is ~2000. 8M ints. 64MB. Acceptable.
    # But building it takes time.
    
    # Alternative: Just compute the score incrementally.
    
    def greedy_construct(random_seed=42):
        rng = random.Random(random_seed)
        unselected = set(all_masks)
        selected = []
        
        # Pick first set randomly from extremes? Or just 0?
        # Let's pick 0 (empty set) as it's a "hub".
        # Or pick a random set.
        first_mask = rng.choice(all_masks)
        unselected.remove(first_mask)
        selected.append(first_mask)
        
        current_score = 0
        
        for _ in range(m - 1):
            best_candidate = -1
            best_delta = -1
            best_tie = 0
            
            # To speed up, we can skip candidates that are clearly worse?
            # For each candidate c in unselected:
            # delta = number of s in selected such that c comparable s
            # We want to maximize delta.
            
            # Optimization: 
            # If we have many selected sets, the delta is the sum of comparabilities.
            
            for c in unselected:
                delta = 0
                for s in selected:
                    # Check if c and s are comparable
                    # c subset s: (c & s) == c
                    # s subset c: (c & s) == s
                    inter = c & s
                    if inter == c or inter == s:
                        if c != s: # They are distinct because c is unselected and s is selected
                            delta += 1
                
                if delta > best_delta:
                    best_delta = delta
                    best_candidate = c
                    # Tie breaking: randomize among top candidates?
                    # Or prefer specific sizes?
                    # Let's just take the first one found with max delta.
                    # To introduce randomness, we can shuffle unselected? 
                    # No, unselected is a set. We can create a list and shuffle? 
                    # That's expensive.
                    # Let's just collect all with max delta and pick one randomly.
                    # But that requires two passes or storing candidates.
                    # Given the small m, let's just pick the first one.
                    # To get diversity across restarts, the random_seed changes the initial set,
                    # which propagates through the choices.
                
                # Early exit if delta reaches theoretical max?
                # Max delta is len(selected). If we find one, we can break?
                # No, because we want to pick the *best* candidate for the *next* steps too?
                # No, greedy only looks at immediate gain.
                # So if delta == len(selected), it's perfect. We can break.
                if delta == len(selected):
                    best_candidate = c
                    best_delta = delta
                    break
            
            if best_candidate == -1:
                # Should not happen
                break
                
            unselected.remove(best_candidate)
            selected.append(best_candidate)
            current_score += best_delta
            
        # Convert masks to lists
        res = []
        for mask in selected:
            lst = []
            for i in range(n):
                if (mask >> i) & 1:
                    lst.append(i)
            res.append(lst)
            
        return res, current_score

    # Run multiple greedy restarts
    # Time budget: 85s.
    # Let's run as many as possible.
    # Estimate time for one run:
    # 40 steps.
    # Step k: iterate ~4000 candidates, inner loop k.
    # Sum_{k=1}^{39} 4000 * k = 4000 * (39*40/2) = 4000 * 780 = 3.12M operations.
    # Each operation is a few bit ops and comparisons.
    # In Python, ~10-50M ops/sec. So ~0.1-0.3s per run.
    # We can do 100-200 runs in 85s.
    
    runs = 0
    while time.time() - start_time < max_time - 2.0:
        seed = runs
        res, score = greedy_construct(seed)
        
        # Validate distinctness (should be guaranteed by construction)
        # But let's double check types and distinctness
        if len(res) == m:
            # Check distinctness
            seen = set()
            valid = True
            for s in res:
                t = tuple(sorted(s))
                if t in seen:
                    valid = False
                    break
                seen.add(t)
            
            if valid and score > best_score:
                best_score = score
                best_sets = res
                # print(f"Run {runs}: Score {score}")
        
        runs += 1
    
    # If no valid result (shouldn't happen), fallback to random
    if not best_sets:
        # Fallback: Random distinct sets
        all_subsets = []
        for mask in range(1 << n):
            s = [b for b in range(n) if (mask >> b) & 1]
            all_subsets.append(s)
        random.shuffle(all_subsets)
        best_sets = all_subsets[:m]
        best_score = _count_comparable_pairs(best_sets)
        
    # Final check and conversion
    result = []
    for s in best_sets:
        # Ensure they are lists of Python ints
        result.append([int(x) for x in s])
        
    return result
# EVOLVE-BLOCK-END
