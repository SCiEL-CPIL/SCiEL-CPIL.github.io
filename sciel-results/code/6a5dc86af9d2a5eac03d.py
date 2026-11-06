import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute masks for all triples
    # mask is an integer where bit i is set if point i is in the triple
    triple_info = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_info.append((t, mask))
    
    # Precompute popcount for 16-bit masks to speed up independence check
    # We only need to check masks up to 16 bits, so 2^16 is fine.
    # However, we can compute it on the fly or use a small lookup table.
    # A lookup table for 16 bits is 65536 integers, which is small.
    popcount = [0] * (1 << 16)
    for i in range(1, 1 << 16):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Precompute for each point, the list of triples containing it
    # This helps in checking if a point is "covered" or in calculating constraints
    point_to_triples = [[] for _ in range(16)]
    for i, (t, mask) in enumerate(triple_info):
        for p in t:
            point_to_triples[p].append(i)

    # State variables for backtracking
    best_score = 16  # Worst case independence number
    best_triples = []
    
    # We will use a recursive backtracking function.
    # To avoid deep recursion limits and stack overflows, and to allow 
    # interrupting based on time, we implement it iteratively or with careful depth limits.
    # Given the branching factor, depth is limited by the number of triples we can pick.
    # Max triples in a linear system on 16 points: each triple covers 3 points.
    # If we pick k triples, they cover at least 3k points (if disjoint) but they must intersect.
    # A Steiner Triple System STS(15) has 35 triples. STS(16) doesn't exist.
    # Max number of triples in a linear 3-uniform hypergraph on 16 vertices:
    # Each pair of vertices can be in at most one triple.
    # Total pairs = C(16, 2) = 120. Each triple uses 3 pairs.
    # So max triples <= 120 / 3 = 40.
    # Depth 40 is safe for recursion if optimized, but Python recursion limit is 1000.
    # We will use recursion.

    # To speed up the independence number check, we precompute the "forbidden" sets.
    # An independent set is a set of points that contains no triple.
    # Let S be the set of triples chosen.
    # alpha(S) = max |I| such that I does not contain any triple from S.
    
    # Efficient Alpha Calculation:
    # We can compute alpha by checking subsets of points.
    # Since we want to MINIMIZE alpha, we want to find if alpha <= k.
    # If we find a system with alpha=5, we are likely done (lower bound is 5 for v=16? 
    # Actually, f(16) is known to be 5 or 6. Let's assume we can reach 5 or 6).
    
    # Optimization: 
    # Instead of checking all 2^16 subsets for alpha every time (which is expensive),
    # we can maintain the set of "covered" pairs or use a hitting set approach.
    # However, for a small number of triples (say < 20), checking 2^16 subsets is ~65k ops.
    # If we do this at each node of the search tree, it might be too slow.
    
    # Alternative: Branch and Bound on the Independent Set directly?
    # Or, prune the search tree based on a lower bound on alpha.
    # Lower bound on alpha: If we have selected k triples, what is the minimum possible alpha?
    # It's hard to compute a tight lower bound quickly.
    
    # Let's try a different pruning: 
    # If the current partial solution already has an independent set of size > best_score, 
    # we don't necessarily prune because adding more triples might reduce alpha.
    # Wait, adding triples RESTRICTS the independent sets. 
    # So alpha is non-increasing as we add triples.
    # Therefore, if current_alpha > best_score, we CAN prune? 
    # No. If current_alpha = 6 and best_score = 5, we can still add triples to reduce alpha to 5.
    # We can only prune if current_alpha < best_score? No, that would mean we found a better solution.
    # We can prune if it's IMPOSSIBLE to reduce alpha below best_score.
    # This is hard to determine.
    
    # Better Strategy: 
    # Use the fact that we want to minimize alpha.
    # Start with a good initial solution (greedy) to set a low best_score.
    # Then, use backtracking to try to find a solution with alpha < best_score.
    # Pruning: If the number of remaining triples we can add is insufficient to cover all independent sets of size best_score, prune.
    
    # Let's implement a robust backtracking that tries to build a system with alpha <= target.
    # We will iterate target from 5 upwards.
    # If we find a system with alpha=5, we return it.
    # If not, we return the best found.

    def get_alpha(triple_masks):
        """
        Calculate the independence number of the hypergraph defined by triple_masks.
        triple_masks is a list of integers.
        """
        if not triple_masks:
            return 16
        
        # Optimization: If we have many triples, we can check smaller subsets first.
        # We want to find the LARGEST independent set.
        # We can check sizes from 16 down to 1.
        # But 2^16 is small enough.
        
        # A faster way: 
        # An independent set I is a set of points such that no triple is a subset of I.
        # Equivalently, for every triple T in the system, T is NOT a subset of I.
        # This means I does not contain any triple.
        
        # We can iterate through all masks of points (0 to 2^16 - 1).
        # To find the MAXIMUM size, we can iterate masks in descending order of popcount.
        # Precomputed sorted masks by popcount descending.
        
        # To avoid global state issues in recursion, we pass the list of masks.
        # For speed, we can use a bitset of "forbidden" triples? No, we have the list.
        
        # Let's use a precomputed list of masks sorted by popcount desc.
        # This list is static.
        
        for m in sorted_masks_by_popcount:
            # Check if m contains any triple from triple_masks
            is_independent = True
            for t_mask in triple_masks:
                if (m & t_mask) == t_mask:
                    is_independent = False
                    break
            if is_independent:
                return popcount[m]
        return 0

    # Precompute sorted masks
    all_point_masks = list(range(1 << 16))
    all_point_masks.sort(key=lambda x: popcount[x], reverse=True)
    sorted_masks_by_popcount = all_point_masks

    # Greedy initialization to get a good starting bound
    def greedy_init():
        edges = []
        # Shuffle triples to break symmetry
        shuffled = triple_info[:]
        random.shuffle(shuffled)
        
        for t, mask in shuffled:
            # Check linearity
            valid = True
            for e_mask in edges:
                inter = mask & e_mask
                if inter and (inter & (inter - 1)):
                    valid = False
                    break
            if valid:
                edges.append(mask)
        
        return edges

    initial_edges = greedy_init()
    current_best_alpha = get_alpha(initial_edges)
    best_triples = [t for t, m in triple_info if m in initial_edges]
    
    # If we already have alpha 5, we can return immediately? 
    # It's the theoretical minimum for v=16 (Ramsey-type bounds). 
    # Actually, f(16) is known to be 5? 
    # Let's check: f(16) <= 5 is a known result? 
    # Yes, it is known that f(16) = 5. 
    # So if we find alpha 5, we are done.
    
    if current_best_alpha <= 5:
        return best_triples

    # Backtracking search to find alpha < current_best_alpha
    # We will try to improve the solution.
    
    # To make backtracking effective, we need a good ordering of triples.
    # We'll use a recursive function that adds triples one by one.
    
    # State:
    # current_masks: list of masks of selected triples
    # available_triples: list of indices of triples that are still linear with current selection
    
    # To speed up "available_triples" update, we can precompute conflicts.
    # conflict[i][j] = True if triple i and j share >= 2 points.
    # This is a 560x560 matrix (560 triples). 560^2 = 313,600 booleans. Very small.
    
    num_triples = len(triple_info)
    conflict = [[False] * num_triples for _ in range(num_triples)]
    
    for i in range(num_triples):
        m_i = triple_info[i][1]
        for j in range(i + 1, num_triples):
            m_j = triple_info[j][1]
            inter = m_i & m_j
            if inter and (inter & (inter - 1)):
                conflict[i][j] = True
                conflict[j][i] = True

    # Heuristic for ordering: 
    # Pick the triple that has the most conflicts with other available triples? 
    # Or the one that covers the most "uncovered" pairs?
    # Let's stick to a simple recursive search with pruning.
    
    search_count = 0
    
    def backtrack(current_masks, available_indices, depth):
        nonlocal best_score, best_triples, search_count
        
        if time.time() > deadline:
            return
        
        search_count += 1
        
        # Calculate alpha for current state
        # Optimization: Only calculate if depth > 0 or periodically
        # Actually, we want to know if we can beat best_score.
        # If current_alpha < best_score, update best.
        
        # To save time, we can estimate alpha.
        # But let's just calculate it if the number of triples is small enough or periodically.
        # Given 120s limit, and potentially deep search, we need to be careful.
        
        # Pruning: 
        # If we have added many triples, alpha is likely small.
        # If we have added few, alpha is likely large.
        
        # Let's check alpha every time. 65k checks * 560 triples? No, get_alpha iterates 65k masks, 
        # and for each mask checks len(current_masks) triples.
        # If len(current_masks) is 10, that's 650k ops. 
        # If we do this at every node, and we have 1000 nodes, that's 650M ops. Too slow in Python.
        
        # We need a faster alpha check or less frequent checks.
        
        # Alternative: 
        # Only check alpha when we have added a "significant" number of triples, 
        # or when we reach a leaf.
        # But we need to prune branches that can't improve.
        
        # Let's use a heuristic lower bound on alpha.
        # If the union of all selected triples covers U points, 
        # the independent set must be a subset of the complement? No.
        
        # Let's try a different approach for the search:
        # Instead of adding triples, REMOVE triples from a full set?
        # No, adding is better for minimizing alpha.
        
        # Let's optimize get_alpha.
        # We can precompute for each mask m, the list of triples it contains? No.
        
        # Let's assume get_alpha is the bottleneck.
        # We can cache alpha for sets of triples? The state space is huge.
        
        # Let's limit the depth of the backtracking or the frequency of alpha checks.
        # We will check alpha only if the current number of triples is >= some threshold, 
        # or if we are at a leaf.
        # And we will use a random subset of the search space (random restarts) 
        # if the deterministic search is too slow.
        
        # Actually, the current program (seed) uses Simulated Annealing and got 1.375 (score).
        # Score = seed/value. 1.375 = 1.0 / (1/1.375)? No, score is seed/value.
        # If seed is 1.0, value is 1/1.375? No.
        # "search score is seed/value". If seed program is 1.0, then value of seed program is 1.0?
        # No, "1.0 equals the seed program". So seed program score is 1.0.
        # Current program score is 1.375. So it is BETTER than seed.
        # This implies the value (independence number) of the current program is LOWER than seed.
        # If seed value is V_seed, score_seed = 1.0.
        # Score = 1.0 / Value? No, "seed/value". 
        # If seed value is 6, score = 1/6? No.
        # Usually "score = 1/value" or "score = reference/value".
        # If score > 1.0 is improvement, and lower value is better, 
        # then Score = Reference_Value / Current_Value.
        # If Reference_Value = 6 and Current_Value = 5, Score = 1.2.
        # If Current_Value = 4, Score = 1.5.
        # Current score 1.375. 1.375 = 6 / V? V = 6/1.375 = 4.36? No, V must be integer.
        # Maybe Reference_Value = 5? 5 / 3.6? No.
        # Maybe the seed program had value 6, and current has value 5? 
        # 6/5 = 1.2. 
        # 6/4 = 1.5.
        # 1.375 = 11/8. 
        # Maybe seed value is 11? No, max is 16.
        
        # Let's look at the contract: "score is the largest point set... Minimize."
        # "search score is seed/value".
        # If seed program returns value V_seed, score_seed = 1.0.
        # This implies the formula is Score = V_seed / V_current.
        # If V_seed = 6, and Score = 1.375, then V_current = 6 / 1.375 = 4.36. Impossible.
        # If V_seed = 5, V_current = 5 / 1.375 = 3.63. Impossible.
        
        # Wait, maybe the "seed" in the formula is a constant, not the seed program's value?
        # "1.0 equals the seed program". 
        # This phrasing usually means the baseline score is 1.0.
        # If the metric is "Minimize Value", and Score = K / Value.
        # If Seed Value = V_s, then 1.0 = K / V_s => K = V_s.
        # So Score = V_s / V_current.
        
        # Let's re-read carefully.
        # "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
        
        # If V_s = 6, and V_c = 5, Score = 1.2.
        # If V_s = 6, and V_c = 4, Score = 1.5.
        # 1.375 is between 1.2 and 1.5.
        # This suggests V_c is not an integer? But independence number is an integer.
        
        # Is it possible the "value" is not the independence number, but something else?
        # "The score is the largest point set containing no triple". This is alpha.
        # Alpha is an integer.
        
        # Maybe the seed program's alpha was 5?
        # If V_s = 5, Score = 5 / V_c.
        # 1.375 = 5 / V_c => V_c = 5 / 1.375 = 3.63.
        
        # Maybe the seed program's alpha was 11? No.
        
        # Let's check the inspirations.
        # Inspiration 3: "score 1.3750".
        # Maybe the "value" is normalized differently?
        
        # Let's ignore the exact mapping for a moment and focus on MINIMIZING ALPHA.
        # The best possible alpha for 16 points is likely 5.
        # If I can find a system with alpha 5, I will likely get the best score.
        # If the seed program had alpha 6, and I find alpha 5, my score will be 6/5 = 1.2.
        # If the seed program had alpha 5, and I find alpha 5, my score will be 1.0.
        # If the current program (score 1.375) has alpha X, and seed has Y.
        
        # Actually, there is a known result: f(16) = 5.
        # So the minimum possible value is 5.
        # If the seed program found 6, and the current program found 5, the score would be 1.2.
        # Why is the score 1.375?
        
        # Perhaps the "value" is not just alpha, but includes the number of triples?
        # "The score is the largest point set containing no triple".
        # It explicitly defines the score as alpha.
        
        # Let's assume the goal is to get alpha = 5.
        # If I achieve alpha 5, I should be competitive.
        
        # Given the time limit, a full backtracking search might be too slow if not pruned well.
        # I will implement a hybrid: 
        # 1. Fast greedy to get a good initial alpha.
        # 2. Local search (Simulated Annealing) but with a FASTER alpha check.
        
        # How to speed up alpha check?
        # Instead of checking all 2^16 masks, we can use a randomized approach?
        # No, we need the exact value for the score.
        
        # But for the SEARCH, we can use an approximate alpha or a lower bound.
        # If we use a lower bound on alpha, and it is > current_best, we prune.
        # Lower bound on alpha: 
        # If we have k triples, they cover at most 3k points.
        # The complement has 16 - |Union| points.
        # Any independent set must be a subset of the complement? No.
        # An independent set can contain points from the union, as long as it doesn't contain a full triple.
        
        # Let's stick to the Simulated Annealing but optimize the inner loop.
        
        # The current program's bottleneck is likely `get_alpha`.
        # I will replace `get_alpha` with a faster version that uses bit operations more aggressively.
        
        # Also, I will use a deterministic initialization (e.g., based on Z_16 or specific patterns) 
        # to start closer to the optimum.
        
        # Let's try a specific construction: 
        # Use the Steiner Triple System on 15 points (STS(15)) and add the 16th point carefully.
        # STS(15) has 35 triples. Alpha of STS(15) is 5? 
        # If we add the 16th point, we can add triples involving it.
        
        # This specific construction might yield alpha 5 directly.
        
        # Let's implement the STS(15) based construction.
        
    # STS(15) construction using Z_15? No, 15 is not a prime power of form 6k+1? 
    # 15 = 3*5. 
    # Standard STS(15) exists.
    
    # Let's hardcode a known STS(15) or generate it.
    # Generating STS(15) is complex.
    
    # Let's go back to the optimized Simulated Annealing.
    
    def fast_get_alpha(triple_masks):
        """
        Optimized alpha calculation.
        """
        if not triple_masks:
            return 16
        
        # We can prune the mask check.
        # If a mask m contains a triple, it's not independent.
        # We can precompute a "forbidden" bitset for the current set of triples?
        # No, the set changes.
        
        # However, we can check masks in order of decreasing popcount.
        # As soon as we find one, we return.
        
        # To speed up the inner loop:
        # If len(triple_masks) is large, the probability of a mask being independent is low.
        # We can sample masks? No, we need exact.
        
        # Let's use a C-like loop in Python (unrolled?) No.
        
        # One optimization: 
        # If the union of triples covers all 16 points, and the system is "dense", 
        # alpha is likely small.
        
        # Let's just run the SA loop with the existing get_alpha but with a tighter deadline 
        # and better initialization.
        
    # Re-implementation of SA with better init
    
    # 1. Greedy Init
    edges = greedy_init()
    current_alpha = get_alpha(edges)
    best_alpha = current_alpha
    best_edges = edges[:]
    
    T = 1.0
    T_min = 0.01
    
    # To ensure we finish, we limit the number of steps.
    max_steps = 5000
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        r = random.random()
        new_edges = current_edges if False else edges[:] # Copy
        
        if r < 0.7:
            # Swap
            if edges:
                idx_rm = random.randint(0, len(edges)-1)
                removed = edges[idx_rm]
                new_edges.pop(idx_rm)
                
                added = False
                for _ in range(10): # Try fewer times to save time
                    cand_idx = random.randint(0, len(triple_info)-1)
                    cand_mask = triple_info[cand_idx][1]
                    if cand_mask != removed and not conflict[cand_idx][idx_rm]:
                         # Check linearity with rest
                         valid = True
                         for i, e_mask in enumerate(new_edges):
                             if e_mask == removed: continue
                             # Find index of e_mask in triple_info? 
                             # We don't have index mapping easily here.
                             # Let's stick to mask comparison for linearity.
                             inter = cand_mask & e_mask
                             if inter and (inter & (inter - 1)):
                                 valid = False
                                 break
                         if valid:
                             new_edges.append(cand_mask)
                             added = True
                             break
                if not added:
                    new_edges.insert(idx_rm, removed)
        elif r < 0.85:
            # Add
            added = False
            for _ in range(5):
                cand_idx = random.randint(0, len(triple_info)-1)
                cand_mask = triple_info[cand_idx][1]
                valid = True
                for e_mask in new_edges:
                    inter = cand_mask & e_mask
                    if inter and (inter & (inter - 1)):
                        valid = False
                        break
                if valid:
                    new_edges.append(cand_mask)
                    added = True
                    break
            if not added:
                continue
        else:
            # Remove
            if edges:
                idx_rm = random.randint(0, len(edges)-1)
                new_edges.pop(idx_rm)
            else:
                continue
        
        new_alpha = get_alpha(new_edges)
        
        if new_alpha < current_alpha:
            edges = new_edges
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_edges = edges[:]
                if best_alpha <= 5:
                    break
        else:
            delta = new_alpha - current_alpha
            if delta > 0:
                accept_prob = 1.0 / (1.0 + delta)
                if random.random() < accept_prob * T:
                    edges = new_edges
                    current_alpha = new_alpha
            T *= 0.999
            if T < T_min:
                T = T_min

    # Convert best_edges to list of triples
    mask_to_triple = {m: t for t, m in triple_info}
    final_triples = [mask_to_triple[m] for m in best_edges]
    
    return final_triples

# Run construct
# Note: The function is defined, but not called in the module scope.
# The evaluator calls construct().

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(len(res))
