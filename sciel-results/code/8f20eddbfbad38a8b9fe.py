import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin from 60s limit
    
    N = 256
    n = N
    
    # Precompute reciprocal weights
    recip_float = [0.0] * (N + 1)
    recip_frac = [Fraction(0, 1)] * (N + 1)
    for i in range(1, N + 1):
        recip_float[i] = 1.0 / i
        recip_frac[i] = Fraction(1, i)
    
    # --- Step 1: Greedy initial solution (seed logic) ---
    # This gives a valid 3-AP-free set.
    selected = []
    selected_set = set()
    for x in range(1, N + 1):
        valid = True
        for b in selected:
            a = 2 * b - x
            if a > 0 and a < b and a in selected_set:
                valid = False
                break
        if valid:
            selected.append(x)
            selected_set.add(x)
    
    best_solution = list(selected)
    best_weight = sum(recip_float[x] for x in best_solution)
    
    # --- Step 2: Precompute conflict graph ---
    # Two numbers a, b (a < b) conflict if there exists c such that {a,b,c} is a 3-AP.
    # c = 2b - a (if c > b and c <= N)
    # c = (a+b)/2 (if a+b is even and a < c < b)
    # c = 2a - b (if c < a and c >= 1)
    # If any such c exists in [1, N], then a and b cannot both be in the set.
    # We build a bitset representation of the conflict graph for fast independent set checks.
    # conflict_mask[i] is a Python int where bit j is set if i and j+1 conflict.
    # Indices: 1..N map to bits 0..N-1.
    
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            conflict = False
            # Case 1: a, b, c with c = 2b - a
            c = 2 * b - a
            if c > b and c <= N:
                conflict = True
            # Case 2: a, c, b with c = (a+b)/2
            if (a + b) % 2 == 0:
                c = (a + b) // 2
                if c > a and c < b:
                    conflict = True
            # Case 3: c, a, b with c = 2a - b
            c = 2 * a - b
            if c < a and c >= 1:
                conflict = True
            
            if conflict:
                conflict_mask[a] |= (1 << (b - 1))
                conflict_mask[b] |= (1 << (a - 1))
    
    # --- Step 3: Branch and Bound with bitsets ---
    # We want to maximize sum of reciprocals.
    # State:
    #   current_set: list of selected numbers
    #   current_weight: float sum of reciprocals
    #   available: bitset of numbers that can still be added (conflict-free with current_set)
    #   upper_bound: current_weight + sum(recip[i] for i in available)
    
    # Order vertices by reciprocal weight descending to get tighter bounds early?
    # Or just iterate 1..N?
    # Since reciprocals decrease with index, larger indices have smaller weight.
    # To maximize weight, we should prefer smaller indices.
    # However, small indices have many conflicts.
    
    # Let's use a recursive search.
    # To speed up, we can prune if upper_bound <= best_weight.
    
    # Optimization: Precompute prefix sums of reciprocals for available sets?
    # Available set changes dynamically.
    # We can maintain a list of available indices and sum their weights.
    
    # To make B&B efficient, we need a good ordering of decisions.
    # Let's sort the numbers by reciprocal weight (descending) which is index (ascending).
    # So we consider adding 1, then 2, etc.
    
    # However, standard B&B for MWIS on 256 nodes might be too slow if not pruned well.
    # The conflict graph for 3-APs is sparse.
    # Let's try a simple recursive B&B.
    
    # We will store the best solution found globally.
    global_best_weight = best_weight
    global_best_set = list(best_solution)
    
    # To avoid deep recursion limits or stack overflow, and to manage time,
    # we will implement a limited depth search or a specific heuristic B&B.
    
    # Given the time limit, a full exhaustive B&B might not finish.
    # We will use a "best-first" or "depth-first" with strong pruning.
    # Let's try Depth-First Search with pruning.
    
    # State: (current_set, current_weight, available_mask, available_weight_sum)
    # available_mask is an int.
    # available_weight_sum is a float.
    
    # Initial available mask: all bits 0..N-1 set, except those conflicting with initial set?
    # No, we start from the greedy solution.
    # We can try to *improve* the greedy solution by removing elements and adding others?
    # Or just search the space of all valid sets.
    
    # Searching from scratch with B&B:
    # Start with empty set.
    # available = all bits set.
    # weight = 0.
    
    # But 2^256 is huge. Pruning is key.
    # Upper bound: sum of reciprocals of all available nodes.
    # This bound is loose if available nodes conflict with each other.
    
    # Let's try a different approach: Local Search / Hill Climbing with bitsets.
    # It's faster and often finds good solutions.
    # The prompt suggests B&B, but if it's too slow, we fall back.
    # Let's implement a fast local search first, then maybe a limited B&B if time permits.
    
    # Actually, the "inspiration" mentioned B&B with bitset acceleration finding 1.003.
    # Let's try to implement a robust B&B.
    
    # To make B&B work, we need to order the vertices.
    # Let's order by index (1 to N).
    
    # We'll use a recursive function.
    # To avoid Python recursion overhead, we might need to be careful.
    # Depth is at most N=256. Python default limit is 1000, so it's fine.
    
    # However, the branching factor can be large.
    # We need to prune aggressively.
    
    # Let's try a simpler heuristic search that is guaranteed to finish:
    # 1. Start with greedy solution.
    # 2. Try to add any element that doesn't conflict. (Greedy does this).
    # 3. Try to swap: remove one element, add one or more others.
    
    # Let's implement the local search from inspiration 2, but optimized with bitsets.
    
    # Conflict check with bitsets:
    # A set S is valid if for all a in S, (conflict_mask[a] & mask(S)) == 0?
    # No, conflict_mask[a] contains all b that conflict with a.
    # If a is in S, then no b in conflict_mask[a] can be in S.
    # So S is valid iff for all a in S, (conflict_mask[a] & mask(S)) == 0.
    # This is O(|S|) bit operations.
    
    # Let's define a helper to check validity of a mask.
    def is_valid_mask(mask):
        # Check if the set represented by mask is independent
        # Iterate through set bits
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1 # 0-based index
            val = idx + 1 # 1-based value
            # Check if any neighbor of val is in mask
            if conflict_mask[val] & mask:
                return False
            m -= lsb
        return True
    
    # This is O(N) per check.
    
    # Local Search:
    # Current solution: mask, weight
    # Try to add a vertex:
    #   For each v not in mask:
    #     if (conflict_mask[v] & mask) == 0:
    #        new_mask = mask | (1 << (v-1))
    #        new_weight = weight + recip_float[v]
    #        if new_weight > best_weight: update
    #        # We can keep adding until no more can be added (greedy fill)
    
    # Try to swap:
    #   For each u in mask:
    #     remove u
    #     try to add any v not in new_mask that is compatible
    
    # Let's do a "Greedy Fill" after each removal to see if we can gain.
    
    current_mask = 0
    for x in best_solution:
        current_mask |= (1 << (x - 1))
    current_weight = best_weight
    
    # Greedy Fill function: given a mask, add any compatible vertices greedily
    def greedy_fill(mask, weight):
        # Iterate v from 1 to N
        # If v not in mask and compatible, add it
        # This is not optimal but fast.
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if mask & bit:
                continue
            if (conflict_mask[v] & mask) == 0:
                mask |= bit
                weight += recip_float[v]
        return mask, weight
    
    # Initial greedy fill (should recover best_solution or better)
    current_mask, current_weight = greedy_fill(0, 0.0)
    if current_weight > global_best_weight:
        global_best_weight = current_weight
        # Extract set
        s = []
        m = current_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s.append(idx + 1)
            m -= lsb
        global_best_set = s
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_mask = current_mask
        best_w = current_weight
        
        # Try removing each element
        for u in range(1, N + 1):
            if time.time() > deadline:
                break
            bit_u = 1 << (u - 1)
            if not (best_mask & bit_u):
                continue
            
            # Remove u
            trial_mask = best_mask & ~bit_u
            trial_weight = best_w - recip_float[u]
            
            # Greedy fill the trial set
            trial_mask, trial_weight = greedy_fill(trial_mask, trial_weight)
            
            if trial_weight > best_w:
                best_w = trial_weight
                best_mask = trial_mask
                improved = True
        
        # Try swapping?
        # Removing one and adding one is covered by greedy fill if the added one is better?
        # Not necessarily. Greedy fill adds *all* compatible ones.
        # So if removing u allows adding v, w, x, and their sum > recip[u], we gain.
        # This is captured by the removal + greedy fill step.
        
        # What if we need to remove two elements to add three?
        # That's a 2-swap.
        # We can try random 2-swaps if time permits.
        
        current_mask = best_mask
        current_weight = best_w
        
        if current_weight > global_best_weight:
            global_best_weight = current_weight
            s = []
            m = current_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                s.append(idx + 1)
                m -= lsb
            global_best_set = s
    
    # If time remains, try a more sophisticated B&B on a subset or random restarts?
    # Given the constraints, the local search with greedy fill is a strong baseline.
    # The B&B idea is good but risky for time.
    
    # Let's try one more thing: Random restarts with local search.
    # But we have a deadline.
    
    # Final verification
    result = sorted(global_best_set)
    result_set = set(result)
    
    # Verify no 3-AP
    for a in result:
        for b in result:
            if a >= b: continue
            c = 2 * b - a
            if c > b and c in result_set:
                # Invalid
                return best_solution # Fallback to known valid
            if (a + b) % 2 == 0:
                c = (a + b) // 2
                if c > a and c < b and c in result_set:
                    return best_solution
            c = 2 * a - b
            if c < a and c in result_set:
                return best_solution
    
    if not result:
        result = [1]
        
    return result
# EVOLVE-BLOCK-END
