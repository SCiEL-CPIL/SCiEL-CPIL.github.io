import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s wall-clock
    
    # Domain [51, 100]: 50 elements.
    # Any subset of size >= 4 has sum >= 51+52+53+54 = 210 > 200.
    # Thus, only subsets of size <= 3 can sum to 200.
    # We need to find a Maximum Independent Set in the 3-uniform hypergraph of "bad triples".
    # This is equivalent to finding a Minimum Hitting Set (MHS) of the bad triples.
    
    values = list(range(51, 101))
    n_vars = len(values)
    val_to_idx = {v: i for i, v in enumerate(values)}
    
    # Generate all bad triples (x, y, z) such that x < y < z and x+y+z == 200
    # x, y, z in [51, 100]
    triples = []
    # z can range from 53 to 100 (since min x,y is 51,52 -> z=97? No, x<y<z.
    # Min x=51, y=52 -> z=97. Max z=100.
    # Actually, x+y = 200-z.
    # Min x+y = 51+52 = 103. So 200-z >= 103 => z <= 97.
    # Wait, if z=100, x+y=100. Min x=51, y=49 (but y<z and y>x).
    # Let's just iterate properly.
    
    for z in range(53, 101):
        target_xy = 200 - z
        # x < y < z
        # x + y = target_xy
        # x < y => x < target_xy / 2
        # y < z => target_xy - x < z => x > target_xy - z = 200 - 2z
        # x >= 51
        
        min_x = max(51, 200 - 2*z + 1)
        max_x = (target_xy - 1) // 2
        
        for x in range(min_x, max_x + 1):
            y = target_xy - x
            if x < y < z:
                triples.append((x, y, z))
                
    if not triples:
        return values
        
    # We want to maximize the number of elements kept, which is equivalent to minimizing the number of elements removed.
    # Let H be the set of removed elements. H must be a hitting set for all triples.
    # Minimize |H|.
    
    # The number of triples is small enough for a simple backtracking search or ILP.
    # Let's estimate the number of triples.
    # z ranges roughly 75..100.
    # For z=100, x+y=100. x>=51, y<100. x=51, y=49 (invalid y<x? No x<y).
    # x < 50. So no solutions for z=100?
    # If z=100, x+y=100. x>=51, y>=52. Min sum 103. So 100 is impossible.
    # Max z is when min x+y is 103. 200-z >= 103 => z <= 97.
    # Min z is when max x+y is 99+100? No x<y<z.
    # Max x < z, Max y < z.
    # x+y = 200-z.
    # Max x is z-2, Max y is z-1. Sum = 2z-3.
    # 200-z <= 2z-3 => 203 <= 3z => z >= 68.
    
    # So z in [68, 97].
    
    # Let's use a simple greedy approach with a twist:
    # Greedy Hitting Set:
    # 1. Start with all triples.
    # 2. Find the element that appears in the most triples.
    # 3. Remove it from the set (add to H).
    # 4. Remove all triples containing it.
    # 5. Repeat until no triples left.
    
    # This greedy is not optimal, but often close.
    # To improve, we can try multiple random orderings or local search.
    # However, for this specific structure, the greedy might be good enough.
    # Let's try to implement a better heuristic:
    # Local Search:
    # Start with a solution from Greedy.
    # Try to swap elements in H to reduce size or maintain size but allow more in kept set?
    # Actually, we just want to minimize |H|.
    # If we have a hitting set H, can we remove an element h from H?
    # If H\{h} is still a hitting set, then |H| decreases.
    # We can iterate this "remove if possible" step.
    
    # Let's combine:
    # 1. Greedy to get initial H.
    # 2. Local search: Try to remove elements from H.
    # 3. Try to add elements to H? No, we want to minimize H.
    # 4. Maybe try to swap h in H with k not in H?
    #    If (H - {h} + {k}) is a hitting set, and maybe it allows removing another element later?
    #    This is complex.
    
    # Let's stick to:
    # 1. Greedy Hitting Set.
    # 2. "Minimize" pass: for each h in H, check if H\{h} is a hitting set. If yes, remove h.
    #    Repeat this pass until no more removals possible.
    
    # This "Minimize" pass is exact for the current set structure, but might get stuck in local optima.
    # Given the small size, this should be very fast.
    
    # Implementation of Greedy + Minimize:
    
    def is_hitting_set(H, triples):
        for t in triples:
            if not any(v in H for v in t):
                return False
        return True
        
    def greedy_hitting_set(triples):
        remaining = [set(t) for t in triples]
        H = set()
        while remaining:
            # Count occurrences
            counts = {}
            for t in remaining:
                for v in t:
                    if v not in H:
                        counts[v] = counts.get(v, 0) + 1
            if not counts:
                break
            max_count = max(counts.values())
            candidates = [v for v, c in counts.items() if c == max_count]
            # Pick one
            v = min(candidates)
            H.add(v)
            # Remove triples containing v
            remaining = [t for t in remaining if v not in t]
        return H
        
    def minimize_hitting_set(H, triples):
        H = set(H)
        changed = True
        while changed:
            changed = False
            for h in list(H):
                # Try removing h
                H_trial = H - {h}
                if is_hitting_set(H_trial, triples):
                    H = H_trial
                    changed = True
        return H
        
    # Run Greedy
    H = greedy_hitting_set(triples)
    
    # Run Minimize
    H = minimize_hitting_set(H, triples)
    
    # Verify time
    if time.time() > deadline - 5:
        # Return result
        kept = [v for v in values if v not in H]
        return kept
    
    # Try to improve with a few random restarts of the greedy?
    # Or try different tie-breaking in greedy?
    # Let's try 10 random restarts with random tie-breaking.
    
    best_H = H
    best_size = len(H)
    
    import random
    random.seed(42)
    
    for _ in range(50): # 50 restarts
        if time.time() > deadline - 10:
            break
            
        # Randomized Greedy
        remaining = [set(t) for t in triples]
        H_rand = set()
        while remaining:
            counts = {}
            for t in remaining:
                for v in t:
                    if v not in H_rand:
                        counts[v] = counts.get(v, 0) + 1
            if not counts:
                break
            max_count = max(counts.values())
            candidates = [v for v, c in counts.items() if c == max_count]
            v = random.choice(candidates)
            H_rand.add(v)
            remaining = [t for t in remaining if v not in t]
            
        H_rand = minimize_hitting_set(H_rand, triples)
        
        if len(H_rand) < best_size:
            best_H = H_rand
            best_size = len(H_rand)
            
    # Final check
    # Ensure best_H is valid
    if not is_hitting_set(best_H, triples):
        # Fallback to deterministic greedy
        best_H = greedy_hitting_set(triples)
        best_H = minimize_hitting_set(best_H, triples)
        
    kept = [v for v in values if v not in best_H]
    
    # Final sanity check: no subset sums to 200
    # Since we only removed elements to break triples, and no larger subsets can sum to 200,
    # this should be valid.
    # But let's double check with a quick DP if time permits?
    # No, the logic holds: max sum of 4 is 210.
    # So only subsets of size 1, 2, 3 matter.
    # Size 1: v=200 (not in [51,100])
    # Size 2: x+y=200. x,y in [51,100].
    #    Min x=51, y=149 (out). Max x=100, y=100 (not distinct? "Each element can occur at most once").
    #    Wait, "subset" usually implies distinct elements.
    #    If x=y, 2x=200 => x=100.
    #    Is {100} a subset summing to 200? No, sum is 100.
    #    Is {100, 100} allowed? No, "Each element can occur at most once in the forbidden sum".
    #    So we need distinct x, y.
    #    x+y=200, x<y.
    #    x=51, y=149 (no).
    #    x=99, y=101 (no).
    #    So no pairs in [51,100] sum to 200.
    #    Let's verify: max pair sum is 99+100=199.
    #    So no pairs sum to 200.
    # Size 3: x+y+z=200. These are the triples we handled.
    # Size >= 4: min sum 210.
    
    # So the condition is exactly that no triple sums to 200.
    # Our hitting set ensures no triple is fully contained in the kept set.
    
    return kept

# EVOLVE-BLOCK-END
