import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        if not S:
            return True
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def greedy_addition(vals, seed):
        """Standard greedy addition to get a baseline."""
        rng = random.Random(seed)
        best = set()
        for _ in range(20):
            shuffled = list(vals)
            rng.shuffle(shuffled)
            S = set()
            for x in shuffled:
                ok = True
                for a in S:
                    if (x - a) in S or (x + a) in S:
                        ok = False
                        break
                if ok:
                    S.add(x)
            if len(S) > len(best):
                best = S
        return best

    def prune_and_greedy(vals, seed, time_limit):
        """
        Strategy:
        1. Build a conflict graph where edge u->v exists if v = u + w for some w in vals, 
           or v = w - u, etc. 
           Actually, let's define conflicts more simply for pruning:
           An element x is "conflicting" with a pair (a,b) if a+b=x.
           We want to remove elements that are involved in many such triples.
           
        2. Iteratively remove the element involved in the most "active" conflicts.
           To make this efficient and "flow-like", we can simulate a peeling process.
           
        3. After pruning, run a greedy addition on the remaining set? 
           No, pruning removes elements from the candidate pool. 
           Wait, if we remove elements, we are finding a subset of the original set.
           The goal is to maximize the size of the sum-free subset.
           Removing elements from the *candidate pool* reduces the pool. 
           This is only good if the removed elements were "bad" (high conflict).
           
        Let's refine: 
        We start with all candidates V.
        We identify "bad" elements. 
        If we remove x from V, we don't add it to the solution. 
        But we might have wanted to add x. 
        So this strategy is: "Find a subset of V to discard such that the remaining set is easier to pack, 
        and the greedy packing of the remaining set is larger than the greedy packing of V."
        
        This is risky. Usually, greedy on V is better than greedy on V\{x} unless x blocks many others.
        
        Alternative interpretation of "min-cut":
        Maybe we model it as: 
        Nodes = elements.
        We want to select a subset S.
        Constraint: No a+b=c in S.
        
        Let's stick to the directive: "iteratively remove nodes with the highest 'conflict out-degree' 
        to prune the search space before applying a greedy independent set heuristic on the residual graph."
        
        This implies:
        1. Calculate "conflict out-degree" for each node.
        2. Remove top k nodes from the candidate set V to get V'.
        3. Run Greedy Independent Set (Sum-Free) on V'.
        4. Return the result.
        
        We need to choose k carefully. Or iteratively remove one by one and check if the greedy result improves?
        That's too slow for n=240.
        
        Better: 
        1. Compute conflict counts for all nodes in V.
        2. Sort by count descending.
        3. Try greedy on V (baseline).
        4. Try greedy on V \ {top 1}, V \ {top 2}, ..., V \ {top K}.
        5. Keep the best.
        
        This is a "structured restart" or "pruning" heuristic.
        """
        V = set(vals)
        if not V:
            return set()
        
        # Precompute conflict counts
        # x is in a conflict if exists a,b in V such that a+b=x.
        # Or if x+a=b for some a,b in V.
        # Since a+b=x is equivalent to x-a=b, we can just check:
        # For each x in V, count how many pairs (a,b) in V satisfy a+b=x.
        # This is O(N^2) per x, total O(N^3). For N=240, 240^3 = 13.8M. This is fast enough in Python if optimized? 
        # 13.8M operations might take ~1-2 seconds. 
        # We have 145s. We can afford it.
        
        # Optimization: Use sets for V.
        # For each x in V:
        #   count = 0
        #   for a in V:
        #       if x - a in V:
        #           count += 1
        #   # Note: if a == x-a, i.e., 2a=x, it's one pair. 
        #   # The problem says a=b allowed. 
        #   # Does "a+b=c" count (a,a) as one conflict or does it matter?
        #   # If 2a=x, then a+a=x is a conflict. 
        #   # The loop `for a in V: if x-a in V` will count (a, x-a) and (x-a, a) as two iterations if a != x-a.
        #   # If a == x-a, it counts as one.
        #   # This is fine for a heuristic "degree".
        
        counts = {}
        V_list = list(V)
        for x in V_list:
            c = 0
            for a in V_list:
                if x - a in V:
                    c += 1
            counts[x] = c
            
        # Sort by count descending
        sorted_V = sorted(V_list, key=lambda x: counts[x], reverse=True)
        
        best_set = set()
        baseline = greedy_addition(vals, seed)
        best_set = baseline
        
        # Try removing top K elements
        # K can be 0 to maybe 50?
        # Each greedy run is O(N^2) or O(N * current_size). 
        # Greedy on N=240 is fast (~1ms).
        # We can do many iterations.
        
        max_removals = min(100, len(V))
        
        for k in range(0, max_removals + 1):
            if time.time() > time_limit:
                break
            
            if k == 0:
                remaining = V
            else:
                # Remove first k from sorted_V
                removed = set(sorted_V[:k])
                remaining = V - removed
            
            # Run greedy on remaining
            # To be fast, we can use a fixed seed or multiple seeds?
            # Let's use a deterministic greedy first (sorted order) or random?
            # Random is better for quality.
            # But we have time limits.
            
            # Let's use a fast greedy: 
            # Sort remaining by "safety" (low conflict count in the remaining set)?
            # Or just random shuffle with a fixed seed for this sub-problem?
            
            # Actually, the "greedy independent set heuristic" should be robust.
            # Let's use the same greedy_addition but on the subset.
            # To save time, we'll do fewer random restarts for the pruned versions.
            
            S = greedy_addition(list(remaining), seed + k)
            
            if len(S) > len(best_set):
                best_set = S
                
        return best_set

    def local_search_improve(S, vals, seed, time_limit):
        """
        Improve the solution using local search (add/swap).
        """
        S = set(S)
        if not S:
            return S
            
        current_size = len(S)
        best_set = S
        
        rng = random.Random(seed)
        
        # Precompute which elements are candidates (in vals but not in S)
        candidates = [x for x in vals if x not in S]
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add
            rng.shuffle(candidates)
            for x in candidates:
                if x in S:
                    continue
                ok = True
                for a in S:
                    if (x - a) in S or (x + a) in S:
                        ok = False
                        break
                if ok:
                    S.add(x)
                    candidates.remove(x)
                    current_size += 1
                    if current_size > len(best_set):
                        best_set = set(S)
                    improved = True
                    break
                    
            if improved:
                continue
            
            # 2. Try to swap: remove one, add two?
            # Or remove one, see if we can add more.
            # Simple: Remove a random element, then try to add greedily.
            
            if len(S) > 1:
                # Pick an element to remove. 
                # Heuristic: Pick one with high conflict count in S?
                # Or random.
                # Let's try random for diversity.
                to_remove = rng.choice(list(S))
                S.remove(to_remove)
                
                # Now try to add elements from candidates + [to_remove]
                # Actually, to_remove is now a candidate.
                new_candidates = candidates + [to_remove]
                rng.shuffle(new_candidates)
                
                added_any = False
                for x in new_candidates:
                    if x in S:
                        continue
                    ok = True
                    for a in S:
                        if (x - a) in S or (x + a) in S:
                            ok = False
                            break
                    if ok:
                        S.add(x)
                        if x == to_remove:
                            # We added back the one we removed? 
                            # If we add it back, we are back to previous state (plus maybe others).
                            # If we added others, we improved.
                            pass
                        added_any = True
                        
                if len(S) > current_size:
                    current_size = len(S)
                    if current_size > len(best_set):
                        best_set = set(S)
                else:
                    # If we didn't improve, revert? 
                    # The directive doesn't say we must keep the best, but usually local search keeps the best.
                    # However, "hill climbing" with restarts is better.
                    # Let's revert if we didn't improve to avoid wandering.
                    # But we removed to_remove. We need to put it back if we didn't improve?
                    # Actually, if we didn't improve, the current S is worse or equal.
                    # If equal, it's a different configuration. 
                    # If worse, we should revert.
                    # Let's just keep the best found so far.
                    pass
            
            # If no improvement after a few steps, break?
            # The loop condition handles time.
            
        return best_set

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        vals = instance(size, seed)
        
        # 1. Get a good initial solution using the Pruning + Greedy strategy
        # Allocate time based on size?
        # n=60: 5s, n=120: 10s, n=240: 20s for pruning phase?
        # Then use remaining time for local search.
        
        t_start = time.time()
        time_for_pruning = 5.0 if size == 60 else (10.0 if size == 120 else 20.0)
        
        S_pruned = prune_and_greedy(vals, seed, t_start + time_for_pruning)
        
        # 2. Also try standard greedy in case pruning hurt
        S_greedy = greedy_addition(vals, seed)
        
        # 3. Choose the best initial
        if len(S_greedy) > len(S_pruned):
            S_initial = S_greedy
        else:
            S_initial = S_pruned
            
        # 4. Local Search with remaining time
        time_remaining = deadline - 5.0 - time.time() # 5s safety buffer
        if time_remaining > 1.0:
            S_final = local_search_improve(S_initial, vals, seed, time.time() + time_remaining)
        else:
            S_final = S_initial
            
        # Verify
        if not is_sum_free(S_final):
            # Fallback to a simple valid set (e.g., all odds or all evens, whichever is larger)
            # Odds are sum-free? a+b (odd+odd=even) != c (odd). Yes.
            # Evens are sum-free? a+b (even+even=even) = c (even). No, evens are NOT sum-free.
            # So all odds is always a valid sum-free subset.
            odds = [x for x in vals if x % 2 != 0]
            if len(odds) > len(S_final):
                S_final = set(odds)
            else:
                # If S_final is invalid and odds is smaller, we need a valid set.
                # Greedy addition always produces a valid set.
                S_final = S_greedy
                
        results.append(sorted(S_final))
        
    return results

# Fallback for unexpected errors
try:
    _ = construct
except NameError:
    pass

if __name__ == "__main__":
    # Test
    res = construct()
    print([len(r) for r in res])
