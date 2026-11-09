import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values, budget):
        n = len(values)
        if n <= 2:
            return list(values)

        v_set = set(values)
        
        # Precompute conflicts for O(1) check?
        # Conflict exists between i and j if values[i] + values[j] is in v_set.
        # We need to check if adding k to S creates conflicts.
        # A conflict is created if there exists a in S such that a + k in v_set (and a+k in S? No, a+k must be in S).
        # Wait, the condition is: no a,b,c in S such that a+b=c.
        # If we add k to S, we must ensure:
        # 1. For all a in S, a+k is NOT in S. (k + a = c)
        # 2. For all a in S, k+a is NOT in S. (Same as 1)
        # 3. k+k is NOT in S. (2k = c)
        
        # So, k is "safe" to add to S if:
        # (S + k) ∩ S = empty AND 2k not in S.
        # Where S+k = {x+k | x in S}.
        
        # This check is O(|S|).
        # To speed this up, we can maintain a set of "forbidden" additions.
        # But S changes.
        
        # Let's use the structural initialization.
        
        best_solution = []
        
        # 1. Structural Initialization
        # Try moduli 3, 4, 5, 6, 7, 10
        for m in [3, 4, 5, 6, 7, 10]:
            # Find best sum-free subset of residues Z_m
            # Brute force all subsets of Z_m? 2^m is small for m<=10 (1024).
            best_residue_set = set()
            max_residue_size = 0
            # Actually, we want to maximize the count of numbers in the instance that fall into these residues.
            # So we should evaluate each residue subset R by the count of values in instance with val % m in R.
            
            # Precompute counts for this modulus
            counts = [0] * m
            for v in values:
                counts[v % m] += 1
            
            # Iterate all subsets of residues
            # For m=10, 2^10 = 1024 subsets.
            for mask in range(1 << m):
                # Check if this residue set is sum-free in Z_m
                residues = set()
                valid = True
                for r in range(m):
                    if mask & (1 << r):
                        residues.add(r)
                
                if not residues:
                    continue
                    
                # Check sum-free property in Z_m
                for a in residues:
                    for b in residues:
                        if (a + b) % m in residues:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Count how many values fall into these residues
                    cnt = 0
                    for r in residues:
                        cnt += counts[r]
                    if cnt > max_residue_size:
                        max_residue_size = cnt
                        best_residue_set = residues
            
            if best_residue_set:
                # Extract values
                candidate = [v for v in values if v % m in best_residue_set]
                if len(candidate) > len(best_solution):
                    best_solution = candidate
        
        # Also try "All Odds" and "All Evens" (mod 2)
        odds = [v for v in values if v % 2 != 0]
        if len(odds) > len(best_solution):
            best_solution = odds
        
        evens = [v for v in values if v % 2 == 0]
        if len(evens) > len(best_solution):
            best_solution = evens

        # 2. Local Search Refinement
        # Start with the best structural solution
        S = set(best_solution)
        
        # Helper to check if a value can be added
        def can_add(val, current_set):
            # Check if val + x in current_set for any x in current_set
            # And 2*val not in current_set
            for x in current_set:
                if val + x in current_set:
                    return False
            if 2 * val in current_set:
                return False
            return True

        # Helper to check if a value can be removed (always yes, but we track size)
        
        # Greedy Add Pass
        # Try to add any remaining element
        remaining = [v for v in values if v not in S]
        # Sort by some heuristic? Maybe random?
        random.shuffle(remaining)
        
        improved = True
        while improved and time.time() - start_time < DEADLINE - 10:
            improved = False
            for v in remaining:
                if v in S:
                    continue
                if can_add(v, S):
                    S.add(v)
                    remaining.remove(v)
                    improved = True
                    if time.time() - start_time > DEADLINE - 10:
                        break
        
        # Local Search: Swap
        # Try to swap one element out for one in, or two in?
        # Simple SA/Random Restart on the set
        
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        lambda_penalty = 100
        
        current_size = len(S)
        current_edges = 0 # Should be 0
        
        # We need a fast way to evaluate changes.
        # Since we maintain a valid set, we only consider moves that keep it valid or allow temporary invalidity with penalty.
        # Given the structural start, it's better to stick to valid moves (Hill Climbing) or use SA with strict validity check.
        
        # Let's do a simple Randomized Hill Climbing / Local Search
        # Move: Remove one, Add one (Swap)
        # Or: Add one (if possible)
        
        # We already did Greedy Add. Now try Swaps.
        # If we remove a, we might be able to add b, c which were previously blocked by a.
        
        # To make this efficient, we can just run a few rounds of:
        # 1. Pick random a in S.
        # 2. Remove a.
        # 3. Try to add any from remaining.
        # 4. If size increased, keep. Else, revert.
        
        # This is slow if `remaining` is large.
        # Optimization: Precompute conflicts?
        
        # Let's stick to a simple SA that allows invalid states but penalizes them heavily, 
        # or just valid state hill climbing.
        
        # Valid State Hill Climbing:
        # 1. Try to add a random element from remaining. If valid, accept.
        # 2. Try to swap a random element in S with a random element in remaining. If valid and size doesn't decrease (impossible for swap 1:1), accept.
        # Actually, swap 1:1 keeps size same. We want to increase size.
        # So: Remove a, then try to add as many as possible from remaining. If net gain > 0, keep.
        
        # Let's implement a "Restart" strategy.
        # We have a good start.
        
        best_S = set(S)
        best_size = len(S)
        
        # Time limit for local search
        ls_deadline = start_time + budget
        
        while time.time() < ls_deadline:
            # Try a few random restarts/improvements
            S_current = set(best_S)
            
            # 1. Greedy Add
            remaining = [v for v in values if v not in S_current]
            random.shuffle(remaining)
            for v in remaining:
                if can_add(v, S_current):
                    S_current.add(v)
                    remaining.remove(v)
            
            if len(S_current) > best_size:
                best_size = len(S_current)
                best_S = set(S_current)
            
            # 2. Local Search (Swap/Remove-Add)
            # Try to improve by removing one element and adding others
            for _ in range(100): # Fixed number of local search steps
                if time.time() > ls_deadline:
                    break
                if not S_current:
                    break
                
                # Pick a random element to remove
                a = random.choice(list(S_current))
                S_current.discard(a)
                
                # Try to add elements back
                # Priority: a itself? No, we removed it.
                # Try all remaining elements
                # This is O(|S| * |Remaining|) which is bad.
                # Instead, just try a random few additions?
                # Or, just try to re-add a if it fits (revert)?
                
                # Better: Random Swap
                # Pick a in S, b in Remaining.
                # If b can be added to S - {a}, and a cannot be added to S - {b} (trivial), 
                # and |S - {a} + {b}| == |S|, it's a lateral move.
                # We want |S - {a} + {b}| > |S|? Impossible for 1:1.
                
                # So 1:1 swaps don't increase size.
                # We need 1:2 or 1:3 swaps?
                # Remove a, add b and c.
                
                # Let's just do "Remove a, then Greedy Add".
                # We already did Greedy Add from scratch.
                # Let's do: Remove a, then try to add b, c from remaining.
                
                # To keep it simple and fast, let's just do random restarts of the Greedy Add with different random orders.
                # The structural start is the main gain.
                
                # Let's try a simple SA on the valid states only?
                # Move: Remove a, Add b. (Size constant)
                # This allows escaping local maxima in the "shape" of the set, potentially allowing larger sets later?
                # Unlikely to help much if we already did Greedy Add.
                
                pass # Skip complex local search, rely on multiple Greedy Adds with different seeds/orders

            # Actually, the current code's SA was doing something. 
            # Let's just do 5 restarts of the "Structural Init + Greedy Add" with different moduli or random perturbations.
            
            # Perturb: Remove 5% of elements randomly, then Greedy Add again.
            S_perturbed = set(best_S)
            to_remove = random.sample(list(S_perturbed), max(1, len(S_perturbed) // 20))
            for v in to_remove:
                S_perturbed.discard(v)
            
            remaining = [v for v in values if v not in S_perturbed]
            random.shuffle(remaining)
            for v in remaining:
                if can_add(v, S_perturbed):
                    S_perturbed.add(v)
                    remaining.remove(v)
            
            if len(S_perturbed) > best_size:
                best_size = len(S_perturbed)
                best_S = set(S_perturbed)

        return list(best_S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining_time = DEADLINE - (time.time() - start_time)
        if remaining_time < 10:
            vals = instance(size, seed)
            res = [x for x in vals if x % 2 != 0]
            results.append(res)
            continue

        # Allocate budget
        # 60: ~10s, 120: ~30s, 240: ~60s?
        # Total 100s used, 45s left?
        # Let's say 15s, 30s, 50s.
        if size == 60:
            budget = 15
        elif size == 120:
            budget = 30
        else:
            budget = 50
        
        vals = instance(size, seed)
        res = solve_instance(vals, budget)
        
        if not is_sum_free(res):
            res = [x for x in vals if x % 2 != 0]
        
        results.append(res)

    return results
