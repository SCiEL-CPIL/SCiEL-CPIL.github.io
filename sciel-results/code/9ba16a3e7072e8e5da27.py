"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def can_add(x, S_set):
        """Check if x can be added to S_set to keep it sum-free. Single pass."""
        # Conditions for NOT sum-free after adding x:
        # a+b=c where at least one of a,b,c is x
        # Case 1: x + y = z  => z - x in S_set for some y=z-x in S_set, i.e., exists y in S_set s.t. y+x in S_set
        # Case 2: y + x = z  (same as case 1)
        # Case 3: x + x = z  => 2x in S_set
        # Case 4: y + z = x  => x - y in S_set for some y in S_set
        # Case 5: y + y = x  => exists y in S_set s.t. 2y == x
        
        # Combine: for each y in S_set, check if (y+x) in S_set OR (x-y) in S_set OR (2*y == x)
        # Plus check 2x in S_set
        if 2 * x in S_set:
            return False
        for y in S_set:
            if (y + x) in S_set:
                return False
            if (x - y) in S_set:
                return False
            if 2 * y == x:
                return False
        return True

    def greedy_construct(elements, order):
        """Construct a sum-free set by adding elements in the given order if possible."""
        S = []
        S_set = set()
        for x in order:
            if can_add(x, S_set):
                S.append(x)
                S_set.add(x)
        return S

    def greedy_by_range(elements, lower_frac=0.33, upper_frac=1.0):
        """Greedy: prioritize elements in a range that's likely sum-free."""
        if not elements:
            return []
        min_val = min(elements)
        max_val = max(elements)
        span = max_val - min_val
        lower = min_val + lower_frac * span
        upper = min_val + upper_frac * span
        
        # Elements in the "upper" range are more likely to be sum-free
        # Sort by how far into the upper range they are
        def score(x):
            # Prefer elements in (lower, upper], then by magnitude
            if lower < x <= upper:
                return (1, x)
            elif x > upper:
                return (0, x)
            else:
                return (-1, -x)  # Prefer negative small magnitude
        
        order = sorted(elements, key=score, reverse=True)
        return greedy_construct(elements, order)

    def local_search(S, elements, max_time=10, seed=123):
        """Improve the set using local search with targeted swaps."""
        S = list(S)
        S_set = set(S)
        all_elements_set = set(elements)
        excluded = all_elements_set - S_set
        rng = random.Random(seed)
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            improved = False
            
            # Phase 1: Try to add any excluded element that fits
            added_any = False
            for x in list(excluded):
                if can_add(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    excluded.remove(x)
                    added_any = True
                    improved = True
            
            if not added_any:
                # Phase 2: Targeted swaps - for each excluded x, find minimal removal set
                if S and excluded:
                    # Sort excluded by some heuristic: prefer larger absolute values
                    excluded_list = sorted(excluded, key=abs, reverse=True)
                    for x in excluded_list[:min(len(excluded_list), 20)]:
                        # Find all y in S that conflict with x
                        # Conflict means: removing y might allow x to be added
                        # y conflicts if: (y+x) in S_set OR (x-y) in S_set OR (2*y == x) OR (2*x == y)
                        conflicting_y = set()
                        for y in S_set:
                            if (y + x) in S_set:
                                conflicting_y.add(y)
                                conflicting_y.add(y + x)
                            if (x - y) in S_set:
                                conflicting_y.add(y)
                                conflicting_y.add(x - y)
                            if 2 * y == x:
                                conflicting_y.add(y)
                            if 2 * x == y:
                                conflicting_y.add(y)
                        
                        # Try removing single conflicting elements
                        for y in list(conflicting_y & S_set):
                            temp_S_set = S_set - {y}
                            if can_add(x, temp_S_set):
                                # Perform swap
                                S.remove(y)
                                S_set.remove(y)
                                S.append(x)
                                S_set.add(x)
                                excluded.discard(x)
                                excluded.add(y)
                                improved = True
                                break
                        
                        if improved:
                            break
                        
                        # Try removing pairs of conflicting elements
                        if len(conflicting_y & S_set) >= 2:
                            conflict_list = list(conflicting_y & S_set)
                            for i in range(min(len(conflict_list), 5)):
                                for j in range(i+1, min(len(conflict_list), 10)):
                                    y1, y2 = conflict_list[i], conflict_list[j]
                                    temp_S_set = S_set - {y1, y2}
                                    if can_add(x, temp_S_set):
                                        # Perform double swap
                                        S.remove(y1)
                                        S.remove(y2)
                                        S_set.remove(y1)
                                        S_set.remove(y2)
                                        S.append(x)
                                        S_set.add(x)
                                        excluded.discard(x)
                                        excluded.add(y1)
                                        excluded.add(y2)
                                        improved = True
                                        break
                                if improved:
                                    break
                        
                        if improved:
                            break
                            
            if not improved:
                # Phase 3: Perturbation - remove k random elements, then greedily refill
                if S:
                    k = min(len(S), rng.randint(2, 5))
                    to_remove = rng.sample(S, k)
                    for y in to_remove:
                        S.remove(y)
                        S_set.remove(y)
                        excluded.add(y)
                    # Greedily add back
                    for x in sorted(excluded, key=abs, reverse=True):
                        if can_add(x, S_set):
                            S.append(x)
                            S_set.add(x)
                            excluded.remove(x)
                    
        return S

    results = []
    total_start = time.time()
    for idx, (size, seed) in enumerate(((60, 801), (120, 802), (240, 803))):
        elements = instance(size, seed)
        
        # Generate multiple candidates
        candidates = []
        
        # 1. Greedy Descending
        candidates.append(greedy_construct(elements, sorted(elements, reverse=True)))
        
        # 2. Greedy Ascending
        candidates.append(greedy_construct(elements, sorted(elements)))
        
        # 3. Greedy Random (multiple runs)
        rng = random.Random(seed)
        for _ in range(3):
            shuffled = elements[:]
            rng.shuffle(shuffled)
            candidates.append(greedy_construct(elements, shuffled))
        
        # 4. Range-based greedy (upper 2/3)
        candidates.append(greedy_by_range(elements, 0.33, 1.0))
        
        # 5. Range-based greedy (upper 1/2)
        candidates.append(greedy_by_range(elements, 0.5, 1.0))
        
        # 6. Positive elements only
        pos = [x for x in elements if x > 0]
        if pos:
            candidates.append(greedy_construct(elements, sorted(pos, reverse=True)))
        
        # 7. Negative elements only
        neg = [x for x in elements if x < 0]
        if neg:
            candidates.append(greedy_construct(elements, sorted(neg)))
        
        # Select best candidate
        best_candidate = max(candidates, key=len)
        
        # Allocate remaining time
        elapsed = time.time() - total_start
        remaining = 110 - elapsed  # Leave 10s buffer
        time_per_instance = max(5, min(30, remaining / (3 - idx)))
        
        # Local Search with multiple seeds
        best_improved = best_candidate
        for ls_seed in [123, 456, 789]:
            improved_set = local_search(best_candidate, elements, max_time=time_per_instance/3, seed=ls_seed)
            if len(improved_set) > len(best_improved):
                best_improved = improved_set
        
        # Verify
        if not is_sum_free(best_improved):
            best_improved = best_candidate
            
        results.append(best_improved)
    
    return results
# EVOLVE-BLOCK-END
