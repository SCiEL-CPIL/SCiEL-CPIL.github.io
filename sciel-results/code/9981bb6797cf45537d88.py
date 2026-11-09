"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for i, a in enumerate(subset):
            for b in subset[i:]:
                if a + b in s:
                    return False
        return True

    def greedy_sum_free(S):
        """Greedy: sort by absolute value descending, add if doesn't create conflict."""
        S_sorted = sorted(S, key=lambda x: -abs(x))
        result = []
        result_set = set()
        for x in S_sorted:
            # Check if adding x creates a+b=c conflict
            # x could be the sum: a+b=x where a,b in result_set
            # x could be an addend: x+b=c or a+x=c where b,c or a,c in result_set
            conflict = False
            for a in result:
                if x - a in result_set and x - a != a or (x - a in result_set and x - a == a):
                    # a + a = x or a + b = x
                    if x - a in result_set:
                        conflict = True
                        break
                if a + x in result_set:
                    conflict = True
                    break
            if not conflict:
                result.append(x)
                result_set.add(x)
        return result

    def local_search_improve(S, initial, max_time=20.0):
        """Local search: try to add elements, and do 1-swap moves."""
        t_start = time.time()
        current = list(initial)
        current_set = set(current)
        excluded = [x for x in S if x not in current_set]
        
        best_size = len(current)
        
        # Try to add elements greedily
        added = True
        while added and (time.time() - t_start) < max_time:
            added = False
            new_excluded = []
            for x in excluded:
                # Check if x can be added
                conflict = False
                for a in current:
                    if x - a in current_set:
                        # a + (x-a) = x, so conflict if (x-a) in current_set
                        # But we need a, b, c all in set with a+b=c
                        # Here c=x, a=a, b=x-a
                        conflict = True
                        break
                    if a + x in current_set:
                        # a + x = c where c in current_set
                        conflict = True
                        break
                if not conflict:
                    current.append(x)
                    current_set.add(x)
                    added = True
                    best_size = len(current)
                else:
                    new_excluded.append(x)
            excluded = new_excluded
        
        # Local search: 1-swap moves (remove one, add one or two)
        improved = True
        while improved and (time.time() - t_start) < max_time:
            improved = False
            for i, x in enumerate(current):
                # Try removing x and adding one excluded element
                candidate = current[:i] + current[i+1:]
                candidate_set = set(candidate)
                for y in excluded:
                    # Check if y can be added to candidate
                    conflict = False
                    for a in candidate:
                        if y - a in candidate_set:
                            conflict = True
                            break
                        if a + y in candidate_set:
                            conflict = True
                            break
                    if not conflict:
                        new_current = candidate + [y]
                        new_set = candidate_set | {y}
                        if len(new_current) > len(current):
                            current = new_current
                            current_set = new_set
                            excluded = [z for z in excluded if z != y]
                            improved = True
                            best_size = len(current)
                            break
                if improved:
                    break
            # Also try removing one element and adding two
            if not improved and (time.time() - t_start) < max_time:
                for i, x in enumerate(current):
                    candidate = current[:i] + current[i+1:]
                    candidate_set = set(candidate)
                    found_two = False
                    for j, y in enumerate(excluded):
                        if found_two:
                            break
                        # Try adding y
                        conflict_y = False
                        for a in candidate:
                            if y - a in candidate_set:
                                conflict_y = True
                                break
                            if a + y in candidate_set:
                                conflict_y = True
                                break
                        if conflict_y:
                            continue
                        # Now try adding z != y
                        for z in excluded[j+1:]:
                            if z == y:
                                continue
                            # Check if z can be added to candidate + [y]
                            temp_set = candidate_set | {y}
                            conflict_z = False
                            for a in temp_set:
                                if z - a in temp_set:
                                    conflict_z = True
                                    break
                                if a + z in temp_set:
                                    conflict_z = True
                                    break
                            if not conflict_z:
                                new_current = candidate + [y, z]
                                current = new_current
                                current_set = temp_set | {z}
                                excluded = [w for w in excluded if w not in (y, z)]
                                improved = True
                                found_two = True
                                best_size = len(current)
                                break
                    if found_two:
                        break
                if improved:
                    break
        
        return current

    def solve_instance(S):
        """Solve one instance with multiple strategies and pick the best."""
        t_start = time.time()
        
        # Strategy 1: Upper half (x > max/2)
        M = max(S)
        m = min(S)
        upper = [x for x in S if x > M / 2]
        
        # Strategy 2: Lower half (x < min/2)
        lower = [x for x in S if x < m / 2]
        
        # Strategy 3: Odd numbers
        odds = [x for x in S if x % 2 != 0]
        
        # Strategy 4: Greedy by absolute value
        greedy = greedy_sum_free(S)
        
        # Strategy 5: Greedy by value descending
        S_desc = sorted(S, reverse=True)
        result5 = []
        set5 = set()
        for x in S_desc:
            conflict = False
            for a in result5:
                if x - a in set5:
                    conflict = True
                    break
                if a + x in set5:
                    conflict = True
                    break
            if not conflict:
                result5.append(x)
                set5.add(x)
        
        # Strategy 6: Greedy by value ascending
        S_asc = sorted(S)
        result6 = []
        set6 = set()
        for x in S_asc:
            conflict = False
            for a in result6:
                if x - a in set6:
                    conflict = True
                    break
                if a + x in set6:
                    conflict = True
                    break
            if not conflict:
                result6.append(x)
                set6.add(x)
        
        candidates = [upper, lower, odds, greedy, result5, result6]
        
        # Pick the largest initial candidate for local search
        candidates.sort(key=len, reverse=True)
        
        best = candidates[0]
        best_size = len(best)
        
        # Apply local search to top candidates
        for cand in candidates[:4]:
            if (time.time() - t_start) > 10.0:
                break
            improved = local_search_improve(S, cand, max_time=8.0)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > DEADLINE:
            # Return what we have with fallback
            S = instance(size, seed)
            M = max(S)
            results.append([x for x in S if x > M / 2])
            continue
        S = instance(size, seed)
        result = solve_instance(S)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
