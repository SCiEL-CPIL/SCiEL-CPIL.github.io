"""Stronger algorithm for sum-free subset construction."""
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    
    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if set S is sum-free."""
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True
    
    def conflict_count(S):
        """Count number of sum-free violations (a+b=c with a,b,c in S)."""
        Sset = set(S)
        count = 0
        for a in S:
            for b in S:
                if a + b in Sset:
                    count += 1
        return count
    
    def greedy_add(S, candidates):
        """Greedily add candidates to S if they don't create conflicts."""
        Sset = set(S)
        result = list(S)
        for c in candidates:
            # Check if adding c creates any conflict
            ok = True
            for a in result:
                if a + c in Sset or a + c == c:  # a + c in S or a + c = c (i.e., a=0, but no zeros)
                    if a + c in Sset:
                        ok = False
                        break
                if c + a in Sset:
                    if c + a in Sset:
                        ok = False
                        break
            # Also check c + c
            if 2 * c in Sset:
                ok = False
            if ok:
                result.append(c)
                Sset.add(c)
        return result
    
    def local_search(S, all_elements, max_iters=5000):
        """Improve S via local search: try adding elements, removing conflicting ones."""
        Sset = set(S)
        S = list(S)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try adding each element not in S
            for c in all_elements:
                if c in Sset:
                    continue
                # Check if c can be added
                can_add = True
                conflicts = []
                for a in S:
                    s = a + c
                    if s in Sset:
                        can_add = False
                        conflicts.append(a)
                        break
                if 2 * c in Sset:
                    can_add = False
                if can_add:
                    S.append(c)
                    Sset.add(c)
                    improved = True
                    break
            
            if not improved:
                # Try removing an element that's in many conflicts, then re-add others
                # Find element with most conflicts
                best_remove = None
                best_conflicts = 0
                for a in S:
                    cnt = 0
                    for b in S:
                        if a + b in Sset and a + b != a and a + b != b:
                            cnt += 1
                        if 2 * a in Sset and a != 0:
                            cnt += 1
                    if cnt > best_conflicts:
                        best_conflicts = cnt
                        best_remove = a
                
                if best_remove is not None and best_conflicts > 0:
                    # Remove best_remove
                    S.remove(best_remove)
                    Sset.remove(best_remove)
                    improved = True
        
        return S
    
    def solve_instance(elements, time_budget=30):
        """Find a large sum-free subset from elements."""
        n = len(elements)
        start_time = time.time()
        
        # Strategy 1: Take elements by sign and magnitude
        # Positive elements in upper half tend to be sum-free with each other
        # if their sum exceeds max element
        
        # Try multiple initial strategies
        best_result = []
        best_size = 0
        
        # Strategy A: Odd numbers (baseline)
        odds = [x for x in elements if x % 2 != 0]
        if is_sum_free(odds):
            if len(odds) > best_size:
                best_size = len(odds)
                best_result = odds[:]
        
        # Strategy B: Numbers in a specific range (e.g., top half of absolute values)
        abs_vals = sorted(set(abs(x) for x in elements))
        # Elements whose absolute value is in the upper 1/3
        if abs_vals:
            threshold = abs_vals[len(abs_vals) // 3]
            upper = [x for x in elements if abs(x) >= threshold]
            # Check sum-free
            if is_sum_free(upper):
                if len(upper) > best_size:
                    best_size = len(upper)
                    best_result = upper[:]
        
        # Strategy C: Greedy by absolute value (largest first)
        elems_sorted = sorted(elements, key=lambda x: abs(x), reverse=True)
        greedy_set = []
        greedy_set_set = set()
        for c in elems_sorted:
            ok = True
            for a in greedy_set:
                if a + c in greedy_set_set:
                    ok = False
                    break
            if 2 * c in greedy_set_set:
                ok = False
            if ok:
                greedy_set.append(c)
                greedy_set_set.add(c)
        if is_sum_free(greedy_set):
            if len(greedy_set) > best_size:
                best_size = len(greedy_set)
                best_result = greedy_set[:]
        
        # Strategy D: Greedy by value (positive first, then negative)
        # Positive numbers: if all > max/2, they're sum-free among themselves
        pos = [x for x in elements if x > 0]
        neg = [x for x in elements if x < 0]
        
        # For positives, take those in upper range
        if pos:
            max_pos = max(pos)
            upper_pos = [x for x in pos if x > max_pos / 2]
            if is_sum_free(upper_pos):
                if len(upper_pos) > best_size:
                    best_size = len(upper_pos)
                    best_result = upper_pos[:]
        
        # Strategy E: Mod-based coloring
        for mod in [3, 4, 5, 6, 7]:
            for r in range(mod):
                subset = [x for x in elements if x % mod == r]
                if is_sum_free(subset):
                    if len(subset) > best_size:
                        best_size = len(subset)
                        best_result = subset[:]
        
        # Now do local search from the best initial solution
        if best_result:
            improved = local_search(best_result, elements, max_iters=3000)
            if is_sum_free(improved):
                best_result = improved
                best_size = len(improved)
        
        # Try local search from other good initial solutions
        for init in [odds, greedy_set]:
            if not init:
                continue
            if time.time() - start_time > time_budget * 0.7:
                break
            improved = local_search(init, elements, max_iters=1500)
            if is_sum_free(improved) and len(improved) > best_size:
                best_result = improved
                best_size = len(improved)
        
        return best_result
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 10:
            # Not enough time, use simple strategy
            elems = instance(size, seed)
            odds = [x for x in elems if x % 2 != 0]
            results.append(odds)
            continue
        
        elems = instance(size, seed)
        time_budget = min(35, deadline - time.time() - 5)
        result = solve_instance(elems, time_budget)
        results.append(result)
    
    return results
