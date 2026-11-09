import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s kill

    def get_subset_sums(elements):
        """Return set of all nonempty subset sums."""
        n = len(elements)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    def is_valid(elements):
        """Check if set satisfies both uniqueness and non-divisibility."""
        n = len(elements)
        if n == 0:
            return True
        if n > 16:
            # Too many sums to check efficiently
            return False
        
        # Generate all subset sums and check for duplicates
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums:
                return False
            sums.add(s)
        
        # Check divisibility
        sorted_sums = sorted(sums)
        for i in range(len(sorted_sums)):
            si = sorted_sums[i]
            for j in range(i + 1, len(sorted_sums)):
                if sorted_sums[j] % si == 0:
                    return False
        return True

    def try_add(current, candidate):
        """Check if adding candidate to current set maintains validity."""
        new_set = current + [candidate]
        return is_valid(new_set)

    def greedy_build(order, current=None):
        """Greedy build with given candidate ordering."""
        if current is None:
            current = []
        for cand in order:
            if time.time() > deadline:
                break
            if cand not in current and try_add(current, cand):
                current.append(cand)
        return current

    def local_search(initial, max_iters=200):
        """Local search: try swaps to improve."""
        best = list(initial)
        if not best:
            return best
        
        candidates = list(range(256, 0, -1))
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            for i in range(len(best)):
                if time.time() > deadline:
                    return best
                for cand in candidates:
                    if time.time() > deadline:
                        return best
                    if cand not in best:
                        # Try swapping best[i] with cand
                        new_set = best[:i] + [cand] + best[i+1:]
                        new_set.sort()
                        if is_valid(new_set):
                            # Check if this swap is beneficial
                            # For now, accept any valid swap (we're maximizing size, not minimizing)
                            # But since size stays same, try to improve "quality"
                            # Actually, we want to find configurations that allow more additions
                            best = new_set
                            improved = True
                            break
                if improved:
                    break
        return best

    def try_extend(current):
        """Try to extend current set by adding elements."""
        result = list(current)
        for cand in range(256, 0, -1):
            if time.time() > deadline:
                break
            if cand not in result and try_add(result, cand):
                result.append(cand)
        return result

    # Strategy 1: Superincreasing greedy (fast, good baseline)
    def superincreasing_search():
        best = []
        for start in range(1, 200):
            if time.time() > deadline:
                break
            current = [start]
            running_sum = start
            for cand in range(start + 1, 257):
                if time.time() > deadline:
                    break
                if cand > running_sum and try_add(current, cand):
                    current.append(cand)
                    running_sum += cand
            if len(current) > len(best):
                best = list(current)
        return best

    # Strategy 2: Multiple greedy with different orderings
    def multi_greedy():
        best = []
        random.seed(12345)
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            if attempt == 0:
                order = list(range(256, 0, -1))
            elif attempt == 1:
                order = list(range(1, 257))
            elif attempt == 2:
                order = list(range(256, 0, -1))
                random.shuffle(order)
            else:
                # Random with bias toward larger values
                order = list(range(1, 257))
                random.shuffle(order)
                order.sort(key=lambda x: -x * (0.3 + 0.7 * random.random()))
            
            current = greedy_build(order)
            if len(current) > len(best):
                best = list(current)
        
        return best

    # Strategy 3: Start from known good patterns and extend
    def pattern_search():
        best = []
        
        # Try starting with pairs that are "compatible"
        for a in range(1, 100):
            if time.time() > deadline:
                break
            for b in range(a + 1, 257):
                if time.time() > deadline:
                    break
                if try_add([], a):
                    current = [a]
                    if try_add(current, b):
                        current.append(b)
                        # Now try to extend
                        extended = try_extend(current)
                        if len(extended) > len(best):
                            best = list(extended)
        return best

    # Run all strategies
    best_overall = []
    
    # Strategy 1
    si = superincreasing_search()
    if len(si) > len(best_overall):
        best_overall = si
    
    # Strategy 2
    mg = multi_greedy()
    if len(mg) > len(best_overall):
        best_overall = mg
    
    # Strategy 3
    ps = pattern_search()
    if len(ps) > len(best_overall):
        best_overall = ps
    
    # Local search on best
    if best_overall and time.time() < deadline - 5:
        ls = local_search(best_overall)
        if len(ls) > len(best_overall):
            best_overall = ls
    
    # Final extension
    if best_overall and time.time() < deadline - 2:
        final = try_extend(best_overall)
        if len(final) > len(best_overall):
            best_overall = final

    # Ensure valid and return
    if not is_valid(best_overall) and len(best_overall) > 0:
        # Fallback to a known valid small set
        best_overall = [1, 3, 7]
    
    return [int(x) for x in best_overall]
