"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        """Check if a set is sum-free: no a+b=c with a,b,c in set (a=b allowed)."""
        s = set(subset)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def greedy_construct(arr, order='asc'):
        """Greedy: add elements in given order if sum-free is maintained."""
        subset = []
        s = set()
        for x in arr:
            # Check if adding x violates sum-free
            ok = True
            for a in s:
                if a + x in s or x + a in s:
                    ok = False
                    break
                if 2 * x in s:
                    ok = False
                    break
            if ok:
                subset.append(x)
                s.add(x)
        return subset

    def local_search(subset, arr, max_iters=5000):
        """Try to improve by swapping elements or adding/removing."""
        s = set(subset)
        best = list(subset)
        best_size = len(best)
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            improved = False
            # Try adding an element not in subset
            for x in arr:
                if x not in s:
                    # Check if we can add x
                    ok = True
                    for a in s:
                        if a + x in s:
                            ok = False
                            break
                    if ok:
                        best.append(x)
                        s.add(x)
                        best_size += 1
                        improved = True
                        break
            
            if not improved:
                # Try swapping: remove one, add another
                for i, a in enumerate(best):
                    for x in arr:
                        if x in s or x == a:
                            continue
                        # Try removing a and adding x
                        new_s = s - {a}
                        ok = True
                        for b in new_s:
                            if b + x in new_s:
                                ok = False
                                break
                        if ok:
                            # Also check x+x not in new_s
                            if 2 * x in new_s:
                                ok = False
                        if ok:
                            best[i] = x
                            s = new_s | {x}
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best

    def solve_one(arr):
        """Solve one instance: try multiple strategies, return best sum-free subset."""
        n = len(arr)
        
        # Strategy 1: Odd numbers
        odds = [x for x in arr if x % 2 != 0]
        
        # Strategy 2: Greedy ascending
        greedy_asc = greedy_construct(arr, 'asc')
        
        # Strategy 3: Greedy descending
        greedy_desc = greedy_construct(list(reversed(arr)), 'desc')
        
        # Strategy 4: Greedy by absolute value
        by_abs = sorted(arr, key=lambda x: abs(x))
        greedy_abs = greedy_construct(by_abs, 'abs')
        
        # Strategy 5: Greedy by value (positive first)
        by_val = sorted(arr, key=lambda x: (x < 0, x))
        greedy_val = greedy_construct(by_val, 'val')
        
        # Collect candidates
        candidates = [odds, greedy_asc, greedy_desc, greedy_abs, greedy_val]
        
        # Find best initial
        best = max(candidates, key=len)
        
        # Local search from best
        if time.time() < deadline - 20:
            best = local_search(best, arr, max_iters=10000)
        
        # Also try local search from other candidates if time permits
        for cand in candidates:
            if time.time() > deadline - 15:
                break
            if len(cand) < len(best):
                improved = local_search(cand, arr, max_iters=3000)
                if len(improved) > len(best):
                    best = improved
        
        # Final verification
        if not is_sum_free(best):
            # Fallback to odds which should be sum-free
            best = odds
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        arr = instance(size, seed)
        if time.time() > deadline - 5:
            # Emergency: just use odds
            results.append([x for x in arr if x % 2 != 0])
            continue
        result = solve_one(arr)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
