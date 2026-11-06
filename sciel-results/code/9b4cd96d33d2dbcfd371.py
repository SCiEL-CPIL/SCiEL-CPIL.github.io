"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    def check_sidon_incremental(elements, new_val):
        """Check if adding new_val to elements keeps all differences distinct."""
        # elements is a sorted list, new_val > elements[-1]
        for i, e in enumerate(elements):
            d = new_val - e
            # Check if d already exists as a difference between any two existing elements
            for j in range(i + 1, len(elements)):
                if elements[j] - elements[i] == d:
                    return False
        # Also check differences among existing elements don't conflict with new ones
        # (already covered above since we check all pairs)
        return True

    def get_missing_diffs(elements):
        """Return set of differences in 1..30 not yet realized."""
        s = set(elements)
        present = set()
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                d = s[j] - s[i]
                if 1 <= d <= 30:
                    present.add(d)
        return set(range(1, 31)) - present

    def is_valid_final(elements):
        """Check if elements form a valid Sidon set covering 1..30."""
        s = sorted(elements)
        n = len(s)
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                d = s[j] - s[i]
                diffs.append(d)
        if len(set(diffs)) != len(diffs):
            return False
        diff_set = set(diffs)
        return all(d in diff_set for d in range(1, 31))

    def backtrack_search(max_size=32):
        """
        Backtracking search: build Sidon set element by element.
        We must include 0. Try to add elements in increasing order.
        Prune: if remaining elements can't cover all missing diffs, stop.
        """
        best = [None]  # mutable holder for best solution
        
        def dfs(elements, missing, depth):
            if time.time() > deadline:
                return
            if len(elements) > max_size:
                return
            if not missing:
                # Found a valid solution
                if best[0] is None or max(elements) < max(best[0]):
                    best[0] = list(elements)
                return
            
            # Pruning: with k more elements, we can realize at most k*(len(elements)+k)/2 new differences
            # But more precisely, each new element x adds differences x - e for all e in elements
            # plus differences between new elements. Max new diffs from k more elements:
            # k * len(elements) + k*(k-1)/2
            k_remaining = max_size - len(elements)
            max_new_diffs = k_remaining * len(elements) + k_remaining * (k_remaining - 1) // 2
            if max_new_diffs < len(missing):
                return
            
            # Try to add a new element
            # The new element must be greater than the last one (to maintain sorted order)
            last = elements[-1] if elements else 0
            # Lower bound: at least last + 1
            # Upper bound: we want to minimize max, so try small values first
            # But we also need to realize missing diffs
            
            # Strategy: try candidate values that would realize at least one missing diff
            # For each missing diff d, we can try x = e + d for some e in elements
            candidates = set()
            for d in missing:
                for e in elements:
                    x = e + d
                    if x > last:
                        candidates.add(x)
            
            # Also try just last + 1, last + 2, etc. for small values
            for x in range(last + 1, last + 50):
                candidates.add(x)
            
            # Sort candidates: prefer those that realize more missing diffs
            def score_candidate(x):
                count = 0
                for d in missing:
                    for e in elements:
                        if x - e == d:
                            count += 1
                            break
                return -count  # negative so more-realizing comes first
            
            candidates = sorted(candidates, key=score_candidate)
            
            for x in candidates:
                if x > 10**6:
                    break
                # Check Sidon constraint
                if not check_sidon_incremental(elements, x):
                    continue
                new_elements = elements + [x]
                new_missing = get_missing_diffs(new_elements)
                dfs(new_elements, new_missing, depth + 1)
                if best[0] is not None and max(new_elements) >= max(best[0]):
                    # Prune: if current max already >= best, no point continuing with larger x
                    # (but x is increasing, so all further x will be larger)
                    # Actually x is not strictly increasing in our loop, so we can't prune here
                    pass
        
        dfs([0], set(range(1, 31)), 0)
        return best[0]

    def local_search_improve(a):
        """Try to improve a valid Sidon set by reducing the max element."""
        a = sorted(a)
        best_a = list(a)
        best_max = max(a)
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Try to reduce the largest element
            largest = best_a[-1]
            # Find which differences depend on the largest element
            for new_val in range(largest - 1, max(0, largest - 200), -1):
                if new_val in best_a[:-1]:
                    continue
                trial = sorted(best_a[:-1] + [new_val])
                if is_valid_final(trial):
                    best_a = trial
                    best_max = new_val
                    improved = True
                    break
            
            # Try to remove an element if not needed
            for i in range(len(best_a) - 1, 0, -1):  # Don't remove 0
                trial = best_a[:i] + best_a[i+1:]
                if is_valid_final(trial):
                    best_a = trial
                    best_max = max(best_a)
                    improved = True
                    break
            
            if not improved:
                break
        
        return best_a

    # Step 1: Backtracking search
    result = backtrack_search()
    
    if result is None:
        # Fallback: greedy construction
        a = [0]
        for d in range(1, 31):
            if time.time() > deadline - 10:
                break
            s = set(a)
            if d in {a[j] - a[i] for i in range(len(a)) for j in range(i+1, len(a))}:
                continue
            found = False
            for x in range(1, 5000):
                trial = sorted(a + [x])
                if len(trial) > 32:
                    break
                diffs = [trial[j] - trial[i] for i in range(len(trial)) for j in range(i+1, len(trial))]
                if len(set(diffs)) == len(diffs):
                    a = trial
                    found = True
                    break
            if not found:
                # Add a large element
                y = 2 * max(a) + 3 * d + 1
                while time.time() < deadline - 10:
                    trial = sorted(a + [y])
                    diffs = [trial[j] - trial[i] for i in range(len(trial)) for j in range(i+1, len(trial))]
                    if len(set(diffs)) == len(diffs) and d in diffs:
                        a = trial
                        break
                    y += 1
        result = a
    
    # Step 2: Local search to improve
    if time.time() < deadline - 10:
        result = local_search_improve(result)
    
    return result
# EVOLVE-BLOCK-END
