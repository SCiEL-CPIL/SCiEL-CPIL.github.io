"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    deadline = time.time() + 130  # safety margin under 150s wall-clock
    best = None
    best_val = float('inf')
    
    # Precompute all pairs (i,j) with i<j for n elements
    # We'll work with sorted arrays of up to 32 elements
    
    def diff_set(arr):
        """Return set of positive differences."""
        n = len(arr)
        s = set()
        for i in range(n):
            for j in range(i+1, n):
                s.add(arr[j] - arr[i])
        return s
    
    def is_sidon(arr):
        """Check if arr is a Sidon set (all positive differences distinct)."""
        n = len(arr)
        diffs = []
        for i in range(n):
            for j in range(i+1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.append(d)
        return True
    
    def covers_1_to_30(arr):
        """Check if all d in 1..30 appear as differences."""
        n = len(arr)
        present = set()
        for i in range(n):
            for j in range(i+1, n):
                d = arr[j] - arr[i]
                if 1 <= d <= 30:
                    present.add(d)
        return len(present) == 30
    
    def valid_solution(arr):
        """Check all constraints: Sidon, covers 1..30, size <= 32, includes 0, in [0, 10^6]."""
        if len(arr) > 32:
            return False
        if 0 not in arr:
            return False
        if any(x < 0 or x > 1000000 for x in arr):
            return False
        if not is_sidon(arr):
            return False
        if not covers_1_to_30(arr):
            return False
        return True
    
    def objective(arr):
        """Largest element (diameter)."""
        return max(arr)
    
    # Strategy 1: Try small known constructions / parametric families
    # A Sidon set that covers 1..30 needs at least ~sqrt(30) ~ 6 elements minimum,
    # but realistically we need enough elements to generate 30 distinct differences.
    # With n elements, max differences = n*(n-1)/2. Need >= 30, so n >= 8 (8*7/2=28 <30, so n>=9: 9*8/2=36>=30)
    
    # Try randomized local search
    rng = random.Random(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Start with a random Sidon set containing 0
        # Use a greedy random construction
        arr = [0]
        used_diffs = set()
        
        for _ in range(31):  # try to build up to 32 elements
            candidates = []
            # Try adding elements in range [1, 500]
            for x in range(1, 500):
                if x in arr:
                    continue
                # Check if adding x preserves Sidon property
                ok = True
                for a in arr:
                    d = abs(x - a)
                    if d == 0:
                        continue
                    if d in used_diffs:
                        ok = False
                        break
                if ok:
                    # Also check new differences among existing + x don't collide
                    new_diffs = set()
                    for a in arr:
                        new_diffs.add(abs(x - a))
                    if len(new_diffs) != len(arr):
                        ok = False
                    if ok:
                        candidates.append(x)
            
            if not candidates:
                break
            
            # Prefer smaller x to minimize diameter
            rng.shuffle(candidates)
            candidates.sort()
            x = candidates[0]
            
            # Update used_diffs
            for a in arr:
                used_diffs.add(abs(x - a))
            arr.append(x)
            arr.sort()
        
        if len(arr) < 9:
            continue
        
        # Now try local search to improve: minimize max element while maintaining validity
        # First check if current is valid
        if not valid_solution(arr):
            # Try to repair: add missing differences
            present = set()
            for i in range(len(arr)):
                for j in range(i+1, len(arr)):
                    d = arr[j] - arr[i]
                    if 1 <= d <= 30:
                        present.add(d)
            missing = set(range(1, 31)) - present
            
            # Try to add elements to cover missing differences
            improved = False
            for d in sorted(missing):
                if time.time() > deadline:
                    break
                found = False
                for x in range(1, 2000):
                    if x in arr:
                        continue
                    trial = sorted(arr + [x])
                    if len(trial) > 32:
                        continue
                    if is_sidon(trial):
                        # Check if this helps cover d
                        new_present = set()
                        for i in range(len(trial)):
                            for j in range(i+1, len(trial)):
                                dd = trial[j] - trial[i]
                                if 1 <= dd <= 30:
                                    new_present.add(dd)
                        if d in new_present:
                            arr = trial
                            found = True
                            break
                if not found:
                    break
            
            if not valid_solution(arr):
                continue
        
        # Local search: try to reduce max element
        improved = True
        while improved and time.time() < deadline:
            improved = False
            max_elem = max(arr)
            # Try replacing the largest element with a smaller one
            idx = arr.index(max_elem)
            for new_val in range(1, max_elem):
                if new_val in arr:
                    continue
                trial = arr[:idx] + [new_val] + arr[idx+1:]
                trial.sort()
                if valid_solution(trial):
                    arr = trial
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try removing an element if size > minimum needed
            if len(arr) > 9:
                for i in range(len(arr)):
                    if arr[i] == 0:
                        continue  # must keep 0
                    trial = arr[:i] + arr[i+1:]
                    if valid_solution(trial):
                        arr = trial
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping two elements
            for i in range(len(arr)):
                for j in range(i+1, len(arr)):
                    if arr[i] == 0 or arr[j] == 0:
                        continue
                    # Try reducing the larger one
                    if arr[j] > arr[i]:
                        for new_val in range(arr[i]+1, arr[j]):
                            if new_val in arr:
                                continue
                            trial = arr[:]
                            trial[j] = new_val
                            trial.sort()
                            if valid_solution(trial):
                                arr = trial
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        
        if valid_solution(arr):
            val = objective(arr)
            if val < best_val:
                best_val = val
                best = arr[:]
    
    if best is None:
        # Fallback: use the original greedy approach
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
    
    return best
# EVOLVE-BLOCK-END
