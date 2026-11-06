import random
import time
import sys
from itertools import combinations

def construct():
    """
    Finds a Sidon set containing 0, covering differences 1..30,
    with at most 32 elements, minimizing the largest element.
    
    Strategy:
    1. Build an initial valid Sidon set via greedy + repair.
    2. Use a difference-assignment based local search: 
       - Represent state as sorted list of elements.
       - Moves: (a) move one element to a smaller value, (b) swap two elements,
         (c) insert/delete element (if size allows).
       - Objective: minimize max element subject to validity.
    3. Use iterated local search with restarts to escape local minima.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    def get_diffs(s):
        """Return list of all positive differences in sorted set s."""
        return [s[j] - s[i] for i in range(len(s)) for j in range(i+1, len(s))]
    
    def is_sidon(s):
        """Check if all positive differences are distinct."""
        if len(s) < 2:
            return True
        diffs = get_diffs(s)
        return len(set(diffs)) == len(diffs)
    
    def covers_1_to_30(s):
        """Check if all d in 1..30 appear as a difference."""
        diffs = set(get_diffs(s))
        return all(d in diffs for d in range(1, 31))
    
    def is_valid(s):
        """Full validity check."""
        if len(s) > 32 or 0 not in s:
            return False
        if len(set(s)) != len(s):
            return False
        if any(x < 0 for x in s):
            return False
        if not is_sidon(s):
            return False
        if not covers_1_to_30(s):
            return False
        return True
    
    # Phase 1: Greedy construction to get a valid starting set
    a = [0]
    
    # We need to cover differences 1..30. 
    # Greedy: for each missing difference, try to add a new element that creates it
    # and doesn't break Sidon property.
    
    def greedy_build():
        s = [0]
        needed = set(range(1, 31))
        
        for _ in range(40):  # Try up to 40 rounds
            if not needed:
                break
            # Find which differences are still missing
            current_diffs = set(get_diffs(s))
            needed = needed - current_diffs
            
            if not needed:
                break
            
            # For each missing difference d, try to find a new element x
            # such that x - y = d or y - x = d for some y in s,
            # and adding x preserves Sidon property.
            
            best_candidate = None
            best_diam = float('inf')
            
            for d in sorted(needed):
                # Try x = y + d for each y in s
                for y in s:
                    x = y + d
                    if x in s or x < 0:
                        continue
                    trial = sorted(s + [x])
                    if len(trial) > 32:
                        continue
                    if is_sidon(trial):
                        diam = max(trial)
                        if diam < best_diam:
                            best_diam = diam
                            best_candidate = trial
                    # Also try x = y - d
                    x = y - d
                    if x in s or x < 0:
                        continue
                    trial = sorted(s + [x])
                    if len(trial) > 32:
                        continue
                    if is_sidon(trial):
                        diam = max(trial)
                        if diam < best_diam:
                            best_diam = diam
                            best_candidate = trial
            
            if best_candidate is not None:
                s = best_candidate
            else:
                # Fallback: add a large element to create a missing difference
                d = min(needed)
                x = max(s) + d + 1
                while True:
                    trial = sorted(s + [x])
                    if len(trial) <= 32 and is_sidon(trial):
                        s = trial
                        break
                    x += 1
        
        return s
    
    best_set = greedy_build()
    best_val = max(best_set) if best_set else float('inf')
    
    # Verify it's valid; if not, do a more aggressive repair
    if not is_valid(best_set):
        # Force validity by adding elements
        s = [0]
        for d in range(1, 31):
            current_diffs = set(get_diffs(s))
            if d not in current_diffs:
                # Find x to add
                found = False
                for y in s:
                    for sign in [1, -1]:
                        x = y + sign * d
                        if x < 0 or x in s:
                            continue
                        trial = sorted(s + [x])
                        if len(trial) <= 32 and is_sidon(trial):
                            s = trial
                            found = True
                            break
                    if found:
                        break
                if not found:
                    x = max(s) + d + 1
                    while True:
                        trial = sorted(s + [x])
                        if len(trial) <= 32 and is_sidon(trial):
                            s = trial
                            break
                        x += 1
        best_set = s
        best_val = max(best_set)
    
    # Phase 2: Local search to minimize diameter
    # Move types:
    # 1. Move element i (not 0) to a smaller value
    # 2. Move element i to a random value in [1, current_max]
    # 3. If len < 32, insert a new element at a small value
    # 4. If len > minimum_needed, remove an element
    
    def local_search(initial, timeout):
        """Run local search starting from initial set, return best found."""
        current = sorted(initial[:])
        current_max = max(current)
        best_local = current[:]
        best_local_max = current_max
        
        t0 = time.time()
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() - t0 < timeout:
            if no_improve_count > max_no_improve:
                break
            
            improved = False
            n = len(current)
            
            # Try moving each non-zero element to a smaller value
            for i in range(n):
                if i == 0:
                    continue  # Keep 0 fixed
                old_val = current[i]
                
                # Try values from 1 up to old_val - 1
                # But only try a subset to keep it fast
                if old_val > 20:
                    # Try a random sample of smaller values
                    candidates = list(range(1, old_val))
                    random.shuffle(candidates)
                    candidates = candidates[:50]
                else:
                    candidates = list(range(1, old_val))
                
                for new_val in candidates:
                    if new_val in current:
                        continue
                    trial = current[:]
                    trial[i] = new_val
                    trial.sort()
                    if len(trial) > 32:
                        continue
                    if is_sidon(trial) and covers_1_to_30(trial):
                        new_max = max(trial)
                        if new_max < current_max:
                            current = trial
                            current_max = new_max
                            if new_max < best_local_max:
                                best_local = current[:]
                                best_local_max = new_max
                            improved = True
                            no_improve_count = 0
                            break
                if improved:
                    break
                if time.time() - t0 > timeout:
                    break
            
            if not improved:
                no_improve_count += 1
            
            # Occasionally try a larger perturbation to escape local minima
            if no_improve_count % 1000 == 0 and n > 1:
                i = random.randint(1, n - 1)
                new_val = random.randint(1, current_max + 20)
                if new_val not in current:
                    trial = current[:]
                    trial[i] = new_val
                    trial.sort()
                    if is_sidon(trial) and covers_1_to_30(trial):
                        current = trial
                        current_max = max(current)
                    elif is_sidon(trial):
                        # Accept even if coverage is broken, but track it
                        current = trial
                        current_max = max(current)
        
        return best_local
    
    # Phase 3: Iterated local search with restarts
    # We run multiple rounds of local search, each starting from the best found
    # or from a perturbed version of it.
    
    while time.time() - start_time < time_limit:
        # Run local search with a time budget
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
        
        ls_timeout = min(10.0, remaining - 2)
        if ls_timeout < 1:
            break
        
        # Start from best_set, possibly perturbed
        if random.random() < 0.3 and len(best_set) > 1:
            # Perturb: move one random element
            perturbed = best_set[:]
            i = random.randint(1, len(perturbed) - 1)
            perturbed[i] = random.randint(1, max(2, best_val + 10))
            perturbed.sort()
            if len(set(perturbed)) == len(perturbed) and is_sidon(perturbed):
                start_ls = perturbed
            else:
                start_ls = best_set[:]
        else:
            start_ls = best_set[:]
        
        result = local_search(start_ls, ls_timeout)
        
        if max(result) < best_val:
            best_set = result
            best_val = max(best_set)
    
    return best_set

if __name__ == "__main__":
    res = construct()
    print(res)
    print("Max:", max(res))
