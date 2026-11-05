"""Sidon set construction: minimize max element while covering differences 1..30."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # We need a Sidon set S containing 0, with all positive differences distinct,
    # and every d in 1..30 appearing as a difference. Minimize max(S).
    # At most 32 elements.

    # Strategy: Use a randomized local search / simulated annealing approach.
    # Start with a known small construction and try to reduce the max element.

    # First, get a valid solution using the greedy approach as a baseline.
    def greedy_solution():
        a = [0]
        def get_diffs(s):
            s_sorted = sorted(s)
            return [y - x for i, x in enumerate(s_sorted) for y in s_sorted[i + 1:]]
        
        for d in range(1, 31):
            current_diffs = get_diffs(a)
            if d in current_diffs:
                continue
            # Try adding x = v + d or v - d for each v in a
            candidates = []
            for v in a:
                for s in (d, -d):
                    x = v + s
                    if x >= 0 and x not in a:
                        candidates.append(x)
            
            candidates = sorted(set(candidates))
            found = False
            for x in candidates:
                trial = sorted(a + [x])
                trial_diffs = get_diffs(trial)
                if len(set(trial_diffs)) == len(trial_diffs):
                    a = trial
                    found = True
                    break
            
            if not found:
                # Add two elements
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    trial_diffs = get_diffs(trial)
                    if len(set(trial_diffs)) == len(trial_diffs):
                        a = trial
                        break
                    y += 1
        return a

    # Check if a set is valid
    def is_valid(s):
        if len(s) > 32:
            return False
        if 0 not in s:
            return False
        if len(set(s)) != len(s):
            return False
        if any(x < 0 or x > 10**6 for x in s):
            return False
        s_sorted = sorted(s)
        diffs = [y - x for i, x in enumerate(s_sorted) for y in s_sorted[i + 1:]]
        if len(set(diffs)) != len(diffs):
            return False
        diff_set = set(diffs)
        for d in range(1, 31):
            if d not in diff_set:
                return False
        return True

    # Get baseline
    baseline = greedy_solution()
    best = baseline
    best_val = max(baseline)
    
    # Try to improve using local search: remove the largest element and try to replace
    # it with smaller elements, or rearrange elements to reduce the max.
    
    def get_diffs(s):
        s_sorted = sorted(s)
        return [y - x for i, x in enumerate(s_sorted) for y in s_sorted[i + 1:]]

    def valid_sidon(s):
        """Check Sidon property only (distinct positive differences)."""
        s_sorted = sorted(s)
        diffs = [y - x for i, x in enumerate(s_sorted) for y in s_sorted[i + 1:]]
        return len(set(diffs)) == len(diffs)

    def covers_all_diffs(s):
        s_sorted = sorted(s)
        diffs = set(y - x for i, x in enumerate(s_sorted) for y in s_sorted[i + 1:])
        return all(d in diffs for d in range(1, 31))

    # Local search: try to reduce max element
    # Approach: take current set, try replacing the largest element with a smaller one,
    # or swapping pairs to reduce the maximum.
    
    current = list(baseline)
    current_val = max(current)
    
    # Simulated annealing style
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        if not is_valid(current):
            current = list(baseline)
            current_val = max(current)
        
        if max(current) < best_val:
            best = list(current)
            best_val = max(current)
        
        # Try a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove the largest element and try to add a smaller one
            if len(current) > 1:
                largest = max(current)
                new_set = [x for x in current if x != largest]
                # Try adding a random small element
                max_other = max(new_set) if new_set else 0
                # We need to cover all diffs 1..30 with new_set + [x]
                # Try candidates
                found = False
                for _ in range(50):
                    x = random.randint(0, max(max_other, 30))
                    if x in new_set:
                        continue
                    trial = new_set + [x]
                    if valid_sidon(trial) and covers_all_diffs(trial):
                        current = trial
                        found = True
                        break
                if not found:
                    # Revert
                    current.append(largest)
        
        elif move_type < 0.7:
            # Swap the largest with a smaller random element
            if len(current) > 1:
                largest = max(current)
                others = [x for x in current if x != largest]
                if others:
                    new_elem = random.randint(0, max(others))
                    if new_elem in current:
                        # Try swapping a different element
                        swap_candidates = [x for x in current if x != largest]
                        if swap_candidates:
                            old_elem = random.choice(swap_candidates)
                            new_set = [x for x in current if x != old_elem] + [new_elem]
                            if valid_sidon(new_set) and covers_all_diffs(new_set) and max(new_set) < current_val:
                                current = new_set
        
        elif move_type < 0.9:
            # Try to reduce the largest by 1 and check validity
            largest = max(current)
            if largest > 0:
                new_elem = largest - 1
                if new_elem not in current:
                    new_set = [x for x in current if x != largest] + [new_elem]
                    if valid_sidon(new_set) and covers_all_diffs(new_set):
                        current = new_set
        
        else:
            # Random restart with smaller max bound
            # Try to build a set with max <= some bound
            bound = random.randint(30, best_val)
            # Random construction
            s = [0]
            random.shuffle(s)
            # Greedy with random order
            for d in range(1, 31):
                if d in set(y - x for i, x in enumerate(sorted(s)) for y in sorted(s)[i+1:]):
                    continue
                # Try random candidates
                for _ in range(20):
                    x = random.randint(0, bound)
                    if x in s:
                        continue
                    trial = s + [x]
                    if valid_sidon(trial) and len(trial) <= 32:
                        s = trial
                        break
            if is_valid(s):
                if max(s) < best_val:
                    best = list(s)
                    best_val = max(s)
                current = list(s)
    
    # Final polish: try to further reduce
    for _ in range(5000):
        if time.time() > deadline:
            break
        if max(best) == 0:
            break
        # Try removing the largest and finding a replacement
        largest = max(best)
        rest = [x for x in best if x != largest]
        for _ in range(100):
            x = random.randint(0, largest - 1)
            if x in rest:
                continue
            trial = rest + [x]
            if is_valid(trial) and max(trial) < max(best):
                best = trial
                break
    
    return best
# EVOLVE-BLOCK-END
