"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def is_valid_sidon_with_differences(elements, required_diffs):
        """Check if elements form a Sidon set and cover all required differences."""
        if len(elements) > 32:
            return False
        if max(elements) > 10**6:
            return False
        if 0 not in elements:
            return False
        s = sorted(elements)
        diffs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                d = s[j] - s[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in required_diffs:
            if d not in diffs:
                return False
        return True
    
    # Known good solution: perfect difference set of order 6 (16 elements)
    # This gives differences 1..24. We need 1..30.
    # Let's try to find a better solution using backtracking.
    
    best_solution = None
    best_max_val = float('inf')
    
    # Start with a greedy solution as baseline
    a = [0]
    def get_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for d in range(1, 31):
        if d in get_diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(get_diffs(trial))) == len(get_diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(get_diffs(trial))) == len(get_diffs(trial)):
                    a = trial
                    break
                y += 1
    
    if is_valid_sidon_with_differences(a, range(1, 31)):
        best_solution = a[:]
        best_max_val = max(a)
    
    # Now try to improve using local search / backtracking
    # Try adding elements strategically to cover missing diffs while keeping Sidon property
    
    import random
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline - 5 and attempts < 50:
        attempts += 1
        
        # Random restart with different initial greedy order
        a2 = [0]
        order = list(range(1, 31))
        random.shuffle(order)
        
        for d in order:
            if d in get_diffs(sorted(a2)):
                continue
            found = False
            candidates = sorted(set(v + s for v in a2 for s in (d, -d) if v + s >= 0) - set(a2))
            random.shuffle(candidates)
            for x in candidates:
                trial = sorted(a2 + [x])
                if len(set(get_diffs(trial))) == len(get_diffs(trial)):
                    a2 = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a2) + 3 * d + 1
                while True:
                    trial = sorted(a2 + [y, y + d])
                    if len(set(get_diffs(trial))) == len(get_diffs(trial)):
                        a2 = trial
                        break
                    y += 1
        
        if is_valid_sidon_with_differences(a2, range(1, 31)):
            if max(a2) < best_max_val:
                best_solution = a2[:]
                best_max_val = max(a2)
    
    # Try more targeted optimization: try to reduce the max element
    if best_solution is not None:
        # Try removing elements that aren't needed
        current = best_solution[:]
        current_set = set(current)
        current_diffs = set()
        for i in range(len(sorted(current))):
            for j in range(i + 1, len(sorted(current))):
                current_diffs.add(sorted(current)[j] - sorted(current)[i])
        
        # Try to see if we can replace large elements with smaller ones
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            if len(current) <= 1:
                break
            # Try removing the largest element and seeing if we can still cover 1..30
            s = sorted(current)
            largest = s[-1]
            without = [x for x in current if x != largest]
            if len(without) < 1:
                break
            w_diffs = set()
            ws = sorted(without)
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)):
                    w_diffs.add(ws[j] - ws[i])
            missing = [d for d in range(1, 31) if d not in w_diffs]
            if not missing:
                # We can remove the largest element!
                current = without
                best_solution = current[:]
                best_max_val = max(current)
            else:
                # Try to find a replacement
                for new_val in range(1, largest):
                    if new_val in current_set:
                        continue
                    trial = sorted(without + [new_val])
                    t_diffs = set()
                    valid = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            d = trial[j] - trial[i]
                            if d in t_diffs:
                                valid = False
                                break
                            t_diffs.add(d)
                        if not valid:
                            break
                    if valid and all(d in t_diffs for d in range(1, 31)) and len(trial) <= 32:
                        current = trial
                        current_set = set(current)
                        best_solution = current[:]
                        best_max_val = max(current)
                        break
    
    return best_solution
# EVOLVE-BLOCK-END
