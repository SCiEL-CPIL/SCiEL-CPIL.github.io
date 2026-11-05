import time
import random

DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    best_solution = None
    best_diameter = float('inf')

    # Greedy fallback to always have a valid answer
    def greedy_construct():
        a = [0]
        used_diffs = set()
        for d in range(1, 31):
            if d in used_diffs:
                continue
            # Try to add a single element that creates difference d
            found = False
            for x in sorted(set(v + d for v in a) | set(v - d for v in a if v - d >= 0)):
                if x in a:
                    continue
                # Check all new differences
                new_diffs_ok = True
                new_diffs = []
                for y in a:
                    diff = abs(x - y)
                    if diff in used_diffs:
                        new_diffs_ok = False
                        break
                    new_diffs.append(diff)
                if new_diffs_ok and d in new_diffs:
                    a.append(x)
                    used_diffs.update(new_diffs)
                    found = True
                    break
            if not found:
                # Add two elements
                y = 2 * max(a) + 3 * d + 1
                while True:
                    new_diffs = []
                    ok = True
                    for y_val in (y, y + d):
                        for a_val in a:
                            diff = abs(y_val - a_val)
                            if diff in used_diffs:
                                ok = False
                                break
                            new_diffs.append(diff)
                        if not ok:
                            break
                    diff_between = abs((y + d) - y)
                    if diff_between in used_diffs:
                        ok = False
                    else:
                        new_diffs.append(diff_between)
                    if ok:
                        a.extend([y, y + d])
                        used_diffs.update(new_diffs)
                        break
                    y += 1
        return sorted(a)

    # Backtracking with forward checking
    def backtrack(current_set, used_diffs, remaining_diffs, depth):
        nonlocal best_solution, best_diameter, start_time

        if time.time() - start_time > DEADLINE:
            return False

        if not remaining_diffs:
            diam = max(current_set) - min(current_set)
            if diam < best_diameter:
                best_diameter = diam
                best_solution = list(current_set)
            return True

        # Try to find the smallest missing difference
        target_d = min(remaining_diffs)

        # Try adding a single element that creates target_d
        candidates = set()
        for v in current_set:
            candidates.add(v + target_d)
            if v - target_d >= 0:
                candidates.add(v - target_d)

        for x in sorted(candidates):
            if x in current_set:
                continue

            # Check if adding x creates target_d and no duplicate diffs
            new_diffs = []
            valid = True
            for y in current_set:
                diff = abs(x - y)
                if diff in used_diffs:
                    valid = False
                    break
                new_diffs.append(diff)

            if not valid or target_d not in new_diffs:
                continue

            new_current = current_set | {x}
            new_used = used_diffs | set(new_diffs)
            new_remaining = remaining_diffs - {target_d}

            if backtrack(new_current, new_used, new_remaining, depth + 1):
                return True

        # Try adding two elements that together create target_d
        # x and x+target_d, where x is new
        # This is more expensive, so only try a limited range
        max_val = max(current_set) if current_set else 0
        for x in range(0, max_val + target_d + 10):
            y = x + target_d
            if x in current_set or y in current_set:
                continue

            new_diffs = []
            valid = True
            for z in (x, y):
                for w in current_set:
                    diff = abs(z - w)
                    if diff in used_diffs:
                        valid = False
                        break
                    new_diffs.append(diff)
                if not valid:
                    break
            diff_xy = abs(y - x)
            if diff_xy in used_diffs:
                valid = False
            else:
                new_diffs.append(diff_xy)

            if not valid or target_d not in new_diffs:
                continue

            new_current = current_set | {x, y}
            new_used = used_diffs | set(new_diffs)
            new_remaining = remaining_diffs - {target_d}

            if backtrack(new_current, new_used, new_remaining, depth + 1):
                return True

        return False

    # Start backtracking
    initial_set = {0}
    initial_used = set()
    initial_remaining = set(range(1, 31))

    success = backtrack(initial_set, initial_used, initial_remaining, 0)

    if best_solution is not None:
        return best_solution
    else:
        # Fallback to greedy
        return greedy_construct()
