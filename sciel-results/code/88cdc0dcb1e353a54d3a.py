import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Base: upper tail [171, 256] is valid
    s = set(range(171, 257))

    def can_add(x, current_set):
        """Check if x can be added to current_set without violating the property."""
        elems = sorted(current_set)

        # Case 1: x is the smallest (a=x, b,c in current_set with x<b<=c)
        for b in elems:
            if b <= x:
                continue
            for c in elems:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False

        # Case 2: x is the middle (b=x, a in current_set with a<x, c in current_set with x<=c)
        for a in elems:
            if a >= x:
                continue
            for c in elems:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False

        # Case 3: x is the largest (c=x, a,b in current_set with a<b<=x)
        for a in elems:
            for b in elems:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    return False

        return True

    def can_add_fast(x, current_set, sorted_elems):
        """Faster version using pre-sorted list."""
        elems = sorted_elems

        # Case 1: x is the smallest (a=x, b,c in current_set with x<b<=c)
        for i, b in enumerate(elems):
            if b <= x:
                continue
            for j in range(i, len(elems)):
                c = elems[j]
                if (b + c) % x == 0:
                    return False

        # Case 2: x is the middle (b=x, a in current_set with a<x, c in current_set with x<=c)
        for a in elems:
            if a >= x:
                continue
            for c in elems:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False

        # Case 3: x is the largest (c=x, a,b in current_set with a<b<=x)
        for a in elems:
            for b in elems:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    return False

        return True

    # Phase 1: Greedy addition from 170 down to 1
    for x in range(170, 0, -1):
        if time.time() > deadline - 15:
            break
        if x not in s and can_add(x, s):
            s.add(x)

    best_set = set(s)
    best_size = len(s)

    # Phase 2: Try different greedy orderings
    orderings = [
        list(range(170, 0, -1)),
        list(range(1, 171)),
    ]
    # Add a few random orderings
    rng = random.Random(42)
    for _ in range(3):
        orderings.append(rng.sample(range(1, 171), 170))

    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        s_trial = set(range(171, 257))
        for x in ordering:
            if x not in s_trial and can_add(x, s_trial):
                s_trial.add(x)
        if len(s_trial) > best_size:
            best_size = len(s_trial)
            best_set = set(s_trial)

    # Phase 3: Local search - try swaps to improve
    # Try removing one element and adding another to increase size
    for iteration in range(50):
        if time.time() > deadline - 5:
            break

        s_trial = set(best_set)
        improved = False

        # Try to remove a small element and add a different one
        small_elements = sorted([x for x in s_trial if x < 171])
        if not small_elements:
            break

        for rem in small_elements:
            if time.time() > deadline - 5:
                break
            s_trial.discard(rem)
            # Try to add elements not in s_trial
            candidates = [x for x in range(1, 171) if x not in s_trial]
            for add in candidates:
                if can_add(add, s_trial):
                    s_trial.add(add)
                    if len(s_trial) > best_size:
                        best_size = len(s_trial)
                        best_set = set(s_trial)
                        improved = True
                        break
                    s_trial.discard(add)
            # Restore
            s_trial.add(rem)
            if improved:
                break

        if not improved:
            break

    # Phase 4: More aggressive local search - try multiple removals
    for iteration in range(20):
        if time.time() > deadline - 3:
            break

        s_trial = set(best_set)
        small_elements = sorted([x for x in s_trial if x < 171])
        if len(s_trial) < 10:
            break

        # Try removing 1-2 small elements
        removals = []
        for _ in range(random.randint(1, 2)):
            if small_elements:
                removals.append(random.choice(small_elements))
                small_elements.remove(removals[-1])

        for rem in removals:
            s_trial.discard(rem)

        # Try to add as many as possible
        candidates = sorted([x for x in range(1, 171) if x not in s_trial], reverse=True)
        for add in candidates:
            if time.time() > deadline - 3:
                break
            if can_add(add, s_trial):
                s_trial.add(add)

        if len(s_trial) > best_size:
            best_size = len(s_trial)
            best_set = set(s_trial)

    # Phase 5: Final polish - try to add any remaining elements
    s_final = set(best_set)
    for x in range(1, 171):
        if time.time() > deadline - 1:
            break
        if x not in s_final and can_add(x, s_final):
            s_final.add(x)

    if len(s_final) > best_size:
        best_set = s_final

    return sorted(best_set)
