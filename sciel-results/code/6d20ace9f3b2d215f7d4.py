import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    n = 256

    def can_add(x, S_sorted):
        """Check if adding x to S preserves validity. S_sorted is a sorted list."""
        if not S_sorted:
            return True
        
        # Case 1: x is the smallest: x < b <= c, check x does not divide b+c
        # Only need to check pairs (b,c) where b,c > x
        idx = bisect.bisect_right(S_sorted, x)
        for i in range(idx, len(S_sorted)):
            b = S_sorted[i]
            for j in range(i, len(S_sorted)):
                c = S_sorted[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle: a < x <= c, check a does not divide x+c
        # a < x, c >= x
        for i in range(idx):
            a = S_sorted[i]
            for j in range(idx, len(S_sorted)):
                c = S_sorted[j]
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest: a < b <= x, check a does not divide b+x
        # a < b <= x
        for i in range(idx):
            a = S_sorted[i]
            for j in range(i + 1, idx):
                b = S_sorted[j]
                if (b + x) % a == 0:
                    return False
        
        return True

    def greedy_construct(order, time_limit):
        """Greedy construction with given element ordering."""
        S = set()
        S_sorted = []
        for x in order:
            if time.time() > time_limit:
                break
            if can_add(x, S_sorted):
                S.add(x)
                bisect.insort(S_sorted, x)
        return S, S_sorted

    candidates = []

    # Strategy 1: Descending (larger numbers first)
    limit1 = start_time + 5.0
    S1, _ = greedy_construct(range(256, 0, -1), limit1)
    candidates.append(S1)

    # Strategy 2: Try several random orderings with different seeds
    rng = random.Random(42)
    for trial in range(8):
        limit = start_time + 5.0 + trial * 2.0
        if time.time() > limit:
            break
        order = list(range(1, 257))
        rng.shuffle(order)
        Sr, _ = greedy_construct(order, limit)
        candidates.append(Sr)

    # Strategy 3: "Gap-aware" ordering - prioritize numbers that are far from multiples
    # Heuristic: for each x, count how many pairs (b,c) in [x+1, 256] have x | (b+c)
    # Numbers with fewer such pairs are "safer" to include early
    limit3 = start_time + 15.0
    if time.time() < limit3:
        safety_scores = {}
        for x in range(1, 257):
            count = 0
            for b in range(x + 1, 257):
                for c in range(b, 257):
                    if (b + c) % x == 0:
                        count += 1
            safety_scores[x] = count
        # Sort by safety (fewer conflicts first), then by value (larger first)
        order3 = sorted(range(1, 257), key=lambda x: (safety_scores[x], -x))
        S3, _ = greedy_construct(order3, limit3)
        candidates.append(S3)

    # Pick the best candidate
    best_S = max(candidates, key=len)
    best_size = len(best_S)
    current = set(best_S)
    current_sorted = sorted(current)

    # Local search phase 1: try to add any missing element (multiple rounds)
    limit_ls1 = start_time + 40.0
    if time.time() < limit_ls1:
        improved = True
        while improved and time.time() < limit_ls1:
            improved = False
            for x in range(1, 257):
                if time.time() > limit_ls1:
                    break
                if x in current:
                    continue
                if can_add(x, current_sorted):
                    current.add(x)
                    bisect.insort(current_sorted, x)
                    improved = True
                    if len(current) > best_size:
                        best_S = set(current)
                        best_size = len(current)

    # Local search phase 2: 1-remove + 1-add swaps
    limit_ls2 = start_time + 60.0
    if time.time() < limit_ls2:
        for _ in range(500):
            if time.time() > limit_ls2:
                break
            found_improvement = False
            for rem in sorted(current):
                if time.time() > limit_ls2:
                    break
                trial = set(current)
                trial.discard(rem)
                trial_sorted = [x for x in current_sorted if x != rem]
                
                # Try to add the best missing element
                best_add = None
                for add in range(256, 0, -1):
                    if add in trial:
                        continue
                    if can_add(add, trial_sorted):
                        best_add = add
                        break
                
                if best_add is not None:
                    trial.add(best_add)
                    bisect.insort(trial_sorted, best_add)
                    if len(trial) > len(current):
                        current = trial
                        current_sorted = trial_sorted
                        if len(current) > best_size:
                            best_S = set(current)
                            best_size = len(current)
                        found_improvement = True
                        break
            
            if not found_improvement:
                # Try a random perturbation: remove a random element and try to fill
                if len(current) > 10:
                    rem = random.choice(list(current))
                    trial = set(current)
                    trial.discard(rem)
                    trial_sorted = [x for x in current_sorted if x != rem]
                    added = 0
                    for add in range(256, 0, -1):
                        if add in trial:
                            continue
                        if can_add(add, trial_sorted):
                            trial.add(add)
                            bisect.insort(trial_sorted, add)
                            added += 1
                            if added >= 1:
                                break
                    if len(trial) > len(current):
                        current = trial
                        current_sorted = trial_sorted
                        if len(current) > best_size:
                            best_S = set(current)
                            best_size = len(current)

    # Local search phase 3: 2-remove + k-add
    limit_ls3 = start_time + 75.0
    if time.time() < limit_ls3:
        for _ in range(200):
            if time.time() > limit_ls3:
                break
            found_improvement = False
            # Try removing pairs of small elements
            smalls = sorted([x for x in current if x < 64])
            if len(smalls) >= 2:
                for i in range(len(smalls) - 1):
                    if time.time() > limit_ls3:
                        break
                    rem1, rem2 = smalls[i], smalls[i + 1]
                    if rem1 == rem2:
                        continue
                    trial = set(current)
                    trial.discard(rem1)
                    trial.discard(rem2)
                    trial_sorted = [x for x in current_sorted if x != rem1 and x != rem2]
                    
                    added = 0
                    for add in range(256, 0, -1):
                        if add in trial:
                            continue
                        if can_add(add, trial_sorted):
                            trial.add(add)
                            bisect.insort(trial_sorted, add)
                            added += 1
                            if added >= 2:
                                break
                    
                    if len(trial) > len(current):
                        current = trial
                        current_sorted = trial_sorted
                        if len(current) > best_size:
                            best_S = set(current)
                            best_size = len(current)
                        found_improvement = True
                        break
            
            if not found_improvement:
                # Random 2-removal
                if len(current) > 10:
                    sample = random.sample(list(current), 2)
                    trial = set(current)
                    for r in sample:
                        trial.discard(r)
                    trial_sorted = [x for x in current_sorted if x not in sample]
                    added = 0
                    for add in range(256, 0, -1):
                        if add in trial:
                            continue
                        if can_add(add, trial_sorted):
                            trial.add(add)
                            bisect.insort(trial_sorted, add)
                            added += 1
                            if added >= 2:
                                break
                    if len(trial) > len(current):
                        current = trial
                        current_sorted = trial_sorted
                        if len(current) > best_size:
                            best_S = set(current)
                            best_size = len(current)

    # Final polish: one more round of additions
    limit_final = start_time + 83.0
    if time.time() < limit_final:
        current = set(best_S)
        current_sorted = sorted(current)
        for x in range(1, 257):
            if time.time() > limit_final:
                break
            if x not in current and can_add(x, current_sorted):
                current.add(x)
                bisect.insort(current_sorted, x)
        if len(current) > len(best_S):
            best_S = current

    if not best_S:
        return [256]
    
    return list(best_S)
