"""Construct a large subset of [1,256] where no a<b<=c has a|(b+c)."""
import time
import random

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 256
    
    def is_valid(S_set):
        """Check if S_set is valid: for all a<b<=c in S, a does not divide b+c."""
        elems = sorted(S_set)
        for i, a in enumerate(elems):
            for j in range(i + 1, len(elems)):
                b = elems[j]
                # Check b+c for all c >= b in S
                for k in range(j, len(elems)):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def count_violations(S_set):
        """Count number of violating triples."""
        elems = sorted(S_set)
        count = 0
        for i, a in enumerate(elems):
            for j in range(i + 1, len(elems)):
                b = elems[j]
                for k in range(j, len(elems)):
                    c = elems[k]
                    if (b + c) % a == 0:
                        count += 1
        return count
    
    def greedy_construction(order):
        """Build set greedily in given order, adding element if it doesn't create violations."""
        S = set()
        for x in order:
            trial = S | {x}
            if is_valid(trial):
                S = trial
        return S
    
    # Strategy 1: Greedy from high to low (seed approach)
    best_set = set(range(192, 257))
    best_size = len(best_set)
    
    # Strategy 2: Greedy from low to high - try to include small numbers
    # Small numbers are more restrictive, so this may give different results
    order_low_to_high = list(range(1, N + 1))
    S2 = greedy_construction(order_low_to_high)
    if len(S2) > best_size:
        best_set = S2
        best_size = len(S2)
    
    # Strategy 3: Greedy with various orderings
    for seed in range(20):
        if time.time() - start > DEADLINE:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S = greedy_construction(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Strategy 4: Start from a base set and try to add/remove elements (local search)
    # Try different starting points
    for base_start in [128, 160, 192, 200, 220]:
        if time.time() - start > DEADLINE:
            break
        base = set(range(base_start, N + 1))
        # Try to add elements from below base_start
        for x in range(1, base_start):
            if time.time() - start > DEADLINE:
                break
            trial = base | {x}
            if is_valid(trial):
                base = trial
        # Now try to add elements that were excluded
        for x in range(1, N + 1):
            if x in base:
                continue
            if time.time() - start > DEADLINE:
                break
            trial = base | {x}
            if is_valid(trial):
                base = trial
        if len(base) > best_size:
            best_set = base
            best_size = len(base)
    
    # Strategy 5: Local search - swap elements to improve
    if time.time() - start < DEADLINE:
        current = set(best_set)
        improved = True
        while improved and time.time() - start < DEADLINE:
            improved = False
            # Try adding each excluded element
            for x in range(1, N + 1):
                if x in current:
                    continue
                trial = current | {x}
                if is_valid(trial):
                    current = trial
                    improved = True
                    break
            # Try removing one element and adding two
            if not improved:
                for rem in sorted(current):
                    if time.time() - start > DEADLINE:
                        break
                    without_rem = current - {rem}
                    for add1 in range(1, N + 1):
                        if add1 in without_rem:
                            continue
                        trial1 = without_rem | {add1}
                        if is_valid(trial1):
                            for add2 in range(1, N + 1):
                                if add2 in trial1 or add2 == rem:
                                    continue
                                trial2 = trial1 | {add2}
                                if is_valid(trial2) and len(trial2) > len(current):
                                    current = trial2
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Strategy 6: Try to build from scratch with a smarter greedy
    # Use the insight that large numbers are less restrictive
    # Try: take all numbers > n/2, then try to add numbers in [n/4, n/2], etc.
    if time.time() - start < DEADLINE:
        S = set(range(N // 2 + 1, N + 1))  # 129..256
        # Try to add from 65..128
        for x in range(N // 4 + 1, N // 2 + 1):
            if time.time() - start > DEADLINE:
                break
            trial = S | {x}
            if is_valid(trial):
                S = trial
        # Try to add from 33..64
        for x in range(N // 8 + 1, N // 4 + 1):
            if time.time() - start > DEADLINE:
                break
            trial = S | {x}
            if is_valid(trial):
                S = trial
        # Try to add from 17..32
        for x in range(N // 16 + 1, N // 8 + 1):
            if time.time() - start > DEADLINE:
                break
            trial = S | {x}
            if is_valid(trial):
                S = trial
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Strategy 7: Randomized restarts with different thresholds
    for trial_seed in range(50):
        if time.time() - start > DEADLINE:
            break
        rng = random.Random(1000 + trial_seed)
        threshold = rng.randint(100, 256)
        S = set(range(threshold, N + 1))
        # Add random elements
        candidates = [x for x in range(1, threshold) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() - start > DEADLINE:
                break
            trial = S | {x}
            if is_valid(trial):
                S = trial
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    # Final: try to add any remaining elements to best_set
    if time.time() - start < DEADLINE:
        for x in range(1, N + 1):
            if x in best_set:
                continue
            trial = best_set | {x}
            if is_valid(trial):
                best_set = trial
    
    return sorted(best_set)
