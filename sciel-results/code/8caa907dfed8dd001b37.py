import time
import random
import sys

DEADLINE = 50.0  # Safety margin before 60s limit

def _triple_sums(S):
    """Compute all unordered triple sums (allowing repetitions) for set S."""
    sums = set()
    n = len(S)
    S_sorted = sorted(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S_sorted[i] + S_sorted[j] + S_sorted[k])
    return sums

def _check_valid(S):
    """Check if S has all unique unordered triple sums."""
    sums = _triple_sums(S)
    expected = len(S) * (len(S) + 1) * (len(S) + 2) // 6
    return len(sums) == expected

def _greedy_build(n_max=4096, order=None):
    """Greedy construction with optional custom ordering."""
    S = []
    used_sums = set()
    pair_sums = set()
    
    if order is None:
        order = list(range(1, n_max + 1))
    
    for x in order:
        # Quick checks
        if 3 * x in used_sums:
            continue
        
        valid = True
        for y in S:
            if (2 * x + y) in used_sums or (x + 2 * y) in used_sums:
                valid = False
                break
        if not valid:
            continue
        
        for p in pair_sums:
            if (x + p) in used_sums:
                valid = False
                break
        if not valid:
            continue
        
        # Add x
        S.append(x)
        used_sums.add(3 * x)
        for y in S[:-1]:
            used_sums.add(2 * x + y)
            used_sums.add(x + 2 * y)
            pair_sums.add(x + y)
        for p in list(pair_sums):
            if p == 2 * x or p == x + x:
                continue
            used_sums.add(x + p)
        pair_sums.add(2 * x)
    
    return S

def _local_search_improve(S, n_max=4096, time_limit=10.0, rng=None):
    """Improve set S via local search: try adding elements or swapping."""
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    S = sorted(S)
    
    def can_add(x, S_set, used_sums, pair_sums):
        if 3 * x in used_sums:
            return False
        for y in S_set:
            if (2 * x + y) in used_sums or (x + 2 * y) in used_sums:
                return False
        for p in pair_sums:
            if (x + p) in used_sums:
                return False
        return True
    
    def recompute(S_list):
        used = set()
        pairs = set()
        for x in S_list:
            used.add(3 * x)
        for i, x in enumerate(S_list):
            for y in S_list[i+1:]:
                used.add(2 * x + y)
                used.add(x + 2 * y)
                pairs.add(x + y)
        for x in S_list:
            for p in pairs:
                used.add(x + p)
        return used, pairs
    
    S_set = set(S)
    used_sums, pair_sums = recompute(S)
    
    # Try adding elements
    candidates = list(set(range(1, n_max + 1)) - S_set)
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            if x in S_set:
                continue
            if can_add(x, S_set, used_sums, pair_sums):
                # Add x
                S.append(x)
                S.sort()
                S_set.add(x)
                used_sums.add(3 * x)
                for y in S_set:
                    if y == x:
                        continue
                    used_sums.add(2 * x + y)
                    used_sums.add(x + 2 * y)
                    pair_sums.add(x + y)
                for p in list(pair_sums):
                    used_sums.add(x + p)
                pair_sums.add(2 * x)
                candidates.remove(x)
                improved = True
                break
    
    # Try swaps: remove one, add one
    if time.time() - start_time < time_limit:
        swap_attempts = 0
        while time.time() - start_time < time_limit and swap_attempts < 1000:
            swap_attempts += 1
            if len(S) < 2:
                break
            # Pick a random element to remove
            remove_idx = rng.randint(0, len(S) - 1)
            x_remove = S[remove_idx]
            
            # Pick a random candidate to add
            y_add = rng.randint(1, n_max)
            if y_add == x_remove or y_add in S_set:
                continue
            
            # Try the swap
            S_test = [s for s in S if s != x_remove]
            S_test.append(y_add)
            S_test.sort()
            
            if _check_valid(S_test):
                # Accept the swap
                S = S_test
                S_set = set(S)
                used_sums, pair_sums = recompute(S)
    
    return S

def construct():
    start_time = time.time()
    n_max = 4096
    rng = random.Random(42)
    
    best = []
    
    # Multiple greedy restarts with different orderings
    for restart in range(15):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Different ordering strategies
        if restart == 0:
            order = list(range(1, n_max + 1))
        elif restart == 1:
            order = list(range(1, n_max + 1))
            random.Random(42).shuffle(order)
        elif restart == 2:
            order = list(range(n_max, 0, -1))
        elif restart == 3:
            order = list(range(1, n_max + 1, 2))  # Odd numbers
        elif restart == 4:
            order = list(range(2, n_max + 1, 2))  # Even numbers
        elif restart == 5:
            order = list(range(1, n_max + 1))
            random.Random(123).shuffle(order)
        elif restart == 6:
            order = list(range(1, n_max + 1))
            random.Random(456).shuffle(order)
        elif restart == 7:
            order = list(range(1, n_max + 1, 3))
        elif restart == 8:
            order = list(range(2, n_max + 1, 3))
        elif restart == 9:
            order = list(range(4, n_max + 1, 3))
        elif restart == 10:
            order = list(range(1, n_max + 1))
            random.Random(789).shuffle(order)
        elif restart == 11:
            order = list(range(1, n_max + 1))
            random.Random(999).shuffle(order)
        elif restart == 12:
            order = list(range(1, n_max + 1))
            random.Random(111).shuffle(order)
        elif restart == 13:
            order = list(range(1, n_max + 1))
            random.Random(222).shuffle(order)
        else:
            order = list(range(1, n_max + 1))
            random.Random(333).shuffle(order)
        
        S = _greedy_build(n_max, order)
        if len(S) > len(best):
            best = S
        
        # Local search improvement
        remaining_time = DEADLINE - (time.time() - start_time)
        if remaining_time > 10:
            S_improved = _local_search_improve(S, n_max, min(remaining_time - 5, 15.0), rng)
            if len(S_improved) > len(best):
                best = S_improved
    
    # Final local search on best
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        best = _local_search_improve(best, n_max, min(remaining_time - 2, 10.0), rng)
    
    # Verify and return
    if _check_valid(best):
        return best
    else:
        # Fallback: return greedy result
        fallback = _greedy_build(n_max)
        return fallback

if __name__ == "__main__":
    result = construct()
    print(len(result))
