import random
import time

def _has_triple_conflict(S_set, a, b, c):
    """Check if adding a,b,c as a triple would violate (all 3 pair sums in S_set)."""
    # We need: a+b, a+c, b+c all in S_set, and a,b,c distinct
    return (a + b) in S_set and (a + c) in S_set and (b + c) in S_set

def _is_valid(S):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            for k in range(j+1, n):
                c = lst[k]
                if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                    return False
    return True

def _check_conflict_if_add(S_set, S_list, x):
    """Check if adding x to S creates a forbidden triple.
    A triple (a,b,c) all in S∪{x} with all 3 pair sums in S∪{x} is forbidden.
    Since x is new, the triple must include x (otherwise it was already forbidden).
    So we need: exists a,b in S, a!=b, a!=x, b!=x, such that
      x+a, x+b, a+b are all in S∪{x}.
    x+a and x+b are in S∪{x} iff they are in S (since x+a != x unless a=0, not possible).
    a+b must be in S∪{x}; a+b != x is possible but let's just check in S_set or ==x.
    """
    # For each pair (a,b) in S with a<b:
    #   need: (x+a) in S_set, (x+b) in S_set, and (a+b) in S_set or (a+b)==x
    for i, a in enumerate(S_list):
        if a == x:
            continue
        for b in S_list[i+1:]:
            if b == x:
                continue
            if (x + a) in S_set and (x + b) in S_set:
                ab = a + b
                if ab in S_set or ab == x:
                    return True
    return False

def _greedy_construct(rng, n=128):
    """Randomized greedy: shuffle, add if no conflict."""
    S = []
    S_set = set()
    candidates = list(range(1, n+1))
    rng.shuffle(candidates)
    for x in candidates:
        if not _check_conflict_if_add(S_set, S, x):
            S.append(x)
            S_set.add(x)
    return S

def _local_search(S, rng, n=128, max_iters=5000):
    """Local search: try additions, removals+additions, swaps."""
    S = list(S)
    S_set = set(S)
    all_vals = set(range(1, n+1))
    
    for _ in range(max_iters):
        improved = False
        # Try adding
        missing = all_vals - S_set
        if missing:
            # Try a random subset of missing
            miss_list = list(missing)
            rng.shuffle(miss_list)
            for x in miss_list[:50]:
                if not _check_conflict_if_add(S_set, S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
        # Try remove one, add two
        if not improved and len(S) >= 2:
            # Pick a random element to remove, then try to add two
            idx = rng.randrange(len(S))
            remove_val = S[idx]
            new_S = S[:idx] + S[idx+1:]
            new_S_set = S_set - {remove_val}
            missing = all_vals - new_S_set
            miss_list = list(missing)
            rng.shuffle(miss_list)
            added = []
            for x in miss_list:
                if x == remove_val:
                    continue
                if not _check_conflict_if_add(new_S_set, new_S, x):
                    added.append(x)
                    new_S.append(x)
                    new_S_set.add(x)
                    if len(added) == 2:
                        break
            if len(added) >= 2:
                S = new_S
                S_set = new_S_set
                improved = True
        if not improved:
            break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    best_S = None
    best_len = 0
    
    rng = random.Random(42)
    
    # Run multiple restarts
    while time.time() < deadline:
        S = _greedy_construct(rng)
        S = _local_search(S, rng)
        if len(S) > best_len:
            best_len = len(S)
            best_S = list(S)
            # Verify
            if not _is_valid(best_S):
                best_S = None
                best_len = 0
                continue
        # If we've found a good set, keep searching
        if best_len >= 80:
            # Try more local search on best
            pass
    
    if best_S is None:
        # Fallback to seed
        return list(range(65, 129))
    
    return list(best_S)
