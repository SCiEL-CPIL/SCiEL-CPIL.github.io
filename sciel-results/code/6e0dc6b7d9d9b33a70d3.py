import time

N = 256


def _can_add(S, x):
    """Check if x can be added to sorted list S without violating the constraint.
    Only need to check triples where x is the smallest element (a=x), since
    if x is not the smallest, the constraint was already satisfied for those triples.
    For triples where x is b or c, we need a < x, so we check a in S where a < x.
    """
    # Case 1: x is the smallest (a=x). Need: for all b,c in S with b>=x, c>=b,
    # x does NOT divide b+c. But since S is sorted and we're adding x, all elements
    # in S that are >= x would be b,c. However, we also need a<b<=c, so b>x, c>=b.
    # Actually, if x is added, it becomes the smallest only if all other elements > x.
    # Let's be more careful: after adding x, for every triple (a,b,c) in new set with
    # a<b<=c, a must not divide b+c.
    # New triples involving x:
    # 1) x is a: need for all b,c in S with x<b<=c: x does not divide b+c
    # 2) x is b: need for all a,c in S with a<x<=c: a does not divide x+c
    # 3) x is c: need for all a,b in S with a<b<x: a does not divide b+x
    # 4) x is b=c: need for all a in S with a<x: a does not divide 2*x
    # But cases 2,3,4 involve a<x, and those triples already existed in S (with b or c
    # replaced by x). Wait no - if x is new, triples (a,x,c) or (a,b,x) are new.
    # Let me just check all new triples involving x.
    
    L = S  # sorted list
    m = len(L)
    
    # Find indices where elements are < x and >= x
    # Elements < x: indices 0..lo-1
    # Elements >= x: indices lo..m-1
    lo = 0
    while lo < m and L[lo] < x:
        lo += 1
    
    # Case 1: x is the smallest (a=x). b,c must be > x from S.
    # b > x, c >= b, both from S[lo:]
    for i in range(lo, m):
        b = L[i]
        if b <= x:
            continue
        for j in range(i, m):
            c = L[j]
            if (b + c) % x == 0:
                return False
    
    # Case 2: x is b (middle). a < x from S[:lo], c >= x from S[lo:].
    # Need a < x <= c, and a does not divide x + c.
    for i in range(lo):
        a = L[i]
        for j in range(lo, m):
            c = L[j]
            if (x + c) % a == 0:
                return False
    
    # Case 3: x is c (largest). a < x, b < x, a < b < x from S[:lo].
    # Need a does not divide b + x.
    for i in range(lo):
        a = L[i]
        for j in range(i + 1, lo):
            b = L[j]
            if (b + x) % a == 0:
                return False
    
    # Case 4: x is b=c (repeated). a < x from S[:lo].
    # Need a does not divide 2*x.
    for i in range(lo):
        a = L[i]
        if (2 * x) % a == 0:
            return False
    
    return True


def _validate(S):
    """Return True iff S satisfies: for all a<b<=c in S, a does not divide b+c."""
    L = sorted(S)
    m = len(L)
    for i in range(m):
        a = L[i]
        for j in range(i + 1, m):
            b = L[j]
            for k in range(j, m):
                c = L[k]
                if (b + c) % a == 0:
                    return False
    return True


def _interval_seed():
    """Construct a valid set using the interval [floor(2N/3)+1, N]."""
    n = N
    start = (2 * n) // 3 + 1
    if start < 1:
        start = 1
    return list(range(start, n + 1))


def _local_search(start_set, deadline):
    """Hill-climbing: try to add elements from 1..N not in start_set.
    Returns the best valid set found."""
    S = sorted(start_set)
    S_set = set(S)
    candidates = [x for x in range(1, N + 1) if x not in S_set]
    
    # Try adding candidates in various orders
    best_S = list(S)
    best_size = len(S)
    
    # Multiple passes with different orderings
    orders = [
        sorted(candidates, reverse=True),  # descending
        sorted(candidates),                # ascending
        sorted(candidates, key=lambda x: x % 7),  # modular
    ]
    
    for order in orders:
        if time.time() > deadline - 2:
            break
        # Work on a copy
        cur_S = list(best_S)
        cur_set = set(cur_S)
        
        for x in order:
            if time.time() > deadline - 2:
                break
            if x in cur_set:
                continue
            if _can_add(cur_S, x):
                cur_S.append(x)
                cur_S.sort()
                cur_set.add(x)
        
        if len(cur_S) > best_size:
            best_S = cur_S
            best_size = len(cur_S)
    
    # Also try: start from greedy descending
    greedy_S = []
    for x in range(N, 0, -1):
        if _can_add(greedy_S, x):
            greedy_S.append(x)
    if len(greedy_S) > best_size:
        best_S = greedy_S
        best_size = len(greedy_S)
    
    return best_S


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Start with interval seed
    seed = _interval_seed()
    
    # Run local search
    result = _local_search(seed, deadline)
    
    # Validate result
    if not _validate(set(result)):
        # Fallback to interval seed
        result = seed
    
    return sorted(result)


# Self-check at import (cheap, deterministic).
if __name__ == "__main__":
    _s = construct()
    assert _validate(set(_s)), "constructed set is invalid"
    print("size", len(_s), "first", _s[:5], "last", _s[-5:])
