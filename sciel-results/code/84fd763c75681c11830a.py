import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 88  # safety margin before 90s kill

    # Precompute divisibility: div_by[a] is a set of sums s in [2a, 512] such that s % a == 0
    div_by = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for s in range(2 * a, 513, a):
            div_by[a].add(s)

    # Start with the "safe" upper set
    K = 171
    S = set(range(K, n + 1))

    def check_add(x, S_set):
        """Check if x can be added to S_set. Returns True if valid."""
        # Case 1: x is the smallest element (a=x, b,c in S, b>c? No a<b<=c)
        # Condition: x | (b+c) for some b,c in S with x < b <= c
        # We iterate over b in S where b > x.
        # For a fixed b, we need c in S such that c >= b and b+c in div_by[x].
        # c = s - b for s in div_by[x].
        for b in S_set:
            if b <= x:
                continue
            # Check if there exists c in S_set such that c >= b and (b+c) in div_by[x]
            # i.e., s >= 2b
            for s in div_by[x]:
                if s < 2 * b:
                    continue
                c = s - b
                if c in S_set:
                    return False

        # Case 2: x is the middle element (a=b, b=x, c in S? No a<b<=c)
        # Triple is (a, b, c) with a < b <= c.
        # If x is the middle, then a < x <= c.
        # Here a is from S, c is from S.
        # Condition: a | (x+c) for some a, c in S with a < x <= c.
        # We iterate over a in S where a < x.
        # For a fixed a, we need c in S such that c >= x and x+c in div_by[a].
        # c = s - x for s in div_by[a].
        for a in S_set:
            if a >= x:
                continue
            for s in div_by[a]:
                # Need c = s - x
                # Conditions: c in S_set, c >= x
                # So s - x >= x => s >= 2x
                if s < 2 * x:
                    continue
                c = s - x
                if c in S_set:
                    return False

        # Case 3: x is the largest element (a, b in S, c=x)
        # Triple is (a, b, x) with a < b <= x.
        # Condition: a | (b+x) for some a, b in S with a < b < x (since b<=c=x and if b=x then b=c, but x is new, so b < x).
        # We iterate over a in S where a < x.
        # For a fixed a, we need b in S such that a < b < x and b+x in div_by[a].
        # b = s - x for s in div_by[a].
        for a in S_set:
            if a >= x:
                continue
            for s in div_by[a]:
                # Need b = s - x
                # Conditions: b in S_set, a < b < x
                # So a < s - x < x => a + x < s < 2x
                b = s - x
                if b > a and b < x and b in S_set:
                    return False

        return True

    # Greedy add smaller numbers from K-1 down to 1
    for x in range(K - 1, 0, -1):
        if time.time() > deadline - 30:
            break
        if check_add(x, S):
            S.add(x)

    # Local search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_size = len(S)
        
        # Try adding elements
        for a in range(1, n + 1):
            if time.time() > deadline - 5:
                break
            if a in S:
                continue
            
            if check_add(a, S):
                S.add(a)
                improved = True
                current_size += 1
        
        # 1-for-1 swaps
        if time.time() > deadline - 2:
            break
        for rem in list(S):
            if time.time() > deadline - 2:
                break
            S_without = S - {rem}
            # Try to add any element not in S_without
            for add in range(1, n + 1):
                if add in S_without:
                    continue
                if check_add(add, S_without):
                    S = S_without | {add}
                    improved = True
                    break
            if improved:
                break

    return list(S)
