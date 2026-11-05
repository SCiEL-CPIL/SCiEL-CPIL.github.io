def construct():
    n = 256
    S = []
    # Greedy addition from largest to smallest
    # When adding x, x is the smallest in S_new.
    # Check if x divides any sum of two larger elements in S.
    for x in range(n, 0, -1):
        valid = True
        # Check against all pairs (b, c) in S such that x < b <= c
        # Since S is built from larger numbers, all elements in S are > x.
        # So we just need to check all pairs in S.
        # Optimization: If 3*x > 2*n, no pair can sum to a multiple of x.
        if 3 * x > 2 * n:
            S.append(x)
            continue
            
        # Check pairs
        # S is a list. We can use set for faster lookup?
        # We need to check if there exist b, c in S such that b+c is a multiple of x.
        # Multiples: 3x, 4x, ... up to 2n.
        # For each multiple m, check if there is b in S such that c = m - b is in S and b <= c.
        # Since we iterate x downwards, S contains numbers > x.
        # Max sum is 2n.
        S_set = set(S)
        for k in range(3, 2 * n // x + 1):
            target = k * x
            # b must be in S, c = target - b must be in S.
            # b < c or b == c.
            # b > x.
            # Iterate b in S? S can be large.
            # Better: Iterate b in range(x+1, target//2 + 1)?
            # But b must be in S.
            # Let's just check if any b in S has target - b in S.
            # To avoid O(|S|^2), we can do:
            # for b in S:
            #    if b > target: break
            #    if target - b in S_set:
            #        if b <= target - b:
            #             valid = False
            #             break
            # This is O(|S|) per multiple.
            # Number of multiples is ~ n/x.
            # Total O(n^2/x). Sum over x is O(n^2 log n).
            # n=256, n^2 log n ~ 65000 * 8 ~ 500k. Very fast.
            
            for b in S:
                if b > target:
                    break
                c = target - b
                if c in S_set:
                    if b <= c:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            S.append(x)
            
    return S
