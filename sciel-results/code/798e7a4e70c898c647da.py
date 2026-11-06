    def can_add(S, x):
        # 1. Triples in S
        n = len(S)
        if n >= 3:
            # Manual loop might be faster than combinations?
            # combinations is C, so it's fast.
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # 2. Pairs in S
        for y in S:
            for a, b in combinations(S, 2):
                if a == y or b == y:
                    continue
                if (x + a + b) % y == 0:
                    return False
        return True
