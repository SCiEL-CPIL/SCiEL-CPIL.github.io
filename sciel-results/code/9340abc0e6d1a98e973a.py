"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best)
    
    p1 = 10**9 + 7
    p2 = 10**9 + 9
    
    def is_dissociated_mod(S, p):
        n = len(S)
        invs = [pow(a, p - 2, p) for a in S]
        sums = {0}
        for inv in invs:
            new_sums = set()
            for s in sums:
                new_sums.add((s + inv) % p)
            sums |= new_sums
            if len(sums) < (1 << (invs.index(inv) + 1)):  # early termination
                # Actually this check is wrong, let me just count
                pass
        # After processing all, check if we have 2^n distinct sums
        # But we need to track count properly
        return len(sums) == (1 << n)
    
    # Better: build incrementally and check size at each step
    def is_dissociated_mod_fast(S, p):
        n = len(S)
        invs = [pow(a, p - 2, p) for a in S]
        sums = {0}
        for i, inv in enumerate(invs):
            new_sums = set()
            for s in sums:
                new_sums.add((s + inv) % p)
            sums = sums | new_sums
            # After adding element i+1, should have 2^(i+1) distinct sums
            if len(sums) != (1 << (i + 1)):
                return False
        return True
    
    def is_dissociated_fast(S):
        if not is_dissociated_mod_fast(S, p1):
            return False
        if not is_dissociated_mod_fast(S, p2):
            return False
        from fractions import Fraction
        n = len(S)
        if n <= 15:
            sums = {Fraction(0)}
            for a in S:
                inv = Fraction(1, a)
                new_sums = set()
                for s in sums:
                    new_sums.add(s + inv)
                sums = sums | new_sums
                if len(sums) != (1 << (list(S).index(a) + 1)):
                    # This index trick is wrong, let me fix
                    pass
            return len(sums) == (1 << n)
        return True
    
    # Exact check
    def is_dissociated_exact(S):
        from fractions import Fraction
        n = len(S)
        sums = {Fraction(0)}
        for i, a in enumerate(S):
            inv = Fraction(1, a)
            new_sums = set()
            for s in sums:
                new_sums.add(s + inv)
            sums = sums | new_sums
            if len(sums) != (1 << (i + 1)):
                return False
        return True
    
    # Search: try to find larger sets
    # First, try systematic search for k=12 with early termination
    for k in range(13, best_size, -1):
        if time.time() > deadline:
            break
        found = False
        # Try all combinations but with early termination in the check
        for combo in combinations(range(1, 23), k):
            if time.time() > deadline:
                break
            S = list(combo)
            if is_dissociated_mod_fast(S, p1) and is_dissociated_mod_fast(S, p2):
                if is_dissociated_exact(S):
                    best = S
                    best_size = k
                    found = True
                    break
        if found:
            break
    
    # If we didn't find size 13 or 12, try random search for size 12
    if best_size < 12:
        if time.time() < deadline:
            for _ in range(50000):
                if time.time() > deadline:
                    break
                S = random.sample(range(1, 23), 12)
                if is_dissociated_mod_fast(S, p1) and is_dissociated_mod_fast(S, p2):
                    if is_dissociated_exact(S):
                        best = S
                        best_size = 12
                        break
    
    return best
# EVOLVE-BLOCK-END
