"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    deadline = time.time() + 50  # safety margin before 90s kill
    
    def can_add(S, a):
        """Check if a can be added to S without violating constraints."""
        for d in S:
            if d >= a:
                continue
            for c in S:
                if c < a:
                    continue
                if (a + c) % d == 0:
                    return False
            for b in S:
                if b <= d or b > a:
                    continue
                if (b + a) % d == 0:
                    return False
        for b in S:
            if b <= a:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % a == 0:
                    return False
        return True
    
    def greedy_add(S, candidates):
        """Try to add candidates to S in the given order."""
        for a in candidates:
            if time.time() > deadline:
                break
            if a not in S and can_add(S, a):
                S.add(a)
        return S
    
    # Start with upper tail
    k = 171
    S = set(range(k, n + 1))
    
    # Try multiple orderings for adding smaller elements
    S1 = set(S)
    greedy_add(S1, range(k - 1, 0, -1))
    
    S2 = set(S)
    greedy_add(S2, range(1, k))
    
    def num_divisors(x):
        count = 0
        for i in range(1, int(x**0.5) + 1):
            if x % i == 0:
                count += 1
                if i != x // i:
                    count += 1
        return count
    
    S3 = set(S)
    small_nums = sorted(range(1, k), key=lambda x: num_divisors(x))
    greedy_add(S3, small_nums)
    
    best_S = max([S, S1, S2, S3], key=len)
    
    # Local search: try removing one element and adding another
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_size = len(best_S)
        for x in sorted(best_S):
            if time.time() > deadline - 5:
                break
            best_S.discard(x)
            for y in range(1, n + 1):
                if y in best_S:
                    continue
                if can_add(best_S, y):
                    best_S.add(y)
                    if len(best_S) > current_size:
                        improved = True
                        break
                    else:
                        best_S.discard(y)
            if time.time() > deadline - 5:
                break
            best_S.add(x)
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
