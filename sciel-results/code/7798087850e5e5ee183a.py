import time
import itertools

DEADLINE = 60.0  # seconds

def construct():
    start_time = time.time()
    
    # We want to find a large subset S of [1, 24] such that for every x in S,
    # 1/x is not equal to the sum of reciprocals of any subset of S \ {x}.
    
    # Let's first check a few candidate sets and see if they are valid.
    # We'll use a helper to check if a set is valid.
    
    def is_valid(S):
        S = list(S)
        n = len(S)
        for i, x in enumerate(S):
            target = 1.0 / x
            others = [y for j, y in enumerate(S) if j != i]
            # Check all subsets of 'others'
            # If len(others) is large, this is 2^k which is too big.
            # But [1, 24] has size 24. If we pick ~16, others has size 15. 2^15 = 32768.
            # 24 * 32768 ~ 800k operations. Very fast.
            m = len(others)
            for mask in range(1, 1 << m):
                s = 0.0
                for b in range(m):
                    if (mask >> b) & 1:
                        s += 1.0 / others[b]
                        # Early break if sum exceeds target (since all terms positive)
                        if s > target:
                            break
                else:
                    if abs(s - target) < 1e-12:
                        return False
        return True

    # Let's try the set [9, 24] first.
    cand1 = list(range(9, 25))
    if is_valid(cand1):
        # Can we add 8?
        cand2 = list(range(8, 25))
        if is_valid(cand2):
            # Can we add 7?
            cand3 = list(range(7, 25))
            if is_valid(cand3):
                return cand3
            return cand2
        return cand1
    
    # Fallback to the seed
    return list(range(13, 25))
