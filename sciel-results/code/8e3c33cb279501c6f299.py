from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    def is_valid(S):
        S_list = list(S)
        for u in S_list:
            others = [v for v in S_list if v != u]
            target = Fraction(1, u)
            achievable = {Fraction(0, 1)}
            for v in others:
                new_sums = set()
                for s in achievable:
                    ns = s + Fraction(1, v)
                    if ns <= target:
                        new_sums.add(ns)
                achievable |= new_sums
                if target in achievable:
                    return False
            # Check if target is achievable (and not the empty sum)
            if target in achievable and target != Fraction(0, 1):
                return False
        return True
    
    # Greedy: add elements from largest to smallest
    S = set()
    for u in range(24, 0, -1):
        if (time.time() - start_time) > deadline:
            break
        S_test = S | {u}
        if is_valid(S_test):
            S = S_test
    
    return sorted(S)
