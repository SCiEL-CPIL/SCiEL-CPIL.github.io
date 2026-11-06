"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    def is_valid(S):
        n = len(S)
        if n == 0:
            return True
        recips = [Fraction(1, x) for x in S]
        for i in range(n):
            target = recips[i]
            others = [recips[j] for j in range(n) if j != i]
            m = len(others)
            if m == 0:
                continue
            if sum(others) < target:
                continue
            if m <= 20:
                for mask in range(1, 1 << m):
                    s = Fraction(0)
                    for bit in range(m):
                        if mask & (1 << bit):
                            s += others[bit]
                            if s > target:
                                break
                    if s == target:
                        return False
            else:
                for r in range(1, min(m, 4) + 1):
                    for subset in combinations(range(m), r):
                        s = sum(others[j] for j in subset)
                        if s == target:
                            return False
        return True
    
    S = list(range(1, 25))
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            others = [x for j, x in enumerate(S) if j != i]
            target = Fraction(1, S[i])
            m = len(others)
            if m == 0:
                continue
            recips = [Fraction(1, x) for x in others]
            if sum(recips) < target:
                continue
            found = False
            if m <= 20:
                for mask in range(1, 1 << m):
                    s = Fraction(0)
                    for bit in range(m):
                        if mask & (1 << bit):
                            s += recips[bit]
                            if s > target:
                                break
                    if s == target:
                        found = True
                        break
            else:
                for r in range(1, min(m, 4) + 1):
                    for subset in combinations(range(m), r):
                        s = sum(recips[j] for j in subset)
                        if s == target:
                            found = True
                            break
                        if s > target:
                            break
                    if found:
                        break
            if found:
                S.pop(i)
                changed = True
                break
    
    for x in range(1, 25):
        if time.time() > deadline:
            break
        if x not in S:
            if is_valid(S + [x]):
                S.append(x)
    
    return sorted(S) if S else [13]
# EVOLVE-BLOCK-END
