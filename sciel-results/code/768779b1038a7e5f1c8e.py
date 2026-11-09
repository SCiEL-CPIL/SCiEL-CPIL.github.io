import time
import math
from math import gcd
import random

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 2.0

def _is_valid_solution(solution):
    if not solution:
        return False
    seen = set()
    for r, m in solution:
        if m < 2 or m > MAXIMUM or m in seen or r < 0 or r >= m:
            return False
        seen.add(m)
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            r1, m1 = solution[i]
            r2, m2 = solution[j]
            if r1 % gcd(m1, m2) == r2 % gcd(m1, m2):
                return False
    return True

def _greedy(order, deadline):
    sol = []
    used = set()
    for m in order:
        if time.time() > deadline:
            break
        if m in used:
            continue
        for r in range(m):
            ok = True
            for (rp, mp) in sol:
                if r % gcd(m, mp) == rp % gcd(mp, m):
                    ok = False
                    break
            if ok:
                sol.append([r, m])
                used.add(m)
                break
    return sol

def _local_search(sol, deadline):
    if not sol:
        return sol
    best = [list(x) for x in sol]
    all_mods = list(range(2, MAXIMUM + 1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        used = set(m for _, m in best)
        
        # Try adding
        for m in all_mods:
            if time.time() > deadline:
                break
            if m in used:
                continue
            for r in range(m):
                test = best + [[r, m]]
                if _is_valid_solution(test):
                    best = test
                    improved = True
                    break
        
        # Try reassigning
        if not improved:
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                mi = best[i][1]
                for r in range(mi):
                    if r == best[i][0]:
                        continue
                    test = [list(x) for x in best]
                    test[i][0] = r
                    if _is_valid_solution(test):
                        # Try to add
                        um = set(x[1] for x in test)
                        for ma in all_mods:
                            if ma in um:
                                continue
                            for ra in range(ma):
                                t2 = test + [[ra, ma]]
                                if _is_valid_solution(t2) and len(t2) > len(best):
                                    best = t2
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                if improved:
                    break
        
        # Try swapping
        if not improved:
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(best)):
                    if time.time() > deadline:
                        break
                    mi, mj = best[i][1], best[j][1]
                    for ri in range(mi):
                        if time.time() > deadline:
                            break
                        for rj in range(mj):
                            test = [list(x) for x in best]
                            test[i][0] = ri
                            test[j][0] = rj
                            if _is_valid_solution(test):
                                um = set(x[1] for x in test)
                                for ma in all_mods:
                                    if ma in um:
                                        continue
                                    for ra in range(ma):
                                        t2 = test + [[ra, ma]]
                                        if _is_valid_solution(t2) and len(t2) > len(best):
                                            best = t2
                                            improved = True
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                    if improved:
                        break
                if improved:
                    break
    
    return best

def construct():
    start = time.time()
    deadline = start + DEADLINE - SAFETY_MARGIN
    best = []
    all_mods = list(range(2, MAXIMUM + 1))
    
    # Fixed orderings
    even = [m for m in all_mods if m % 2 == 0]
    odd = [m for m in all_mods if m % 2 != 0]
    orderings = [even + odd, odd + even, sorted(even, reverse=True) + odd,
                 sorted(even) + odd, sorted(all_mods, reverse=True), sorted(all_mods)]
    
    def v2(n):
        c = 0
        while n % 2 == 0:
            c += 1
            n //= 2
        return c
    orderings.append(sorted(all_mods, key=lambda m: -v2(m)))
    orderings.append(sorted(all_mods, key=lambda m: v2(m)))
    
    for order in orderings:
        if time.time() > deadline:
            break
        sol = _greedy(order, deadline)
        if len(sol) > len(best):
            best = [list(x) for x in sol]
    
    # Local search
    if best:
        imp = _local_search(best, deadline)
        if len(imp) > len(best):
            best = imp
    
    # Random restarts
    for _ in range(50):
        if time.time() > deadline:
            break
        s = all_mods[:]
        random.shuffle(s)
        sol = _greedy(s, deadline)
        if len(sol) > len(best):
            best = [list(x) for x in sol]
        if best:
            imp = _local_search(best, deadline)
            if len(imp) > len(best):
                best = imp
    
    if not best:
        return [[0, 2]]
    return best
