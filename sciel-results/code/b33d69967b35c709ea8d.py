import time
import random
from fractions import Fraction

def is_valid(S):
    S_set = set(S)
    for i in S:
        others = [j for j in S if j != i]
        target = Fraction(1, i)
        sums = {Fraction(0)}
        for j in others:
            new_sums = set()
            for s in sums:
                new_sums.add(s + Fraction(1, j))
            sums = sums.union(new_sums)
            if max(sums) < target:
                break
            if target in sums:
                return False
    return True

def construct():
    start_time = time.time()
    best_set = list(range(13, 25)) # Size 12
    
    # Try greedy descending
    S = []
    for i in range(24, 0, -1):
        candidate = S + [i]
        if is_valid(candidate):
            S = candidate
        if len(S) > len(best_set):
            best_set = S
        if time.time() - start_time > 40:
            break
            
    # Try greedy ascending
    S = []
    for i in range(1, 25):
        candidate = S + [i]
        if is_valid(candidate):
            S = candidate
        if len(S) > len(best_set):
            best_set = S
        if time.time() - start_time > 50:
            break
            
    # Try random shuffles
    for _ in range(100):
        if time.time() - start_time > 80:
            break
        perm = list(range(1, 25))
        random.shuffle(perm)
        S = []
        for i in perm:
            candidate = S + [i]
            if is_valid(candidate):
                S = candidate
        if len(S) > len(best_set):
            best_set = S
            
    return best_set
