import itertools
import time
from math import log

def compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0  # Can't be below 1 in a meaningful way, but guard
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Exhaustive search for small n and M
    # n=4, M up to 20
    for M in range(3, 21):
        if time.time() > deadline:
            break
        # U = {0, a, b, c} with 1 <= a < b < c <= M
        for a in range(1, M + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, M + 1):
                if time.time() > deadline:
                    break
                for c in range(b + 1, M + 1):
                    U = [0, a, b, c]
                    score = compute_score(U)
                    if score > best_score:
                        best_score = score
                        best_set = U
    
    # n=5, M up to 15 (smaller M because search space grows)
    for M in range(4, 16):
        if time.time() > deadline:
            break
        for a in range(1, M + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, M + 1):
                if time.time() > deadline:
                    break
                for c in range(b + 1, M + 1):
                    if time.time() > deadline:
                        break
                    for d in range(c + 1, M + 1):
                        U = [0, a, b, c, d]
                        score = compute_score(U)
                        if score > best_score:
                            best_score = score
                            best_set = U
        
    # n=6, M up to 12
    for M in range(5, 13):
        if time.time() > deadline:
            break
        for a in range(1, M + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, M + 1):
                if time.time() > deadline:
                    break
                for c in range(b + 1, M + 1):
                    if time.time() > deadline:
                        break
                    for d in range(c + 1, M + 1):
                        if time.time() > deadline:
                            break
                        for e in range(d + 1, M + 1):
                            U = [0, a, b, c, d, e]
                            score = compute_score(U)
                            if score > best_score:
                                best_score = score
                                best_set = U
    
    return best_set
