import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if subset has all unordered pair sums (including diagonals) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def largest_sidon_size(arr):
    """Find the size of the largest Sidon subset of arr."""
    n = len(arr)
    for k in range(n, 0, -1):
        for comb in combinations(range(n), k):
            if is_sidon([arr[i] for i in comb]):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    best_set = list(range(12))
    best_score = largest_sidon_size(best_set)
    
    # Try structured candidates
    structured = [
        list(range(0, 12)),
        list(range(0, 24, 2)),
        list(range(0, 36, 3)),
        [0, 1, 2, 3, 4, 5, 60, 61, 62, 63, 64, 65],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66],  # triangular numbers
        [0, 2, 4, 8, 16, 32, 64, 65, 66, 67, 68, 69],
        [0, 1, 2, 4, 8, 16, 32, 33, 34, 35, 36, 37],
    ]
    
    for cand in structured:
        if time.time() > deadline:
            break
        cand = sorted(cand)
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            score = largest_sidon_size(cand)
            if score < best_score:
                best_score = score
                best_set = cand[:]
    
    # Random + local search
    random.seed(12345)
    
    while time.time() < deadline:
        strategy = random.random()
        
        if strategy < 0.25:
            # Dense cluster
            base = random.randint(0, 110)
            upper = min(base + 15, 128)
            if upper - base < 11:
                continue
            cand = sorted(random.sample(range(base, upper + 1), 12))
        elif strategy < 0.45:
            # Two tight clusters
            b1 = random.randint(0, 40)
            b2 = random.randint(80, 115)
            c1 = random.sample(range(b1, min(b1 + 8, 129)), 6)
            c2 = random.sample(range(b2, min(b2 + 8, 129)), 6)
            if len(set(c1 + c2)) < 12:
                continue
            cand = sorted(c1 + c2)
        elif strategy < 0.6:
            # AP
            start = random.randint(0, 100)
            step = random.randint(1, 4)
            cand = [start + i * step for i in range(12)]
            if max(cand) > 128:
                continue
        elif strategy < 0.75:
            # Three clusters
            b1 = random.randint(0, 30)
            b2 = random.randint(45, 75)
            b3 = random.randint(90, 115)
            c1 = random.sample(range(b1, min(b1 + 6, 129)), 4)
            c2 = random.sample(range(b2, min(b2 + 6, 129)), 4)
            c3 = random.sample(range(b3, min(b3 + 6, 129)), 4)
            if len(set(c1 + c2 + c3)) < 12:
                continue
            cand = sorted(c1 + c2 + c3)
        else:
            # Fully random
            cand = sorted(random.sample(range(0, 129), 12))
        
        if len(set(cand)) != 12:
            continue
        
        score = largest_sidon_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
            # Local improvement
            if time.time() < deadline - 5:
                improved = True
                while improved and time.time() < deadline - 2:
                    improved = False
                    for _ in range(100):
                        if time.time() > deadline - 2:
                            break
                        i = random.randint(0, 11)
                        new_val = random.randint(0, 128)
                        if new_val in best_set:
                            continue
                        old = best_set[i]
                        best_set[i] = new_val
                        best_set.sort()
                        ns = largest_sidon_size(best_set)
                        if ns < best_score:
                            best_score = ns
                            improved = True
                        else:
                            best_set[i] = old
                            best_set.sort()
    
    return sorted(best_set)
