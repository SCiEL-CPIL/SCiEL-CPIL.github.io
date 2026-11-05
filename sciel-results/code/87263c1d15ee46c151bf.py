import time
from itertools import combinations

def is_sidon(subset):
    """Check if a given subset has all unique pair sums including diagonals."""
    sums = set()
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(candidate_set):
    """Find the size of the largest Sidon subset of the given candidate set."""
    n = len(candidate_set)
    # Check from largest possible subset size down
    for k in range(n, 0, -1):
        for subset in combinations(candidate_set, k):
            if is_sidon(subset):
                return k
    return 0

def max_sidon_size_fast(candidate_set, max_check=6):
    """Fast check: only verify if Sidon subset of size >= max_check exists."""
    n = len(candidate_set)
    for k in range(n, max_check - 1, -1):
        for subset in combinations(candidate_set, k):
            if is_sidon(subset):
                return k
    return max_check - 1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (90s kill, leave buffer)
    
    # Structured candidates designed to maximize pair-sum collisions
    candidates = []
    
    # Candidate A: Arithmetic progression (many sum collisions)
    candidates.append(list(range(0, 25, 2)))  # [0,2,4,6,8,10,12,14,16,18,20,22]
    
    # Candidate B: Two interleaved APs
    candidates.append([0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21])
    
    # Candidate C: Quadratic-like spacing
    candidates.append([0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66])
    
    # Candidate D: Dense cluster + sparse tail
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 100, 110, 120])
    
    # Candidate E: Modulo structure - all same residue mod small number
    candidates.append(list(range(0, 25, 2)))  # Even numbers
    
    # Candidate F: Mix of close pairs
    candidates.append([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    
    # Candidate G: Fibonacci-like
    candidates.append([0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 120])
    
    # Candidate H: Symmetric around center
    candidates.append([0, 1, 2, 3, 64, 65, 66, 67, 126, 127, 128, 125])
    
    # Candidate I: Clustered in thirds
    candidates.append([0, 1, 2, 3, 42, 43, 44, 45, 84, 85, 86, 87])
    
    # Candidate J: Geometric-ish
    candidates.append([0, 1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112])
    
    # Candidate K: Very dense small range
    candidates.append(list(range(12)))
    
    # Candidate L: Alternating dense/sparse
    candidates.append([0, 1, 2, 3, 10, 11, 20, 21, 30, 31, 40, 41])
    
    # Candidate M: All multiples of 3
    candidates.append(list(range(0, 37, 3)))  # [0,3,6,9,12,15,18,21,24,27,30,33]
    
    # Candidate N: Two clusters
    candidates.append([0, 1, 2, 3, 4, 5, 60, 61, 62, 63, 64, 65])
    
    best_set = candidates[0]
    best_size = 13  # Start high
    
    for c in candidates:
        if time.time() > deadline:
            break
        if len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        sz = max_sidon_size(c)
        if sz < best_size:
            best_size = sz
            best_set = c
            if best_size <= 3:
                return best_set
    
    # If we haven't found size <= 4, do targeted random search
    if best_size > 4:
        import random
        random.seed(42)
        
        # Try biased random: cluster elements
        while time.time() < deadline:
            # Strategy: pick a base, then add elements with small offsets
            base = random.randint(0, 100)
            offsets = random.sample(range(0, 30), 12)
            s = sorted(set(base + o for o in offsets))
            if len(s) < 12:
                # Fill in missing
                remaining = [x for x in range(129) if x not in s]
                s.extend(random.sample(remaining, 12 - len(s)))
                s = sorted(s[:12])
            
            if len(set(s)) != 12 or any(x < 0 or x > 128 for x in s):
                continue
                
            sz = max_sidon_size_fast(s, max_check=best_size)
            if sz < best_size:
                # Verify with exact
                sz_exact = max_sidon_size(s)
                if sz_exact < best_size:
                    best_size = sz_exact
                    best_set = s
                    if best_size <= 3:
                        break
    
    return best_set
