import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if subset has all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the largest Sidon subset size of S."""
    S = sorted(S)
    n = len(S)
    for k in range(n, 0, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 0

def construct():
    start_time = time.time()
    # Evaluator kills at 90s; use 85s deadline for safety margin
    deadline = start_time + 85
    
    # Initial best candidate: Arithmetic Progression [0,1,...,11]
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    # Try various structured and random candidates
    candidates = []
    
    # 1. Pure APs with different step sizes
    for step in range(1, 11):
        if 11 * step <= 128:
            candidates.append(list(range(0, 12 * step, step)))
    
    # 2. APs with different starting points
    for start in range(0, 20, 3):
        for step in [1, 2, 3, 5, 7]:
            if start + 11 * step <= 128:
                candidates.append(list(range(start, start + 12 * step, step)))
    
    # 3. Near-AP: perturb an AP slightly
    base = list(range(12))
    for _ in range(200):
        c = base[:]
        idx = random.randint(0, 11)
        c[idx] = c[idx] + random.choice([-2, -1, 1, 2])
        if 0 <= c[idx] <= 128 and len(set(c)) == 12:
            candidates.append(c)
    
    # 4. Random sets
    for _ in range(3000):
        candidates.append(random.sample(range(129), 12))
    
    # 5. Sets with many additive collisions: use symmetric structures
    # e.g., {a, b, a+b, 2a, 2b, ...} patterns
    for a in range(1, 15):
        for b in range(a + 1, 20):
            if a + b <= 128:
                s = {0, a, b, a + b, 2 * a, 2 * b}
                # Extend to 12 elements
                while len(s) < 12:
                    s.add(random.randint(0, 128))
                if len(s) == 12 and all(0 <= x <= 128 for x in s):
                    candidates.append(list(s))
    
    # Evaluate all candidates
    for candidate in candidates:
        if time.time() > deadline:
            break
        
        # Quick filter: if candidate has duplicate elements, skip
        if len(set(candidate)) != 12:
            continue
        if any(x < 0 or x > 128 for x in candidate):
            continue
        
        score = max_sidon_size(candidate)
        
        if score < best_score:
            best_score = score
            best_set = candidate
            if best_score == 2:
                break
    
    return best_set
