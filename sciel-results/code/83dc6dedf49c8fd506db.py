import math
import time
import random

def _score(U):
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    n_plus = len(plus)
    n_minus = len(minus)
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _hill_climb(U, max_iter=2000, time_limit=10.0):
    U = sorted(set(U))
    best_score = _score(U)
    best_U = list(U)
    start = time.time()
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        current_score = _score(best_U)
        improved = False
        # Try adding a random element
        if random.random() < 0.5:
            # Add element
            max_val = max(best_U) if best_U else 0
            # Try values in a range around max_val or small values
            candidates = []
            for _ in range(10):
                if random.random() < 0.5:
                    candidates.append(random.randint(0, max_val * 2 + 10))
                else:
                    candidates.append(random.randint(0, 50))
            
            for c in candidates:
                if c in best_U:
                    continue
                new_U = list(best_U) + [c]
                s = _score(new_U)
                if s > current_score:
                    current_score = s
                    best_U = new_U
                    improved = True
                    break
        else:
            # Try removing an element
            if len(best_U) > 1:
                idx = random.randint(0, len(best_U) - 1)
                if best_U[idx] == 0:
                    continue
                new_U = best_U[:idx] + best_U[idx+1:]
                s = _score(new_U)
                if s > current_score:
                    current_score = s
                    best_U = new_U
                    improved = True
        
        if not improved:
            # Try swap
            if len(best_U) > 1:
                i = random.randint(0, len(best_U) - 1)
                if best_U[i] == 0:
                    continue
                j = random.randint(0, len(best_U) - 1)
                if i == j:
                    continue
                new_U = list(best_U)
                new_U[i] = new_U[j]
                if len(new_U) > 1:
                    new_U.pop()
                s = _score(new_U)
                if s > current_score:
                    current_score = s
                    best_U = new_U
                    improved = True

    return best_U, best_score

def construct():
    deadline = time.time() + 110.0
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Generate seeds
    seeds = []
    seeds.append([0, 1, 3])
    seeds.append([0, 1, 3, 9])
    seeds.append([0, 1, 4])
    seeds.append([0, 2, 3])
    
    # Random seeds
    for _ in range(20):
        size = random.randint(3, 10)
        s = random.sample(range(1, 100), size)
        s.append(0)
        seeds.append(s)
        
    # Structured seeds
    for base in [2, 3, 4]:
        s = [0]
        v = 1
        for _ in range(5):
            s.append(v)
            v *= base
        seeds.append(s)

    for seed in seeds:
        if time.time() > deadline:
            break
        U, score = _hill_climb(seed, max_iter=500, time_limit=5.0)
        if score > best_score:
            best_score = score
            best_U = U
            
    return best_U
