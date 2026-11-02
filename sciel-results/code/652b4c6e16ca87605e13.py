import time
import math
from itertools import combinations

def compute_score(U):
    U = sorted(set(U))
    n = len(U)
    if n <= 1:
        return 1.0
    max_u = max(U)
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 115
    
    best_score = 0
    best_U = [0, 1]
    
    # Search over small sets
    # Try all subsets of {0,...,M} of size up to N
    M = 20
    N = 12
    
    # Brute force for small N
    for n in range(2, N+1):
        for combo in combinations(range(1, M+1), n-1):
            U = [0] + list(combo)
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # Local search from best found
    while time.time() < deadline:
        improved = False
        max_u = max(best_U)
        # Try adding
        for c in range(1, max_u + 15):
            if c not in best_U:
                new_U = best_U + [c]
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_U = new_U
                    best_score = s
                    improved = True
                    break
        if improved:
            continue
        # Try removing
        for c in best_U:
            if c != 0:
                new_U = [x for x in best_U if x != c]
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_U = new_U
                    best_score = s
                    improved = True
                    break
        if improved:
            continue
        # Try swapping
        for i, c in enumerate(best_U):
            if c == 0:
                continue
            for d in range(1, max_u + 15):
                if d not in best_U:
                    new_U = [x for x in best_U if x != c] + [d]
                    s = compute_score(new_U)
                    if s > best_score + 1e-12:
                        best_U = new_U
                        best_score = s
                        improved = True
                        break
            if improved:
                break
        if not improved:
            # Try a random restart with perturbation
            import random
            random.seed(int(time.time()*1000) % 10000)
            perturbed = best_U[:]
            if len(perturbed) > 2:
                idx = random.randint(1, len(perturbed)-1)
                perturbed[idx] = random.randint(1, max(perturbed) + 10)
                if 0 not in perturbed:
                    perturbed.append(0)
                s = compute_score(perturbed)
                if s > best_score:
                    best_U = perturbed
                    best_score = s
            else:
                # Just try a different small set
                for _ in range(10):
                    trial = [0] + random.sample(range(1, 30), random.randint(2, 8))
                    s = compute_score(trial)
                    if s > best_score:
                        best_U = trial
                        best_score = s
            if not improved and time.time() > deadline - 20:
                break
    
    return sorted(set(best_U))
