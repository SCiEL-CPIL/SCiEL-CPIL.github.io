import time
import random
import math

def _score(U):
    """Compute the raw score for set U."""
    if not U or 0 not in U:
        return -1
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return -1
    maxU = U[-1]
    if maxU == 0:
        return -1
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    P = len(sum_set)
    D = len(diff_set)
    if P == 0:
        return -1
    ratio = D / P
    denom = math.log(2 * maxU + 1)
    if denom <= 0:
        return -1
    return 1.0 + math.log(ratio) / denom

def construct():
    deadline = time.time() + 140
    best_score = 0
    best_set = [0, 1, 3]
    best_score = _score(best_set)
    
    random.seed(42)
    
    # Try structured initial sets
    candidates = []
    # Arithmetic progressions
    for maxK in range(1, 40):
        for step in range(1, min(maxK, 10) + 1):
            cand = list(range(0, maxK + 1, step))
            candidates.append(cand)
    # Geometric-like
    for base in range(2, 6):
        cand = [0]
        v = 1
        while v < 50:
            cand.append(v)
            v *= base
        candidates.append(cand)
    # Random
    for size in range(3, 15):
        for _ in range(10):
            cand = [0] + random.sample(range(1, 60), size - 1)
            candidates.append(sorted(cand))
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_set = cand[:]
    
    # Local search
    current_set = best_set[:]
    current_score = best_score
    temp = 0.5
    cooling = 0.9995
    
    while time.time() < deadline - 5:
        neighbor = current_set[:]
        op = random.choice(['add', 'remove', 'swap'])
        
        if op == 'add':
            max_elem = max(neighbor) if neighbor else 0
            new_elem = random.randint(1, max_elem + random.randint(1, 15))
            if new_elem not in neighbor:
                neighbor.append(new_elem)
                neighbor.sort()
        elif op == 'remove':
            if len(neighbor) > 2:
                idx = random.randint(1, len(neighbor) - 1)
                neighbor.pop(idx)
        elif op == 'swap':
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                max_elem = max(neighbor)
                neighbor[idx] = random.randint(1, max_elem + 10)
                if neighbor[idx] in neighbor[:idx] or neighbor[idx] in neighbor[idx+1:]:
                    pass
                else:
                    neighbor.sort()
        
        if len(neighbor) < 2 or 0 not in neighbor:
            continue
        
        new_score = _score(neighbor)
        if new_score > current_score:
            current_set = neighbor
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = neighbor[:]
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / temp)
                if random.random() < prob:
                    current_set = neighbor
                    current_score = new_score
        
        temp *= cooling
    
    return best_set
