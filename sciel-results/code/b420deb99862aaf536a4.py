import math
import time

def _score(u):
    """Compute the score for a set u (list of nonnegative ints containing 0)."""
    u = sorted(set(u))
    if 0 not in u:
        u = [0] + u
    if len(u) <= 1:
        return 1.0
    max_u = max(u)
    m = 2 * max_u + 1
    if m <= 1:
        return 1.0
    
    n = len(u)
    # Compute |U+U| and |U-U|
    sumset = set()
    diffset = set()
    for i in range(n):
        for j in range(i, n):  # sums: a+b where a<=b, but we need all pairs
            pass
    # Actually need all pairs for both
    for a in u:
        for b in u:
            sumset.add(a + b)
            diffset.add(a - b)
    
    r = len(diffset) / len(sumset)
    if r <= 0:
        return 1.0
    return 1 + math.log(r) / math.log(m)

def _score_fast(u):
    """Faster score computation using the fact that we can count distinct sums/diffs."""
    u = sorted(set(u))
    if 0 not in u:
        u = [0] + u
    if len(u) <= 1:
        return 1.0
    max_u = max(u)
    m = 2 * max_u + 1
    if m <= 1:
        return 1.0
    
    n = len(u)
    sumset = set()
    diffset = set()
    for i in range(n):
        ai = u[i]
        for j in range(n):
            aj = u[j]
            sumset.add(ai + aj)
            diffset.add(ai - aj)
    
    r = len(diffset) / len(sumset)
    if r <= 0:
        return 1.0
    return 1 + math.log(r) / math.log(m)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Start with known good candidates
    candidates = [
        [0, 1, 4, 10, 18, 23, 25],  # current best from seed
        [0, 1, 4, 9, 11],
        [0, 1, 4, 6],
        [0, 1, 3],
        [0, 2, 7, 12],  # Golomb ruler
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18, 25],
        [0, 1, 4, 10, 18, 24],
        [0, 1, 4, 10, 17, 23],
        [0, 1, 4, 9, 17, 23],
    ]
    
    best_set = None
    best_score = -1.0
    
    for u in candidates:
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        s = _score_fast(u)
        if s > best_score:
            best_score = s
            best_set = list(u)
    
    # Hill climbing: try adding, removing, or modifying elements
    import random
    random.seed(42)
    
    current = list(best_set)
    current_score = best_score
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        n = len(current)
        
        # Try adding a new element
        if time.time() < deadline:
            # Try adding elements near the current max
            max_u = max(current)
            for new_val in range(1, max_u + 10):
                if new_val in current:
                    continue
                candidate = sorted(current + [new_val])
                s = _score_fast(candidate)
                if s > current_score + 1e-12:
                    current = candidate
                    current_score = s
                    improved = True
                    no_improve_count = 0
                    break
        
        # Try removing an element
        if not improved and time.time() < deadline and len(current) > 2:
            for i in range(1, len(current)):  # never remove 0
                candidate = current[:i] + current[i+1:]
                s = _score_fast(candidate)
                if s > current_score + 1e-12:
                    current = candidate
                    current_score = s
                    improved = True
                    no_improve_count = 0
                    break
        
        # Try replacing an element (not 0)
        if not improved and time.time() < deadline and len(current) > 2:
            max_u = max(current)
            for i in range(1, len(current)):
                for new_val in range(1, max_u + 10):
                    if new_val in current:
                        continue
                    if new_val == current[i]:
                        continue
                    candidate = current[:i] + [new_val] + current[i+1:]
                    candidate = sorted(set(candidate))
                    if len(candidate) != len(current):
                        continue
                    s = _score_fast(candidate)
                    if s > current_score + 1e-12:
                        current = candidate
                        current_score = s
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break
        
        if not improved:
            no_improve_count += 1
            # Try random perturbations
            if no_improve_count < 50 and time.time() < deadline:
                max_u = max(current)
                for _ in range(100):
                    # Randomly modify
                    candidate = list(current)
                    op = random.choice(['add', 'remove', 'replace'])
                    if op == 'add' and len(candidate) < 15:
                        new_val = random.randint(1, max_u + 15)
                        if new_val not in candidate:
                            candidate.append(new_val)
                            candidate.sort()
                    elif op == 'remove' and len(candidate) > 2:
                        idx = random.randint(1, len(candidate) - 1)
                        candidate.pop(idx)
                    elif op == 'replace' and len(candidate) > 2:
                        idx = random.randint(1, len(candidate) - 1)
                        new_val = random.randint(1, max_u + 15)
                        if new_val not in candidate:
                            candidate[idx] = new_val
                            candidate = sorted(set(candidate))
                            if len(candidate) != len(current):
                                continue
                    
                    if len(candidate) != len(current) and op != 'add' and op != 'remove':
                        continue
                    
                    s = _score_fast(candidate)
                    if s > current_score + 1e-12:
                        current = candidate
                        current_score = s
                        improved = True
                        no_improve_count = 0
                        break
                if not improved:
                    no_improve_count += 1
            else:
                # Try larger random restarts
                if no_improve_count >= 50 and time.time() < deadline:
                    # Restart with a different initial set
                    for restart_size in range(3, 12):
                        if time.time() >= deadline:
                            break
                        # Generate a random set
                        max_val = random.randint(restart_size * 2, restart_size * 8)
                        while True:
                            trial = [0]
                            for _ in range(restart_size - 1):
                                trial.append(random.randint(1, max_val))
                            trial = sorted(set(trial))
                            if len(trial) == restart_size:
                                break
                        s = _score_fast(trial)
                        if s > current_score + 1e-12:
                            current = trial
                            current_score = s
                            no_improve_count = 0
                            break
        
        if no_improve_count > 200:
            break
    
    if current_score > best_score:
        best_set = current
        best_score = current_score
    
    return best_set
