import time
import random
import math

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    sums = set()
    diffs = set()
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + math.log(ratio) / math.log(D)

def local_search_small(start_set, deadline):
    current = sorted(set(start_set))
    current_score = score_set(current)
    best_set = current[:]
    best_score = current_score
    
    # Limit search to small sets to keep denominator small
    MAX_SIZE = 12
    MAX_VAL = 100
    
    iterations = 0
    max_iter = 5000
    while iterations < max_iter and time.time() < deadline - 5:
        iterations += 1
        moved = False
        new_set = current[:]
        
        # Randomly choose a move type
        move_type = random.random()
        
        if move_type < 0.6 and len(current) > 1:
            # Change one element
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            # Try values in a neighborhood or random small values
            candidates = set()
            for d in range(-5, 6):
                candidates.add(old_val + d)
            candidates.add(random.randint(1, MAX_VAL))
            candidates.add(old_val * 2)
            candidates.add(old_val // 2 if old_val > 1 else 1)
            
            for new_val in candidates:
                if new_val <= 0: continue
                if new_val in current and current.index(new_val) != i: continue
                
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                
                if 0 not in new_set or len(new_set) < 2: continue
                if new_set[-1] > MAX_VAL: continue
                
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current[:]
                    moved = True
                    break
        elif move_type < 0.8 and len(current) < MAX_SIZE:
            # Add an element
            candidates = set()
            for x in current:
                candidates.add(x + 1)
                candidates.add(x - 1)
                candidates.add(x * 2)
            candidates.add(random.randint(1, MAX_VAL))
            candidates.add(current[-1] + 1)
            
            for new_val in candidates:
                if new_val <= 0: continue
                if new_val in current: continue
                if new_val > MAX_VAL: continue
                
                new_set = current + [new_val]
                new_set = sorted(new_set)
                
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current[:]
                    moved = True
                    break
        else:
            # Remove an element (if size is large)
            if len(current) > 3:
                i = random.randint(1, len(current) - 1)
                new_set = [x for j, x in enumerate(current) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current[:]
                    moved = True
        
        # If no move improved, we could try simulated annealing or just continue
        # For now, we stick to hill climbing. To escape local optima, 
        # we can occasionally accept a slightly worse move.
        if not moved and random.random() < 0.05:
            # Simulated Annealing step
            i = random.randint(1, len(current) - 1)
            new_val = current[i] + random.randint(-3, 3)
            if new_val > 0 and new_val not in current:
                new_set = current[:]
                new_set[i] = new_val
                new_set = sorted(set(new_set))
                if 0 in new_set and new_set[-1] <= MAX_VAL:
                    new_score = score_set(new_set)
                    # Accept if not too bad
                    if new_score > current_score - 0.05:
                        current = new_set
                        current_score = new_score

    return best_set, best_score

def construct():
    deadline = time.time() + 110
    
    # Start with known good candidates
    candidates = [
        [0, 1, 2, 4, 8],
        [0, 1, 2, 4, 7],
        [0, 1, 3, 7],
        [0, 1, 2, 5],
        [0, 1, 2, 4, 8, 15],
        [0, 1, 2, 4, 8, 16]
    ]
    
    # Generate some random small sets
    random.seed(42)
    for _ in range(20):
        n = random.randint(3, 8)
        U = [0]
        last = 0
        for _ in range(n - 1):
            last += random.randint(1, 5)
            U.append(last)
        if U[-1] <= 50:
            candidates.append(U)
    
    best_set = candidates[0]
    best_score = score_set(best_set)
    
    # Evaluate all candidates
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    # Run local search on the best few candidates
    # Sort candidates by score to prioritize
    scored = [(score_set(c), c) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    seen = set()
    top_starts = []
    for s, c in scored:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            top_starts.append(c)
        if len(top_starts) >= 5:
            break
    
    # Ensure best_set is in top_starts
    if tuple(best_set) not in [tuple(x) for x in top_starts]:
        top_starts.append(best_set)
        
    for start in top_starts:
        if time.time() > deadline - 10:
            break
        local_best, local_score = local_search_small(start, deadline)
        if local_score > best_score:
            best_score = local_score
            best_set = local_best[:]
            
    return best_set
