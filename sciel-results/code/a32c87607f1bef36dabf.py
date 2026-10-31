import math
import time
import random

def compute_score(u):
    """Compute the score for a given set U."""
    if not u:
        return 0.0
    max_u = max(u)
    m = 2 * max_u + 1
    if m <= 1:
        return 1.0
    
    # Sumset
    sumset = set()
    for a in u:
        for b in u:
            sumset.add(a + b)
    
    # Diffset
    diffset = set()
    for a in u:
        for b in u:
            diffset.add(a - b)
    
    r = len(diffset) / len(sumset)
    return 1 + math.log(r) / math.log(m)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Known good Golomb rulers as starting points
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
    ]
    
    best_set = None
    best_score = 0.0
    
    # Evaluate initial candidates
    for u in initial_candidates:
        if time.time() > deadline:
            break
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        score = compute_score(u)
        if score > best_score:
            best_score = score
            best_set = u
    
    if best_set is None:
        return [0, 1, 3]
    
    # Local search: try to improve by adding/removing elements
    # Use hill-climbing with occasional random restarts
    current_set = list(best_set)
    current_score = best_score
    
    # Try adding new elements
    max_val = max(current_set) if current_set else 0
    
    # Generate candidate positions to try
    # Focus on positions that might improve the ratio
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        # Randomly choose to add, remove, or modify
        choice = random.random()
        
        if choice < 0.5 and len(current_set) < 30:
            # Try adding a new element
            # Prefer positions that are "far" from existing elements
            min_pos = 0
            max_pos = max(current_set) * 2 + 10 if current_set else 10
            
            # Try several random positions
            for _ in range(5):
                if time.time() > deadline:
                    break
                new_val = random.randint(min_pos, max_pos)
                if new_val in current_set:
                    continue
                
                new_set = sorted(current_set + [new_val])
                score = compute_score(new_set)
                if score > current_score + 1e-10:
                    current_set = new_set
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_set = list(current_set)
                    break
        
        elif choice < 0.8 and len(current_set) > 3:
            # Try removing an element
            # Remove a random element (not 0)
            removable = [x for x in current_set if x != 0]
            if removable:
                idx = random.choice(removable)
                new_set = [x for x in current_set if x != idx]
                score = compute_score(new_set)
                if score > current_score + 1e-10:
                    current_set = new_set
                    current_score = score
                    if score > best_score:
                        best_score = score
                        best_set = list(current_set)
        
        else:
            # Try modifying an element
            modifiable = [x for x in current_set if x != 0]
            if modifiable:
                idx = random.choice(modifiable)
                # Try shifting it
                for delta in [-2, -1, 1, 2, 3, 5]:
                    if time.time() > deadline:
                        break
                    new_val = idx + delta
                    if new_val < 0 or new_val in current_set:
                        continue
                    
                    new_set = sorted([x if x != idx else new_val for x in current_set])
                    score = compute_score(new_set)
                    if score > current_score + 1e-10:
                        current_set = new_set
                        current_score = score
                        if score > best_score:
                            best_score = score
                            best_set = list(current_set)
                        break
    
    # Final polish: try small adjustments to the best set found
    if best_set and time.time() < deadline - 5:
        for _ in range(200):
            if time.time() > deadline:
                break
            # Try small perturbations
            modifiable = [x for x in best_set if x != 0]
            if not modifiable:
                break
            idx = random.choice(modifiable)
            for delta in [-1, 1, -2, 2]:
                if time.time() > deadline:
                    break
                new_val = idx + delta
                if new_val < 0 or new_val in best_set:
                    continue
                
                new_set = sorted([x if x != idx else new_val for x in best_set])
                score = compute_score(new_set)
                if score > best_score + 1e-10:
                    best_score = score
                    best_set = new_set
                    break
    
    return best_set
