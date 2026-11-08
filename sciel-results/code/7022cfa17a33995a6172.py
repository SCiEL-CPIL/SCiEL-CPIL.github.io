import time
import random
import math

def _score_set(U):
    """Compute the score for a given set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    # Compute sumset and difference set
    sum_set = set()
    diff_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_score = _score_set([0, 1, 3])
    best_set = [0, 1, 3]
    
    # Try various initial constructions
    initial_candidates = []
    
    # Small exhaustive search for sets up to size 6 with elements up to 15
    # This gives us good starting points
    from itertools import combinations
    
    # Try all subsets of {0,...,10} of size 3-6 that contain 0
    for size in range(3, 7):
        for combo in combinations(range(1, 11), size - 1):
            U = [0] + list(combo)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
            initial_candidates.append((s, U[:]))
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Sort candidates by score, keep top ones
    initial_candidates.sort(key=lambda x: -x[0])
    top_candidates = [c[1] for c in initial_candidates[:20]]
    
    # Simulated annealing / local search
    random.seed(42)
    
    for init_set in top_candidates[:5]:
        if time.time() > deadline:
            break
        
        current_set = init_set[:]
        current_score = _score_set(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = current_set[:]
        
        temperature = 0.1
        for step in range(5000):
            if time.time() > deadline:
                break
            
            # Generate a move
            move_type = random.random()
            new_set = current_set[:]
            
            if move_type < 0.3:
                # Add an element
                max_elem = max(current_set) if current_set else 0
                new_elem = random.randint(1, max_elem + 5)
                if new_elem not in new_set:
                    new_set.append(new_elem)
            elif move_type < 0.5:
                # Remove an element (not 0)
                if len(new_set) > 2:
                    non_zero = [x for x in new_set if x != 0]
                    if non_zero:
                        new_set.remove(random.choice(non_zero))
            elif move_type < 0.7:
                # Replace an element
                if len(new_set) > 2:
                    non_zero = [x for x in new_set if x != 0]
                    if non_zero:
                        idx = random.choice(non_zero)
                        new_set.remove(idx)
                        max_elem = max(new_set) if new_set else 0
                        new_set.append(random.randint(1, max_elem + 5))
            elif move_type < 0.9:
                # Fine-tune: shift an element by ±1
                if len(new_set) > 2:
                    non_zero = [x for x in new_set if x != 0]
                    if non_zero:
                        idx = random.choice(non_zero)
                        new_val = idx + random.choice([-1, 1])
                        if new_val > 0 and new_val not in new_set:
                            new_set.remove(idx)
                            new_set.append(new_val)
            else:
                # Swap two elements
                if len(new_set) > 3:
                    non_zero = [x for x in new_set if x != 0]
                    if len(non_zero) >= 2:
                        a, b = random.sample(non_zero, 2)
                        new_set.remove(a)
                        new_set.remove(b)
                        new_set.append(a + random.choice([-1, 0, 1]))
                        new_set.append(b + random.choice([-1, 0, 1]))
                        # Ensure no duplicates and all positive
                        new_set = list(set(x for x in new_set if x >= 0))
                        if 0 not in new_set:
                            new_set.append(0)
            
            if not new_set or 0 not in new_set:
                continue
            
            new_score = _score_set(new_set)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-10)):
                current_set = new_set
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set[:]
            
            # Cool down
            temperature *= 0.999
        
        # Restart with best found so far
        if best_score > current_score:
            current_set = best_set[:]
            current_score = best_score
    
    return sorted(best_set)
