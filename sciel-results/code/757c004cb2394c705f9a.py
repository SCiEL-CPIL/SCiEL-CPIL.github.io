import itertools
import random
import time
import math

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(POINTS)
POINT_INDEX = {p: i for i, p in enumerate(POINTS)}

# Precompute squared distances
dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    xi, yi = POINTS[i]
    for j in range(i + 1, NUM_POINTS):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        dist[i][j] = d
        dist[j][i] = d

def is_isosceles(p, a, b):
    """Check if points p, a, b form an isosceles triangle."""
    d_pa = dist[p][a]
    d_pb = dist[p][b]
    d_ab = dist[a][b]
    return d_pa == d_pb or d_pa == d_ab or d_pb == d_ab

def can_add(p, chosen_indices):
    """Check if point p can be added to the set of chosen_indices without creating an isosceles triangle."""
    for i in range(len(chosen_indices)):
        for j in range(i + 1, len(chosen_indices)):
            if is_isosceles(p, chosen_indices[i], chosen_indices[j]):
                return False
    return True

def count_addable(chosen_indices):
    """Count how many points NOT in chosen_indices can be added to chosen_indices."""
    chosen_set = set(chosen_indices)
    count = 0
    for p in range(NUM_POINTS):
        if p not in chosen_set:
            if can_add(p, chosen_indices):
                count += 1
    return count

def greedy_random():
    """Generate a random maximal no-isosceles set using greedy approach."""
    order = list(range(NUM_POINTS))
    random.shuffle(order)
    chosen = []
    for p in order:
        if can_add(p, chosen):
            chosen.append(p)
    return chosen

def local_search(initial_set, time_limit=2.0):
    """Perform local search to minimize size + penalty."""
    current_set = list(initial_set)
    current_size = len(current_set)
    current_penalty = count_addable(current_set)
    current_cost = current_size + 10.0 * current_penalty
    
    best_set = list(current_set)
    best_size = current_size
    best_penalty = current_penalty
    best_cost = current_cost
    
    start_time = time.time()
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() - start_time < time_limit and T > min_T:
        # Generate a random move
        move_type = random.random()
        new_set = list(current_set)
        
        if move_type < 0.3 and current_set:
            # Remove a point
            idx = random.randint(0, len(new_set) - 1)
            p = new_set.pop(idx)
        elif move_type < 0.6 and len(new_set) < NUM_POINTS:
            # Add a point
            available = [p for p in range(NUM_POINTS) if p not in new_set]
            if available:
                p = random.choice(available)
                new_set.append(p)
            else:
                continue
        else:
            # Swap
            if current_set:
                idx = random.randint(0, len(new_set) - 1)
                p_remove = new_set[idx]
                available = [p for p in range(NUM_POINTS) if p not in new_set]
                if available:
                    p_add = random.choice(available)
                    new_set[idx] = p_add
                else:
                    continue
            else:
                continue
        
        new_size = len(new_set)
        # Calculate new penalty
        new_penalty = count_addable(new_set)
        new_cost = new_size + 10.0 * new_penalty
        
        delta_cost = new_cost - current_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_set = new_set
            current_size = new_size
            current_penalty = new_penalty
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_set = list(current_set)
                best_size = current_size
                best_penalty = current_penalty
        
        T *= cooling
        
    return best_set, best_size, best_penalty

def construct():
    start_time = time.time()
    deadline = 330.0
    
    best_set = []
    best_size = float('inf')
    best_penalty = float('inf')
    
    # 1. Initial Randomized Greedy runs
    # Do as many as possible in first 10s
    while time.time() - start_time < 10.0:
        s = greedy_random()
        size = len(s)
        penalty = count_addable(s)
        if size < best_size or (size == best_size and penalty < best_penalty):
            best_set = s
            best_size = size
            best_penalty = penalty
            
    # 2. Local Search on the best found set and a few others
    # Collect top 5 distinct sets from greedy runs? 
    # For simplicity, just run local search on the current best and a few new random ones.
    
    candidates = [best_set]
    
    # Generate a few more candidates
    for _ in range(5):
        if time.time() - start_time > deadline - 30:
            break
        s = greedy_random()
        candidates.append(s)
        
    # Sort candidates by size to prioritize smaller ones
    candidates.sort(key=lambda x: len(x))
    
    for s in candidates:
        if time.time() - start_time > deadline - 10:
            break
        # Allocate remaining time / number of candidates
        time_left = deadline - time.time()
        # Give more time to earlier (better) candidates
        # Simple allocation: 1/3 of remaining for first, 1/3 for second, etc?
        # Or just fixed chunks.
        alloc_time = min(30.0, time_left / (len(candidates) + 1))
        
        res_set, res_size, res_penalty = local_search(s, alloc_time)
        
        # We prefer sets with penalty 0 (maximal)
        if res_penalty == 0:
            if res_size < best_size:
                best_size = res_size
                best_set = res_set
                best_penalty = 0
        else:
            # If we don't have a valid set yet, or this is "closer" to valid?
            # Actually, we always have a valid set from step 1.
            # So we only care about valid sets.
            pass

    # Convert indices back to points
    return [POINTS[i] for i in best_set]

# Run construct to ensure it works
if __name__ == "__main__":
    res = construct()
    print(len(res))
