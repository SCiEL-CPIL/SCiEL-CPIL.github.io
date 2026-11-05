import time
import random
import math

def set_sizes(A):
    A = list(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods)

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin under 60s
    
    # We will search for the optimal "shape" in a small integer range,
    # then scale it up to fit the 10000 limit.
    # The objective function is scale-invariant: 
    # |kA+kA| = |A+A| and |kA*kA| = |A*A|.
    
    best_obj = float('inf')
    best_small_set = None
    
    # Search in a small domain to allow dense exploration
    # Max element in small domain
    SMALL_MAX = 100 
    
    def simulated_annealing_small(start_set, max_iters=2000):
        current_set = list(start_set)
        current_obj = objective(current_set)
        T = 20.0
        T_min = 0.1
        cooling_rate = 0.995
        
        best_local_set = current_set[:]
        best_local_obj = current_obj
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Generate a neighbor
            move_type = random.random()
            neighbor = current_set[:]
            
            if move_type < 0.6:
                # Replace one element with a random value in [1, SMALL_MAX]
                idx = random.randint(0, 19)
                new_val = random.randint(1, SMALL_MAX)
                if new_val not in current_set or new_val == current_set[idx]:
                    neighbor[idx] = new_val
            elif move_type < 0.8:
                # Swap two elements (preserves multiset, but changes positions? 
                # No, set is unordered. Swapping doesn't change the set.
                # Instead, let's do a "jump": replace two elements)
                i, j = random.sample(range(20), 2)
                new_val_i = random.randint(1, SMALL_MAX)
                new_val_j = random.randint(1, SMALL_MAX)
                if new_val_i not in current_set or new_val_i == current_set[i]:
                    neighbor[i] = new_val_i
                if new_val_j not in current_set or new_val_j == current_set[j]:
                    neighbor[j] = new_val_j
            else:
                # Perturb one element by a small amount
                idx = random.randint(0, 19)
                delta = random.choice([-1, 1, -2, 2, -5, 5, -10, 10])
                new_val = current_set[idx] + delta
                if 1 <= new_val <= SMALL_MAX and new_val not in current_set:
                    neighbor[idx] = new_val
                else:
                    # If invalid, try random
                    new_val = random.randint(1, SMALL_MAX)
                    if new_val not in current_set:
                        neighbor[idx] = new_val
                    else:
                        continue # Skip if can't make valid move

            # Validate distinctness
            if len(set(neighbor)) != 20:
                continue
                
            neighbor_obj = objective(neighbor)
            delta_obj = neighbor_obj - current_obj
            
            # Acceptance criterion
            if delta_obj < 0 or random.random() < math.exp(-delta_obj / T):
                current_set = neighbor
                current_obj = neighbor_obj
                
                if current_obj < best_local_obj:
                    best_local_obj = current_obj
                    best_local_set = current_set[:]
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return best_local_set, best_local_obj

    # Generate diverse seeds in the small space
    random.seed(42)
    
    seeds = []
    
    # Seed 1: AP 1..20
    seeds.append(list(range(1, 21)))
    
    # Seed 2: AP 2..21
    seeds.append(list(range(2, 22)))
    
    # Seed 3: Random small
    for _ in range(5):
        s = random.sample(range(1, SMALL_MAX + 1), 20)
        seeds.append(s)
        
    # Seed 4: "Dense" cluster
    base = random.randint(1, 20)
    s = []
    for i in range(20):
        val = base + random.randint(0, 30)
        if val not in s:
            s.append(val)
    while len(s) < 20:
        s.append(random.randint(1, SMALL_MAX))
    # Ensure distinct
    s_unique = list(set(s))
    while len(s_unique) < 20:
        s_unique.append(random.randint(1, SMALL_MAX))
    seeds.append(s_unique[:20])

    # Run SA on each seed
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Ensure seed is valid and in range
        if any(x < 1 or x > SMALL_MAX for x in seed) or len(set(seed)) != 20:
            continue
            
        sa_set, sa_obj = simulated_annealing_small(seed, max_iters=1000)
        if sa_obj < best_obj:
            best_obj = sa_obj
            best_small_set = sa_set[:]
            
    # If no valid set found (should not happen), fallback
    if best_small_set is None:
        best_small_set = list(range(1, 21))
        best_obj = objective(best_small_set)
        
    # Scale up to fit 10000
    max_val = max(best_small_set)
    scale = 10000 // max_val
    if scale < 1:
        scale = 1
        
    final_set = [x * scale for x in best_small_set]
    
    # Verify final set
    if len(set(final_set)) != 20 or any(x > 10000 for x in final_set):
        # Fallback to simple AP
        return list(range(1, 21))
        
    return final_set
