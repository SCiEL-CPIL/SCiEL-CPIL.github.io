import math
import random
import time

def score_set(U):
    U = sorted(set(U))
    if not U or U[0] != 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute |U+U|
    sum_set = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sum_set.add(U[i] + U[j])
            
    # Compute |U-U|
    diff_set = set()
    for i in range(len(U)):
        for j in range(len(U)):
            diff_set.add(U[i] - U[j])
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 0:
        return 0.0

    val = 1 + math.log(ratio) / math.log(2 * max_u + 1)
    return val

def construct():
    # Simulated Annealing to find a better set
    # We will try different sizes N
    best_overall_score = 0.0
    best_overall_set = [0]
    
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Try different set sizes
    for N in range(2, 15):
        # Initialize with a random set or a structured set
        # Start with a geometric progression or random small integers
        U = [0]
        # Add N-1 random distinct integers
        available = list(range(1, 100))
        random.shuffle(available)
        for i in range(N - 1):
            U.append(available[i])
        U = sorted(U)
        
        current_score = score_set(U)
        best_score = current_score
        best_set = U[:]
        
        T = 1.0
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            # Generate a neighbor
            U_neighbor = U[:]
            move_type = random.choice(['add', 'remove', 'swap', 'shift'])
            
            if move_type == 'add' and len(U_neighbor) < 20:
                # Add a random number
                new_val = random.randint(1, 200)
                if new_val not in U_neighbor:
                    U_neighbor.append(new_val)
            elif move_type == 'remove' and len(U_neighbor) > 2:
                # Remove a random non-zero element
                idx = random.randint(1, len(U_neighbor) - 1)
                U_neighbor.pop(idx)
            elif move_type == 'swap':
                if len(U_neighbor) > 1:
                    idx = random.randint(1, len(U_neighbor) - 1)
                    old_val = U_neighbor[idx]
                    new_val = random.randint(1, 200)
                    if new_val not in U_neighbor:
                        U_neighbor[idx] = new_val
            elif move_type == 'shift':
                if len(U_neighbor) > 1:
                    idx = random.randint(1, len(U_neighbor) - 1)
                    delta = random.choice([-1, 1]) * random.randint(1, 10)
                    new_val = U_neighbor[idx] + delta
                    if new_val > 0 and new_val not in U_neighbor:
                        U_neighbor[idx] = new_val
            else:
                continue
                
            U_neighbor = sorted(U_neighbor)
            if 0 not in U_neighbor:
                continue
                
            neighbor_score = score_set(U_neighbor)
            
            # Accept if better, or with probability exp((new - old)/T)
            if neighbor_score > current_score:
                current_score = neighbor_score
                U = U_neighbor
                if current_score > best_score:
                    best_score = current_score
                    best_set = U[:]
            else:
                delta = neighbor_score - current_score
                # Avoid log of negative or zero
                if delta < -100: 
                    accept_prob = 0.0
                else:
                    accept_prob = math.exp(delta / T)
                if random.random() < accept_prob:
                    current_score = neighbor_score
                    U = U_neighbor
                    
            T *= cooling
            if T < 0.001:
                break
                
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_set = best_set[:]
            
    return best_overall_set
