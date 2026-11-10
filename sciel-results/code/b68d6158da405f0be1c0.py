import time
import random
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin for 90s limit
    
    N = 256
    # values[i] = i+2, so index 0 -> value 2, index 254 -> value 256
    n = N - 1  # 255 elements
    values = list(range(2, N + 1))
    
    # Precompute conflict bitmask: conflict_mask[i] has bit j set if lcm(values[i], values[j]) <= 256
    # lcm(a,b) = a*b//gcd(a,b) <= 256  <=>  a*b <= 256*gcd(a,b)
    conflict_mask = [0] * n
    for i in range(n):
        vi = values[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = values[j]
            g = gcd(vi, vj)
            # lcm = vi * vj // g
            if vi * vj <= 256 * g:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Weights
    weights = [1.0 / values[i] for i in range(n)]
    
    # Helper: check if vertex i can be added to set represented by bitmask
    def can_add(i, current_mask):
        # i must not be in current_mask, and must not conflict with any in current_mask
        if (current_mask >> i) & 1:
            return False
        # Check conflicts: for each j in current_mask, conflict_mask[i] has bit j
        # Equivalent: conflict_mask[i] & current_mask == 0
        return (conflict_mask[i] & current_mask) == 0
    
    def score_mask(mask):
        s = 0.0
        m = mask
        while m:
            lb = m & (-m)
            i = lb.bit_length() - 1
            s += weights[i]
            m ^= lb
        return s
    
    # Generate initial solution: greedy by weight (ascending value)
    def greedy_initial():
        mask = 0
        for i in range(n):  # ascending value = descending weight
            if can_add(i, mask):
                mask |= (1 << i)
        return mask
    
    # Simulated Annealing
    def simulate_annealing(initial_mask, time_budget):
        current_mask = initial_mask
        current_score = score_mask(current_mask)
        best_mask = current_mask
        best_score = current_score
        
        # Temperature schedule: start high, cool down
        # Max possible score is sum of all 1/i for i=2..256 ≈ H(256)-1 ≈ 5.25
        # Typical scores ~1.1, so delta per move is small
        T = 0.05  # initial temperature (weight units)
        T_min = 1e-6
        # Number of iterations to fit in time_budget
        # We'll estimate: each iteration is very fast with bitmasks
        
        # Precompute list of all indices for random selection
        all_indices = list(range(n))
        
        end_time = time.time() + time_budget
        iteration = 0
        
        while time.time() < end_time:
            iteration += 1
            
            # Pick a random 2-exchange move: remove up to 2, add up to 2
            # To keep it simple and fast, do: pick 1-2 to remove, 1-2 to add
            # For efficiency, let's do 1-remove-1-add or 2-remove-2-add or 1-add or 1-remove
            
            move_type = random.random()
            
            if move_type < 0.3:
                # 1-add
                i = random.choice(all_indices)
                if can_add(i, current_mask):
                    new_mask = current_mask | (1 << i)
                    new_score = current_score + weights[i]
                    delta = new_score - current_score
                    if delta > 0 or random.random() < min(1.0, delta / T if T > 0 else 0.0):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
            elif move_type < 0.6:
                # 1-remove
                # Pick a random element from current set
                m = current_mask
                if m == 0:
                    continue
                # Random selection from set bits
                # Use random bit position
                k = random.randint(0, current_mask.bit_length() - 1)
                if (current_mask >> k) & 1:
                    i = k
                else:
                    # Find a set bit
                    tmp = current_mask
                    while tmp:
                        lb = tmp & (-tmp)
                        idx = lb.bit_length() - 1
                        if random.random() < 0.5:
                            i = idx
                            break
                        tmp ^= lb
                    else:
                        continue
                new_mask = current_mask & ~(1 << i)
                new_score = current_score - weights[i]
                delta = new_score - current_score
                if delta > 0 or random.random() < min(1.0, delta / T if T > 0 else 0.0):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            elif move_type < 0.85:
                # 1-swap: remove one, add one
                # Pick random i to remove
                m = current_mask
                if m == 0:
                    continue
                # Get list of set bits (for random pick)
                # For speed, just pick random index and check
                i = random.choice(all_indices)
                if not ((current_mask >> i) & 1):
                    continue
                # Pick random j to add
                j = random.choice(all_indices)
                if (current_mask >> j) & 1:
                    continue
                # Check if j can be added to current_mask without i
                temp_mask = current_mask & ~(1 << i)
                if (conflict_mask[j] & temp_mask) == 0:
                    new_mask = temp_mask | (1 << j)
                    new_score = current_score - weights[i] + weights[j]
                    delta = new_score - current_score
                    if delta > 0 or random.random() < min(1.0, delta / T if T > 0 else 0.0):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
            else:
                # 2-swap: remove two, add two
                # Pick two distinct elements to remove
                m = current_mask
                if m == 0 or (m & (m - 1)) == 0:
                    continue  # need at least 2 elements
                
                # Pick i1, i2 from current set
                # Get set bits
                set_bits = []
                tmp = m
                while tmp:
                    lb = tmp & (-tmp)
                    idx = lb.bit_length() - 1
                    set_bits.append(idx)
                    tmp ^= lb
                
                if len(set_bits) < 2:
                    continue
                
                i1, i2 = random.sample(set_bits, 2)
                temp_mask = current_mask & ~(1 << i1) & ~(1 << i2)
                
                # Pick j1, j2 not in current set
                # Build list of available (not in current_mask)
                # For speed, just try random
                j1 = random.choice(all_indices)
                if (current_mask >> j1) & 1 or j1 == i1 or j1 == i2:
                    continue
                j2 = random.choice(all_indices)
                if (current_mask >> j2) & 1 or j2 == i1 or j2 == i2 or j2 == j1:
                    continue
                
                # Check j1, j2 compatible with temp_mask and with each other
                if (conflict_mask[j1] & temp_mask) != 0:
                    continue
                if (conflict_mask[j2] & temp_mask) != 0:
                    continue
                if (conflict_mask[j1] & (1 << j2)) or (conflict_mask[j2] & (1 << j1)):
                    continue
                
                new_mask = temp_mask | (1 << j1) | (1 << j2)
                new_score = current_score - weights[i1] - weights[i2] + weights[j1] + weights[j2]
                delta = new_score - current_score
                if delta > 0 or random.random() < min(1.0, delta / T if T > 0 else 0.0):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            
            # Cool down
            # Use geometric cooling with occasional reheating
            if iteration % 1000 == 0:
                T *= 0.95
            if T < T_min:
                T = T_min
            
            # Periodic reheating to escape deep local optima
            if iteration % 10000 == 0 and T < 0.01:
                T = 0.02
        
        return best_mask, best_score
    
    # Main: run multiple SA restarts with different initial solutions
    best_overall_mask = 0
    best_overall_score = 0.0
    
    # Initial solution 1: greedy ascending
    init1 = greedy_initial()
    
    # Initial solution 2: greedy descending
    init2 = 0
    for i in range(n - 1, -1, -1):
        if can_add(i, init2):
            init2 |= (1 << i)
    
    # Initial solution 3: greedy by conflict degree
    conflict_deg = [bin(conflict_mask[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda i: (conflict_deg[i], -weights[i]))
    init3 = 0
    for i in order:
        if can_add(i, init3):
            init3 |= (1 << i)
    
    initial_solutions = [init1, init2, init3]
    
    # Add a few random initial solutions
    for _ in range(3):
        mask = 0
        shuffled = list(range(n))
        random.shuffle(shuffled)
        for i in shuffled:
            if can_add(i, mask):
                mask |= (1 << i)
        initial_solutions.append(mask)
    
    # Track elapsed time
    elapsed = time.time() - start
    remaining = deadline - time.time()
    
    if remaining <= 0:
        # Fallback: return best initial
        best_mask = max(initial_solutions, key=score_mask)
        result = [values[i] for i in range(n) if (best_mask >> i) & 1]
        if not result:
            result = [2]
        return result
    
    # Allocate time to each restart
    n_restarts = len(initial_solutions)
    # Give more time to earlier restarts? Or equal? Equal is fine.
    time_per_restart = remaining * 0.8 / n_restarts  # 80% for SA, 20% for final polish
    
    for idx, init_mask in enumerate(initial_solutions):
        if time.time() >= deadline - 2.0:
            break
        
        t_budget = min(time_per_restart, deadline - time.time() - 1.0)
        if t_budget < 0.1:
            break
        
        sa_mask, sa_score = simulate_annealing(init_mask, t_budget)
        
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall_mask = sa_mask
    
    # Final polish: run SA on the best mask with remaining time
    remaining = deadline - time.time()
    if remaining > 0.5 and best_overall_mask != 0:
        sa_mask, sa_score = simulate_annealing(best_overall_mask, remaining * 0.9)
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall_mask = sa_mask
    
    # Extract result
    result = [values[i] for i in range(n) if (best_overall_mask >> i) & 1]
    
    # Ensure nonempty
    if not result:
        # Find any valid single element (all single elements are valid since no pairs)
        result = [2]
    
    # Final time check
    if time.time() > deadline:
        # Shouldn't happen due to guards, but return what we have
        pass
    
    return result
