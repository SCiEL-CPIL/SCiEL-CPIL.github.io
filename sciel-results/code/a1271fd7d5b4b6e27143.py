"""Combined GA + simulated annealing for 4-AP-free 3-AP-maximizing subset of [1,256]."""
import time
import random

def construct():
    N = 256
    deadline = time.time() + 55.0  # Safety margin before 90s kill (or 60s if tighter)
    
    # 1. Precompute 4-AP conflicts
    conflicts = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            conflicts[a].append((b, c, e))
            conflicts[b].append((a, c, e))
            conflicts[c].append((a, b, e))
            conflicts[e].append((a, b, c))

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for (y, z, w) in conflicts[x]:
            if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
                return False
        return True

    def count_3aps(mask):
        """Count 3-term APs in the set represented by mask."""
        count = 0
        for i in range(N):
            if not ((mask >> i) & 1):
                continue
            max_d = (N - 1 - i) // 2
            for d in range(1, max_d + 1):
                j = i + d
                k = i + 2 * d
                if ((mask >> j) & 1) and ((mask >> k) & 1):
                    count += 1
        return count

    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        # Case 1: x is the start: x, x+d, x+2d
        max_d = (N - 1 - x) // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1):
                delta += 1
        # Case 2: x is the middle: x-d, x, x+d
        max_d = min(x, N - 1 - x)
        for d in range(1, max_d + 1):
            if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
                delta += 1
        # Case 3: x is the end: x-2d, x-d, x
        max_d = x // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1):
                delta += 1
        return delta

    def delta_3aps_remove(mask, x):
        """Calculate decrease in 3-AP count if x is removed from mask."""
        return delta_3aps_add(mask, x)

    # 2. Greedy Initialization
    def greedy_init():
        mask = 0
        elems = list(range(N))
        random.shuffle(elems)
        
        for x in elems:
            if is_valid_add(mask, x):
                if delta_3aps_add(mask, x) > 0 or random.random() < 0.3:
                    mask |= (1 << x)
        return mask

    # 3. Simulated Annealing + GA hybrid
    best_mask = 0
    best_score = -1
    
    # Initialize population
    pop_size = 20
    population = []
    for _ in range(pop_size):
        m = greedy_init()
        s = count_3aps(m)
        population.append((m, s))
        if s > best_score:
            best_score = s
            best_mask = m

    # Simulated Annealing parameters
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline:
        # Select a random individual to mutate
        idx = random.randint(0, pop_size - 1)
        current_mask, current_score = population[idx]
        
        # Perform a move
        move_type = random.random()
        new_mask = current_mask
        new_score = current_score
        valid_move = False
        
        if move_type < 0.5:
            # Add
            x = random.randint(0, N - 1)
            if not ((current_mask >> x) & 1):
                if is_valid_add(current_mask, x):
                    new_mask = current_mask | (1 << x)
                    new_score = current_score + delta_3aps_add(current_mask, x)
                    valid_move = True
        elif move_type < 0.8:
            # Remove
            x = random.randint(0, N - 1)
            if (current_mask >> x) & 1:
                new_mask = current_mask & ~(1 << x)
                new_score = current_score - delta_3aps_remove(current_mask, x)
                valid_move = True
        else:
            # Swap
            x = random.randint(0, N - 1)
            if (current_mask >> x) & 1:
                y = random.randint(0, N - 1)
                if not ((current_mask >> y) & 1):
                    temp_mask = current_mask & ~(1 << x)
                    if is_valid_add(temp_mask, y):
                        d_rem = delta_3aps_remove(current_mask, x)
                        d_add = delta_3aps_add(temp_mask, y)
                        new_score = current_score - d_rem + d_add
                        new_mask = temp_mask | (1 << y)
                        valid_move = True

        if valid_move:
            # Acceptance
            if new_score >= current_score:
                population[idx] = (new_mask, new_score)
                if new_score > best_score:
                    best_score = new_score
                    best_mask = new_mask
            else:
                if random.random() < pow(2.718281828, (new_score - current_score) / T):
                    population[idx] = (new_mask, new_score)
            
            # Crossover with best individual occasionally
            if random.random() < 0.1:
                # Uniform crossover between current and best
                parent1, _ = population[idx]
                parent2 = best_mask
                child_mask = 0
                for i in range(N):
                    if random.random() < 0.5:
                        if (parent1 >> i) & 1:
                            child_mask |= (1 << i)
                    else:
                        if (parent2 >> i) & 1:
                            child_mask |= (1 << i)
                
                # Ensure validity: remove any elements that create 4-APs
                valid_child = 0
                for i in range(N):
                    if (child_mask >> i) & 1:
                        if is_valid_add(valid_child, i):
                            valid_child |= (1 << i)
                
                child_score = count_3aps(valid_child)
                # Replace worst individual if child is better
                worst_idx = 0
                worst_score = population[0][1]
                for j in range(1, pop_size):
                    if population[j][1] < worst_score:
                        worst_score = population[j][1]
                        worst_idx = j
                
                if child_score > worst_score:
                    population[worst_idx] = (valid_child, child_score)
                    if child_score > best_score:
                        best_score = child_score
                        best_mask = valid_child
        
        steps += 1
        T = max(T_min, T * cooling)
        
        # Occasionally re-seed worst individual
        if steps % 1000 == 0:
            worst_idx = 0
            worst_score = population[0][1]
            for j in range(1, pop_size):
                if population[j][1] < worst_score:
                    worst_score = population[j][1]
                    worst_idx = j
            
            if worst_score < best_score * 0.8:  # If worst is significantly worse
                m = greedy_init()
                s = count_3aps(m)
                population[worst_idx] = (m, s)
                if s > best_score:
                    best_score = s
                    best_mask = m

    # Convert best mask to result
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)
