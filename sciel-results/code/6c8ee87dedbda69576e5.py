import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    N = 512
    K = 32
    
    def get_score(m):
        score = 0
        for d in range(1, 256):
            if (m & (m >> d) & (m >> (2*d))) != 0:
                score += 1
        return score

    def get_set_bits(m):
        return [i for i in range(N) if (m >> i) & 1]
    
    def get_unset_bits(m):
        return [i for i in range(N) if not ((m >> i) & 1)]
    
    def greedy_coverage_improve(mask, iterations=100):
        """Maximum Coverage greedy: swap elements that minimize loss and maximize new differences."""
        current_score = get_score(mask)
        set_bits = get_set_bits(mask)
        unset_bits = get_unset_bits(mask)
        
        for _ in range(iterations):
            best_swap = None
            best_delta = 0
            best_new_mask = mask
            
            # Sample a subset of swaps for speed
            sample_size = min(200, len(set_bits) * len(unset_bits))
            sampled_swaps = []
            for _ in range(sample_size):
                i = random.choice(set_bits)
                j = random.choice(unset_bits)
                sampled_swaps.append((i, j))
            
            for i, j in sampled_swaps:
                new_mask = mask ^ (1 << i) ^ (1 << j)
                new_score = get_score(new_mask)
                delta = new_score - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (i, j)
                    best_new_mask = new_mask
            
            if best_swap is not None:
                mask = best_new_mask
                current_score += best_delta
                # Update set_bits and unset_bits
                i, j = best_swap
                set_bits.remove(i)
                set_bits.append(j)
                unset_bits.remove(j)
                unset_bits.append(i)
            else:
                break
        
        return mask, current_score

    # Initializations
    candidates = []
    
    # 1. Random
    indices = random.sample(range(N), K)
    mask = 0
    for i in indices:
        mask |= (1 << i)
    candidates.append(mask)
    
    # 2. Contiguous Block
    mask = 0
    for i in range(K):
        mask |= (1 << i)
    candidates.append(mask)
    
    # 3. Another Random
    indices = random.sample(range(N), K)
    mask = 0
    for i in indices:
        mask |= (1 << i)
    candidates.append(mask)
    
    # 4. Two Blocks
    mask = 0
    for i in range(16):
        mask |= (1 << i)
    for i in range(30, 46):
        mask |= (1 << i)
    candidates.append(mask)
    
    # 5. Three-cluster initialization
    mask = 0
    for i in range(11):
        mask |= (1 << i)
    for i in range(170, 181):
        mask |= (1 << i)
    for i in range(340, 351):
        mask |= (1 << i)
    candidates.append(mask)
    
    # 6. Geometric spread
    mask = 0
    for i in range(K):
        pos = int(i * (N - 1) / (K - 1))
        mask |= (1 << pos)
    candidates.append(mask)
    
    # 7. Arithmetic progression with small step
    mask = 0
    for i in range(K):
        mask |= (1 << (i * 16))
    candidates.append(mask)

    best_mask = 0
    best_score = -1
    for m in candidates:
        s = get_score(m)
        if s > best_score:
            best_score = s
            best_mask = m

    # Multi-restart Simulated Annealing with 1-swap and 2-swap moves
    
    while time.time() < deadline:
        current_mask = best_mask
        current_score = best_score
        
        set_bits = get_set_bits(current_mask)
        unset_bits = get_unset_bits(current_mask)
        
        T = 5.0
        cooling_rate = 0.999
        min_T = 0.01
        
        steps = 0
        max_steps = 50000
        last_improvement_step = 0
        local_best_score = current_score
        local_best_mask = current_mask
        
        while time.time() < deadline and T > min_T and steps < max_steps:
            # Choose between 1-swap and 2-swap moves
            if random.random() < 0.3:
                # 2-swap move: swap two elements
                if len(set_bits) >= 2 and len(unset_bits) >= 2:
                    idx_i1 = random.randint(0, len(set_bits) - 1)
                    idx_i2 = random.randint(0, len(set_bits) - 1)
                    idx_j1 = random.randint(0, len(unset_bits) - 1)
                    idx_j2 = random.randint(0, len(unset_bits) - 1)
                    
                    if idx_i1 == idx_i2 or idx_j1 == idx_j2:
                        steps += 1
                        continue
                    
                    i1 = set_bits[idx_i1]
                    i2 = set_bits[idx_i2]
                    j1 = unset_bits[idx_j1]
                    j2 = unset_bits[idx_j2]
                    
                    new_mask = current_mask ^ (1 << i1) ^ (1 << i2) ^ (1 << j1) ^ (1 << j2)
                    new_score = get_score(new_mask)
                    delta = new_score - current_score
                    
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        
                        # Update lists
                        set_bits[idx_i1] = j1
                        set_bits[idx_i2] = j2
                        unset_bits[idx_j1] = i1
                        unset_bits[idx_j2] = i2
                        
                        if current_score > local_best_score:
                            local_best_score = current_score
                            local_best_mask = current_mask
                            last_improvement_step = steps
                else:
                    steps += 1
                    continue
            else:
                # 1-swap move
                idx_i = random.randint(0, len(set_bits) - 1)
                idx_j = random.randint(0, len(unset_bits) - 1)
                
                i = set_bits[idx_i]
                j = unset_bits[idx_j]
                
                new_mask = current_mask ^ (1 << i) ^ (1 << j)
                
                new_score = get_score(new_mask)
                
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    
                    set_bits[idx_i] = j
                    unset_bits[idx_j] = i
                    
                    if current_score > local_best_score:
                        local_best_score = current_score
                        local_best_mask = current_mask
                        last_improvement_step = steps
            
            T *= cooling_rate
            steps += 1
            
            # Check for stagnation
            if steps - last_improvement_step > 5000:
                # Restart from local best with higher T
                break

        if local_best_score > best_score:
            best_score = local_best_score
            best_mask = local_best_mask
    
    # Final polish pass with greedy coverage
    if time.time() < deadline - 5.0:
        best_mask, best_score = greedy_coverage_improve(best_mask, iterations=200)
        
        # Another round of SA from the polished solution
        current_mask = best_mask
        current_score = best_score
        set_bits = get_set_bits(current_mask)
        unset_bits = get_unset_bits(current_mask)
        
        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.005
        steps = 0
        max_steps = 30000
        last_improvement_step = 0
        local_best_score = current_score
        local_best_mask = current_mask
        
        while time.time() < deadline - 2.0 and T > min_T and steps < max_steps:
            idx_i = random.randint(0, len(set_bits) - 1)
            idx_j = random.randint(0, len(unset_bits) - 1)
            
            i = set_bits[idx_i]
            j = unset_bits[idx_j]
            
            new_mask = current_mask ^ (1 << i) ^ (1 << j)
            new_score = get_score(new_mask)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                set_bits[idx_i] = j
                unset_bits[idx_j] = i
                
                if current_score > local_best_score:
                    local_best_score = current_score
                    local_best_mask = current_mask
                    last_improvement_step = steps
            
            T *= cooling_rate
            steps += 1
            
            if steps - last_improvement_step > 3000:
                break
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_mask = local_best_mask

    # Convert best_mask to list of integers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
