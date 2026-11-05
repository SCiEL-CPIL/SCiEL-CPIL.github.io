import time
import random

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

    best_mask = 0
    best_score = -1
    for m in candidates:
        s = get_score(m)
        if s > best_score:
            best_score = s
            best_mask = m

    # Multi-restart Simulated Annealing
    # We will run SA for a fixed duration or steps, then restart if stuck.
    
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
            idx_i = random.randint(0, len(set_bits) - 1)
            idx_j = random.randint(0, len(unset_bits) - 1)
            
            i = set_bits[idx_i]
            j = unset_bits[idx_j]
            
            new_mask = current_mask ^ (1 << i) ^ (1 << j)
            
            new_score = get_score(new_mask)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
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

    # Convert best_mask to list of integers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
