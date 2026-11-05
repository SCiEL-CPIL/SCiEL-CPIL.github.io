import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    N = 512
    K = 32
    
    def get_score(m):
        score = 0
        # d can be at most 255 since 2*d < 512
        for d in range(1, 256):
            if (m & (m >> d) & (m >> (2*d))) != 0:
                score += 1
        return score

    def random_mask():
        indices = random.sample(range(N), K)
        mask = 0
        for i in indices:
            mask |= (1 << i)
        return mask

    def sa_search(init_mask, time_limit):
        # Build initial lists
        set_bits = []
        unset_bits = []
        for i in range(N):
            if (init_mask >> i) & 1:
                set_bits.append(i)
            else:
                unset_bits.append(i)
        
        current_mask = init_mask
        current_score = get_score(current_mask)
        best_mask = current_mask
        best_score = current_score
        
        T = 2.0
        T_min = 0.01
        alpha = 0.9995
        
        # Local variables for speed
        rand_int = random.randint
        rand_float = random.random
        exp = math.exp
        get_score_local = get_score
        N_local = N
        
        while time.time() < time_limit:
            # Pick random index from set_bits and unset_bits
            idx_i = rand_int(0, len(set_bits) - 1)
            idx_j = rand_int(0, len(unset_bits) - 1)
            
            i = set_bits[idx_i]
            j = unset_bits[idx_j]
            
            new_mask = current_mask ^ (1 << i) ^ (1 << j)
            new_score = get_score_local(new_mask)
            
            delta = new_score - current_score
            
            accept = False
            if delta > 0:
                accept = True
            else:
                if T > 0:
                    # delta is negative or zero
                    # exp(delta / T)
                    # If delta is 0, exp(0)=1, always accept
                    # If delta < 0, prob < 1
                    if rand_float() < exp(delta / T):
                        accept = True
            
            if accept:
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                
                # Update lists
                set_bits[idx_i] = j
                unset_bits[idx_j] = i
            
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min

        return best_mask, best_score

    # Run multiple SA restarts
    # 5 restarts
    num_restarts = 5
    
    global_best_mask = None
    global_best_score = -1
    
    # Calculate time per restart
    # Leave 5s for finalization
    available_time = deadline - time.time() - 5.0
    time_per_restart = available_time / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        init_mask = random_mask()
        
        # Adjust time limit for the last restart to ensure we stop before deadline
        if r == num_restarts - 1:
            t_limit = deadline - 2.0
        else:
            t_limit = time.time() + time_per_restart
        
        best_mask, best_score = sa_search(init_mask, t_limit)
        
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_mask = best_mask

    # Convert best mask to result list
    result = []
    if global_best_mask:
        for i in range(N):
            if (global_best_mask >> i) & 1:
                result.append(i + 1)
    else:
        # Fallback
        result = list(range(1, K + 1))
        
    return result
