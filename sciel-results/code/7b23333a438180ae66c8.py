import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    time_limit = 88.0  # Safety margin
    
    # Precompute reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i

    def get_set_bits(mask):
        bits = []
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            bits.append(x)
            m ^= lsb
        return bits

    def can_add(x, mask):
        if mask & (1 << x):
            return False
        
        # Case 1: x is middle. a + c = 2x => c = 2x - a.
        m = mask
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            c = 2 * x - a
            if 1 <= c <= N:
                # c != a is guaranteed because x not in mask implies x != a
                # Wait, if c == a, then 2x = 2a => x=a. But x not in mask, a in mask. So x!=a.
                if mask & (1 << c):
                    return False
            
        # Case 2: x is end. a + x = 2b => a = 2b - x.
        m = mask
        while m:
            lsb = m & -m
            b = lsb.bit_length() - 1
            m ^= lsb
            a = 2 * b - x
            if 1 <= a <= N:
                # a != b and a != x guaranteed as above
                if mask & (1 << a):
                    return False
        
        return True

    def score_of_mask(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            s += recip[x]
            m ^= lsb
        return s

    # 1. Greedy Initialization
    current_mask = 0
    for x in range(1, N + 1):
        if time.time() - start_time > time_limit:
            break
        if can_add(x, current_mask):
            current_mask |= (1 << x)

    current_score = score_of_mask(current_mask)
    best_mask = current_mask
    best_score = current_score

    all_numbers = list(range(1, N + 1))
    
    # 2. Simulated Annealing with Restarts
    # We will run multiple rounds of SA.
    # Each round: Cool from T_high to T_low.
    # If no improvement for a while, restart from best with perturbation.
    
    T_high = 0.5
    T_low = 0.001
    cooling = 0.995
    max_iter_per_round = 50000
    
    rounds = 0
    while time.time() - start_time < time_limit:
        rounds += 1
        T = T_high
        current_mask = best_mask
        current_score = best_score
        
        # Perturb if not the first round
        if rounds > 1:
            # Remove 5-10 random elements from best_mask
            set_bits = get_set_bits(best_mask)
            if len(set_bits) > 10:
                num_remove = random.randint(5, 10)
                to_remove = random.sample(set_bits, num_remove)
                for x in to_remove:
                    current_mask &= ~(1 << x)
                current_score = score_of_mask(current_mask)
        
        no_improve_count = 0
        iter_count = 0
        
        while T > T_low and time.time() - start_time < time_limit:
            iter_count += 1
            
            # Choose move type
            # Bias towards swap and add to maintain size
            move_type = random.choice(['add', 'remove', 'swap', 'swap', 'swap'])
            new_mask = current_mask
            valid = True
            delta = 0.0
            
            if move_type == 'add':
                # Find candidates not in mask
                # Optimization: Use a list of not-in-mask? 
                # For N=256, iterating is fine.
                candidates = [x for x in all_numbers if not (current_mask & (1 << x))]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if can_add(x, current_mask):
                    new_mask |= (1 << x)
                    delta = recip[x]
                else:
                    continue

            elif move_type == 'remove':
                set_bits = get_set_bits(current_mask)
                if not set_bits:
                    continue
                x = random.choice(set_bits)
                new_mask &= ~(1 << x)
                delta = -recip[x]

            elif move_type == 'swap':
                set_bits = get_set_bits(current_mask)
                if not set_bits:
                    continue
                x = random.choice(set_bits)
                
                candidates = [y for y in all_numbers if not (current_mask & (1 << y))]
                if not candidates:
                    continue
                y = random.choice(candidates)
                
                temp_mask = current_mask & ~(1 << x)
                if can_add(y, temp_mask):
                    new_mask = temp_mask | (1 << y)
                    delta = recip[y] - recip[x]
                else:
                    continue

            new_score = current_score + delta
            
            # Metropolis Acceptance
            if delta > 0 or random.random() < math.exp(delta / T):
                current_mask = new_mask
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                # Optional: Break if stuck for a very long time at low T
                if no_improve_count > 10000:
                    break

            T *= cooling
            
            # Check time frequently
            if iter_count % 1000 == 0:
                if time.time() - start_time > time_limit - 2:
                    break

    # Extract best set
    result = get_set_bits(best_mask)
    return result

