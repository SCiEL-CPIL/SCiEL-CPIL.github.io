import time
import math
import random

def construct():
    deadline = time.time() + 55  # 55s safety margin
    N = 256
    
    # Precompute reciprocals
    recips = [0.0] + [1.0 / i for i in range(1, N + 1)]
    
    # Helper: check if adding x to set represented by mask creates a 3-AP
    # mask: integer where bit i is set if (i+1) is in the set
    # x: 1-indexed integer
    def creates_ap(mask, x):
        # x is 1-indexed
        # 1. x is middle: a + b = 2x. a, b in S.
        two_x = 2 * x
        m_iter = mask
        while m_iter:
            lsb = m_iter & -m_iter
            a = lsb.bit_length()  # 1-indexed value
            b = two_x - a
            if b != a and (mask >> (b - 1)) & 1:
                return True
            m_iter ^= lsb
            
        # 2. x is end: a + x = 2b => b = (a+x)/2
        m_iter = mask
        while m_iter:
            lsb = m_iter & -m_iter
            a = lsb.bit_length()
            s = a + x
            if s % 2 == 0:
                b = s // 2
                # b must be in S. 
                # Note: if b==a or b==x, it implies a==x or a==x which is false since x not in mask.
                if (mask >> (b - 1)) & 1:
                    return True
            m_iter ^= lsb
            
        return False

    # Helper: get list of elements in mask
    def get_elements(mask):
        elems = []
        m_iter = mask
        while m_iter:
            lsb = m_iter & -m_iter
            elems.append(lsb.bit_length())
            m_iter ^= lsb
        return elems

    # Greedy initialization
    current_mask = 0
    current_weight = 0.0
    for i in range(1, N + 1):
        # Try adding i
        if not creates_ap(current_mask, i):
            current_mask |= (1 << (i - 1))
            current_weight += recips[i]
            
    best_mask = current_mask
    best_weight = current_weight
    
    # Simulated Annealing
    T = 0.005
    cooling = 0.999
    steps = 0
    
    while time.time() < deadline:
        steps += 1
        
        # Choose a move type: Add, Remove, Swap
        # Probabilities: 0.2 Add, 0.2 Remove, 0.6 Swap
        r = random.random()
        
        if r < 0.2:
            # Add
            # Pick a random candidate not in set
            candidates = [i for i in range(1, N + 1) if not (current_mask >> (i - 1)) & 1]
            if not candidates:
                continue
            x = random.choice(candidates)
            
            if not creates_ap(current_mask, x):
                new_mask = current_mask | (1 << (x - 1))
                new_weight = current_weight + recips[x]
                # Always accept if valid (weight increases)
                current_mask = new_mask
                current_weight = new_weight
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_mask = current_mask
        elif r < 0.4:
            # Remove
            elems = get_elements(current_mask)
            if not elems:
                continue
            x = random.choice(elems)
            
            # Removing always keeps validity
            new_mask = current_mask & ~(1 << (x - 1))
            new_weight = current_weight - recips[x]
            
            # Accept with probability exp(delta_w / T)
            # delta_w is negative
            delta_w = new_weight - current_weight
            if random.random() < math.exp(delta_w / T):
                current_mask = new_mask
                current_weight = new_weight
        else:
            # Swap
            elems_in = get_elements(current_mask)
            elems_out = [i for i in range(1, N + 1) if not (current_mask >> (i - 1)) & 1]
            
            if not elems_in or not elems_out:
                continue
                
            u = random.choice(elems_in)
            v = random.choice(elems_out)
            
            # Remove u, Add v
            # Check if v creates AP with (S \ {u})
            mask_without_u = current_mask & ~(1 << (u - 1))
            
            if not creates_ap(mask_without_u, v):
                new_mask = mask_without_u | (1 << (v - 1))
                new_weight = current_weight - recips[u] + recips[v]
                delta_w = new_weight - current_weight
                
                if delta_w > 0:
                    # Always accept
                    current_mask = new_mask
                    current_weight = new_weight
                    if current_weight > best_weight:
                        best_weight = current_weight
                        best_mask = current_mask
                else:
                    # Accept with prob exp(delta_w / T)
                    if random.random() < math.exp(delta_w / T):
                        current_mask = new_mask
                        current_weight = new_weight
        
        # Cool down
        T *= cooling
        if T < 0.0001:
            # Reheat slightly or reset? 
            # Let's just keep it low, it will find local optima.
            # Maybe restart from best if stuck?
            # For now, just continue.
            pass

    # Return the best set found
    return get_elements(best_mask)

