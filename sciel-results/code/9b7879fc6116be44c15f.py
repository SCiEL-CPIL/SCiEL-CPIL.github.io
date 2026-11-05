import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def calc_score(mask):
        s = 0
        m = mask
        for d in range(1, 256):
            if m & (m >> d) & (m >> (2 * d)):
                s += 1
        return s

    best_mask = 0
    best_score = -1

    def update_best(mask, score):
        nonlocal best_mask, best_score
        if score > best_score:
            best_score = score
            best_mask = mask

    def init_contiguous(start=1):
        mask = 0
        for i in range(32):
            val = start + i
            if val <= 512:
                mask |= (1 << val)
            else:
                break
        return mask

    def init_random():
        mask = 0
        elems = random.sample(range(1, 513), 32)
        for i in elems:
            mask |= (1 << i)
        return mask

    def init_perturbed_contiguous():
        mask = init_contiguous(random.randint(1, 50))
        bits = [i for i in range(1, 513) if (mask & (1 << i))]
        random.shuffle(bits)
        for _ in range(10):
            x = bits.pop()
            y = random.randint(1, 512)
            if not (mask & (1 << y)):
                mask &= ~(1 << x)
                mask |= (1 << y)
        return mask

    def init_dense_small():
        # Focus on small numbers to maximize small d
        mask = 0
        elems = random.sample(range(1, 100), 32)
        for i in elems:
            mask |= (1 << i)
        return mask

    initializers = [init_contiguous, init_perturbed_contiguous, init_dense_small, init_random]

    # Phase 1: Multiple short SA runs
    slice_time = 4.0
    num_slices = int(135 / slice_time)
    
    for k in range(num_slices):
        if time.time() > deadline - 15:
            break
            
        init_func = initializers[k % len(initializers)]
        current_mask = init_func()
        current_score = calc_score(current_mask)
        update_best(current_mask, current_score)
        
        T = 1.5
        cool_rate = 0.9998
        min_T = 0.01
        
        end_time = time.time() + slice_time
        stall_count = 0
        max_stall = 2000
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            x = random.randint(1, 512)
            y = random.randint(1, 512)
            
            if x == y:
                continue
                
            bx = 1 << x
            by = 1 << y
            
            if (current_mask & bx) and not (current_mask & by):
                new_mask = current_mask ^ bx ^ by
                new_score = calc_score(new_mask)
                delta = new_score - current_score
                
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        update_best(current_mask, current_score)
                        stall_count = 0
                    else:
                        stall_count += 1
                else:
                    stall_count += 1
                
                T *= cool_rate
                
                if stall_count > max_stall:
                    T = max(T, 1.0)
                    stall_count = 0

    # Phase 2: Final Polish
    if time.time() < deadline - 2:
        current_mask = best_mask
        current_score = best_score
        
        improved = True
        while time.time() < deadline - 1 and improved:
            improved = False
            for _ in range(50000):
                if time.time() > deadline - 1:
                    break
                x = random.randint(1, 512)
                y = random.randint(1, 512)
                if x == y:
                    continue
                bx = 1 << x
                by = 1 << y
                if (current_mask & bx) and not (current_mask & by):
                    new_mask = current_mask ^ bx ^ by
                    new_score = calc_score(new_mask)
                    if new_score > current_score:
                        current_mask = new_mask
                        current_score = new_score
                        update_best(current_mask, current_score)
                        improved = True
                        break

    result = []
    for i in range(1, 513):
        if best_mask & (1 << i):
            result.append(i)
    
    if len(result) != 32:
        return list(range(1, 33))
        
    return result
