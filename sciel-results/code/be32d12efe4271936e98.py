import random
import time
import math

def construct():
    start_time = time.time()
    # Safety margin: 145 seconds
    deadline = start_time + 145.0 

    def calc_score(mask):
        """Count distinct d > 0 where a, a+d, a+2d all in set (bitmask)."""
        s = 0
        m = mask
        # Max possible difference is 512-1 = 511, but a+2d <= 512 => 2d <= 511 => d <= 255
        for d in range(1, 256):
            if m & (m >> d) & (m >> (2 * d)):
                s += 1
        return s

    def get_elements(mask):
        """Convert bitmask to list of elements."""
        elems = []
        # We can optimize this by iterating bits
        m = mask
        i = 1
        while m:
            if m & 1:
                elems.append(i)
            m >>= 1
            i += 1
        return elems

    def count_bits(mask):
        """Count set bits."""
        c = 0
        m = mask
        while m:
            c += 1
            m &= m - 1
        return c

    def random_mask():
        """Generate a random 32-element bitmask."""
        elems = random.sample(range(1, 513), 32)
        mask = 0
        for e in elems:
            mask |= (1 << e)
        return mask

    def structured_mask(type_id):
        """Generate a structured initial mask."""
        mask = 0
        if type_id == 0:
            # Contiguous
            for i in range(1, 33):
                mask |= (1 << i)
        elif type_id == 1:
            # Sparse AP
            for i in range(32):
                val = 1 + i * 15
                if val <= 512:
                    mask |= (1 << val)
        elif type_id == 2:
            # Two interleaved APs
            for i in range(16):
                mask |= (1 << (1 + i * 32))
                mask |= (1 << (2 + i * 32))
        elif type_id == 3:
            # Random but sorted
            elems = sorted(random.sample(range(1, 513), 32))
            for e in elems:
                mask |= (1 << e)
        elif type_id == 4:
            # Blocky
            # 4 blocks of 8
            for b in range(4):
                start = 1 + b * 128
                for i in range(8):
                    mask |= (1 << (start + i))
        elif type_id == 5:
            # Geometric-ish
            vals = set()
            base = 1
            while len(vals) < 32:
                vals.add(base)
                base = int(base * 1.1) + 1
                if base > 512:
                    break
            if len(vals) < 32:
                # Fill with random
                remaining = 32 - len(vals)
                existing = list(vals)
                pool = [x for x in range(1, 513) if x not in vals]
                vals.update(random.sample(pool, remaining))
            for v in vals:
                if v <= 512:
                    mask |= (1 << v)
        return mask

    best_mask = 0
    best_score = -1

    # Generate diverse initializations
    inits = []
    inits.append(structured_mask(0))
    inits.append(structured_mask(1))
    inits.append(structured_mask(2))
    inits.append(structured_mask(3))
    inits.append(structured_mask(4))
    inits.append(structured_mask(5))
    # Add a few random ones
    for _ in range(4):
        inits.append(random_mask())

    # Simulated Annealing parameters
    # We will run SA for a fixed number of iterations or time per restart
    # Total time budget is ~145s. 10 restarts => ~14s each.
    num_restarts = len(inits)
    time_per_restart = (deadline - start_time) / max(num_restarts, 1)
    
    # To be safe, let's use a fixed number of iterations per restart and check time
    # Or better, run until time limit for each restart.
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break

        current_mask = inits[r]
        current_score = calc_score(current_mask)

        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask

        # SA Parameters
        T = 1.0
        cool_rate = 0.9998  # Slow cooling
        min_T = 0.001
        restart_deadline = start_time + (r + 1) * time_per_restart
        
        # If a restart takes too long, we skip to next
        # But we want to use all time. So we just run until global deadline or local quota.
        
        no_improve = 0
        max_no_improve = 5000
        
        # Pre-calculate set and unset lists for faster swap generation?
        # Updating them on the fly is expensive. 
        # Instead, pick random x, y. If x in set and y not, swap.
        
        while time.time() < restart_deadline - 0.5:
            if T < min_T:
                break

            # Generate move: swap a random element with a random non-element
            # Optimization: Pick x from 1..512, y from 1..512
            x = random.randint(1, 512)
            y = random.randint(1, 512)

            if x == y:
                continue

            bx = 1 << x
            by = 1 << y

            # Check if x is in set and y is not
            # (current_mask & bx) and not (current_mask & by)
            if (current_mask & bx) and not (current_mask & by):
                new_mask = current_mask ^ bx ^ by
                new_score = calc_score(new_mask)
                delta = new_score - current_score

                if delta >= 0:
                    accept = True
                else:
                    # Avoid underflow in exp
                    if delta / T < -50:
                        accept = False
                    else:
                        accept = random.random() < math.exp(delta / T)

                if accept:
                    current_mask = new_mask
                    current_score = new_score
                    no_improve = 0

                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    no_improve += 1

                T *= cool_rate

                if no_improve > max_no_improve:
                    # Kick: random perturbation
                    # Swap 2 elements
                    if random.random() < 0.5:
                        # Try to find a valid 2-swap
                        # Pick 2 random set elements and 2 random unset elements
                        set_elems = []
                        unset_elems = []
                        # This is slow if done every time. 
                        # Alternative: just do 1-swap kicks (random walk)
                        for _ in range(10):
                            x1 = random.randint(1, 512)
                            x2 = random.randint(1, 512)
                            y1 = random.randint(1, 512)
                            y2 = random.randint(1, 512)
                            if x1 == x2 or y1 == y2 or x1 == y1 or x1 == y2 or x2 == y1 or x2 == y2:
                                continue
                            bx1, bx2, by1, by2 = 1 << x1, 1 << x2, 1 << y1, 1 << y2
                            if (current_mask & bx1) and (current_mask & bx2) and not (current_mask & by1) and not (current_mask & by2):
                                new_mask = current_mask ^ bx1 ^ bx2 ^ by1 ^ by2
                                # Accept kick regardless of score change to escape local min
                                current_mask = new_mask
                                current_score = calc_score(current_mask)
                                T = 0.5 # Reheat
                                no_improve = 0
                                break

            # Occasionally try a 2-swap move in the main loop for better exploration
            if random.random() < 0.1:
                # Efficient 2-swap attempt
                # We need 2 in, 2 out.
                # Randomly pick 4 distinct indices
                indices = random.sample(range(1, 513), 4)
                i1, i2, i3, i4 = indices
                b1, b2, b3, b4 = 1 << i1, 1 << i2, 1 << i3, 1 << i4
                
                # Check if i1, i2 are in and i3, i4 are out (or vice versa)
                in1 = bool(current_mask & b1)
                in2 = bool(current_mask & b2)
                in3 = bool(current_mask & b3)
                in4 = bool(current_mask & b4)
                
                if in1 and in2 and not in3 and not in4:
                    new_mask = current_mask ^ b1 ^ b2 ^ b3 ^ b4
                    new_score = calc_score(new_mask)
                    delta = new_score - current_score
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        no_improve = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                elif not in1 and not in2 and in3 and in4:
                    new_mask = current_mask ^ b1 ^ b2 ^ b3 ^ b4
                    new_score = calc_score(new_mask)
                    delta = new_score - current_score
                    if delta >= 0 or random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        no_improve = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask

    # Final Polish: Hill Climbing on the best solution found
    if time.time() < deadline - 2 and best_mask != 0:
        current_mask = best_mask
        current_score = best_score
        T = 0.01
        cool_rate = 0.9995
        min_T = 0.0001
        
        while time.time() < deadline - 0.5:
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
                
                if delta >= 0:
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                else:
                    # In polish, we are stricter, but still allow small downhill moves to escape micro-minima
                    if random.random() < math.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        
                T *= cool_rate

    result = get_elements(best_mask)
    
    # Ensure validity
    if len(result) != 32:
        return list(range(1, 33))
        
    # Ensure distinctness and range
    if len(set(result)) != 32 or any(x < 1 or x > 512 for x in result):
        return list(range(1, 33))
        
    return result
