import time
import random

def construct():
        start_time = time.time()
        deadline = start_time + 145 # Safety margin
        
        def calc_score(mask):
            s = 0
            # d goes up to 255
            # Optimization: local variables
            m = mask
            for d in range(1, 256):
                if m & (m >> d) & (m >> (2 * d)):
                    s += 1
            return s

        best_mask = 0
        best_score = -1
        
        # Multiple restarts
        # We can use the time to do several independent SA runs
        # or one long run.
        # Given the landscape, multiple restarts are safer.
        
        num_restarts = 10 # Adjust based on time
        time_per_restart = (deadline - start_time) / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            # Try different inits
            if r == 0:
                # Start with 1..32
                mask = 0
                for i in range(1, 33):
                    mask |= (1 << i)
            elif r == 1:
                # Start with random
                mask = 0
                elems = random.sample(range(1, 513), 32)
                for i in elems:
                    mask |= (1 << i)
            else:
                # Perturbed best
                if best_mask != 0:
                    mask = best_mask
                    # Flip 4 random bits
                    for _ in range(4):
                        # Pick random bit to toggle
                        # Ensure we keep 32 bits? 
                        # Easier: pick random set bit to clear, random unset bit to set
                        set_bits = []
                        unset_bits = []
                        # This is slow. 
                        # Alternative: just pick random x in 1..512, y in 1..512
                        # If x in mask and y not in mask, swap.
                        pass
                
            # Simulated Annealing
            T = 1.0
            cool_rate = 0.999
            min_T = 0.01
            
            current_mask = mask
            current_score = calc_score(current_mask)
            
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
            
            # Pre-compute list of set and unset bits for faster move generation?
            # Or just generate random moves and check validity.
            
            while time.time() < deadline - 1: # Leave 1s for final check
                if T < min_T:
                    break
                
                # Generate move: Swap a random element with a random non-element
                # Pick random x in 1..512
                # Pick random y in 1..512
                # If x in mask and y not in mask:
                #   new_mask = mask ^ (1<<x) ^ (1<<y)
                # Else:
                #   continue (or retry)
                
                x = random.randint(1, 512)
                y = random.randint(1, 512)
                
                if x == y:
                    continue
                    
                bx = 1 << x
                by = 1 << y
                
                # Check if x is in mask and y is not
                # (x in mask) is (mask & bx) != 0
                # (y not in mask) is (mask & by) == 0
                if (mask & bx) and not (mask & by):
                    pass # This logic is inverted or wrong for swap
                    # We want to remove x and add y.
                    # So x must be in mask, y must NOT be in mask.
                
                # Correct check:
                if (current_mask & bx) and not (current_mask & by):
                    new_mask = current_mask ^ bx ^ by
                    new_score = calc_score(new_mask)
                    delta = new_score - current_score
                    
                    if delta >= 0 or random.random() < pow(2.71828, delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                
                T *= cool_rate
                
        # Convert best_mask to list
        result = []
        for i in range(1, 513):
            if best_mask & (1 << i):
                result.append(i)
        
        # Verify length
        if len(result) != 32:
             # Fallback
             return list(range(1, 33))
             
        return result
