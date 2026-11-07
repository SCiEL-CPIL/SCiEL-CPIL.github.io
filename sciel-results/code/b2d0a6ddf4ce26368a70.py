import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    N = 128
    
    # Bit i represents integer (i+1), so bit (x-1) for value x
    
    def get_bitset(s):
        b = 0
        for x in s:
            b |= (1 << (x - 1))
        return b
    
    def bits_to_list(b):
        """Convert 128-bit int to sorted list of ints (1..128)."""
        result = []
        temp = b
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            result.append(idx + 1)
            temp ^= lsb
        result.sort()
        return result
    
    def popcount(b):
        return bin(b).count('1')
    
    # Precompute: for each x in 1..128, the bitmask of values a such that x+a <= 128
    # i.e., a in 1..(128-x)
    # mask_partners[x] has bits (a-1) set for a in 1..(128-x)
    mask_partners = [0] * (N + 1)
    for x in range(1, N + 1):
        m = 0
        for a in range(1, N - x + 1):
            m |= (1 << (a - 1))
        mask_partners[x] = m
    
    # Precompute sum_mask[s] for s in 2..256: bit (s-1) set if s <= 128
    # Actually we just need to check if s is in B, which is B & (1 << (s-1))
    
    def count_conflicts(B, x):
        """
        Count bad triples involving x when x is NOT in B.
        A bad triple {x, a, b} requires: a,b in B, x+a in B, x+b in B, a+b in B.
        """
        # A = {a in B : x+a in B} = B & (B >> x)
        # But we need to be careful: B >> x shifts bit (i+x) to bit i
        # So bit i in (B >> x) is set iff bit (i+x) in B is set
        # We want a such that a in B AND (a+x) in B
        # a is represented by bit (a-1) in B
        # a+x is represented by bit (a+x-1) in B
        # So we need bit (a-1) in B AND bit (a+x-1) in B
        # (B >> x) has bit (a-1) set iff B has bit (a-1+x) = bit (a+x-1) set. Correct.
        A_bits = B & (B >> x)
        
        # Now for each pair (a, b) in A with a < b, check if a+b in B
        count = 0
        a_list = []
        temp = A_bits
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            a_list.append(idx + 1)
            temp ^= lsb
        
        n = len(a_list)
        for i in range(n):
            a = a_list[i]
            for j in range(i + 1, n):
                b = a_list[j]
                s = a + b
                if s <= N:
                    if B & (1 << (s - 1)):
                        count += 1
        return count
    
    def is_valid(B):
        """Check if bitset B represents a valid set (no bad triples)."""
        sl = bits_to_list(B)
        n = len(sl)
        for i in range(n):
            a = sl[i]
            for j in range(i + 1, n):
                b = sl[j]
                if a + b > N:
                    break
                if not (B & (1 << (a + b - 1))):
                    continue
                for k in range(j + 1, n):
                    c = sl[k]
                    if a + c > N or b + c > N:
                        break
                    if (B & (1 << (a + c - 1))) and (B & (1 << (b + c - 1))):
                        return False
        return True
    
    # Multi-restart randomized local search
    best_size = 0
    best_B = 0
    
    # Start with upper half as baseline
    init_set = set(range(65, 129))
    init_B = get_bitset(init_set)
    if is_valid(init_B):
        best_size = len(init_set)
        best_B = init_B
    
    all_values = list(range(1, N + 1))
    
    # Multiple restarts with different strategies
    restarts = 0
    max_restarts = 200
    
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        
        # Different initialization strategies
        if restarts == 1:
            # Upper half
            current_B = init_B
        elif restarts == 2:
            # Lower half shifted
            current_B = get_bitset(set(range(1, 65)))
            if not is_valid(current_B):
                # Remove elements until valid
                sl = bits_to_list(current_B)
                for x in sl:
                    new_B = current_B & ~(1 << (x - 1))
                    if is_valid(new_B):
                        current_B = new_B
                if not is_valid(current_B):
                    current_B = init_B
        elif restarts == 3:
            # Random subset of size ~50
            current_B = 0
            random.shuffle(all_values)
            for x in all_values[:50]:
                new_B = current_B | (1 << (x - 1))
                if count_conflicts(current_B, x) == 0:
                    current_B = new_B
        elif restarts % 2 == 0:
            # Random valid set via greedy
            current_B = 0
            random.shuffle(all_values)
            for x in all_values:
                if count_conflicts(current_B, x) == 0:
                    current_B = current_B | (1 << (x - 1))
        else:
            # Start from best found so far, perturb
            current_B = best_B
            # Remove 5 random elements
            sl = bits_to_list(current_B)
            if len(sl) > 10:
                to_remove = random.sample(sl, min(5, len(sl) // 4))
                for x in to_remove:
                    current_B &= ~(1 << (x - 1))
        
        current_size = popcount(current_B)
        
        if current_size > best_size and is_valid(current_B):
            best_size = current_size
            best_B = current_B
        
        # Local search phase
        T = 10.0
        cooling = 0.9995
        min_T = 0.01
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 5 and T > min_T and no_improve_count < max_no_improve:
            r = random.random()
            
            if r < 0.6:
                # Swap: remove x from S, add y not in S
                sl = bits_to_list(current_B)
                if not sl:
                    break
                x = random.choice(sl)
                # Candidates for y
                candidates = [z for z in all_values if not (current_B & (1 << (z - 1)))]
                if not candidates:
                    break
                y = random.choice(candidates)
                
                # Check if (S \ {x}) ∪ {y} is valid
                S_temp_B = current_B & ~(1 << (x - 1))
                c = count_conflicts(S_temp_B, y)
                if c == 0:
                    new_B = S_temp_B | (1 << (y - 1))
                    new_size = popcount(new_B)
                    # Accept swap (size unchanged) to explore
                    current_B = new_B
                    current_size = new_size
                    no_improve_count += 1
                else:
                    no_improve_count += 1
                    
            elif r < 0.85:
                # Add: add y not in S
                candidates = [z for z in all_values if not (current_B & (1 << (z - 1)))]
                if not candidates:
                    break
                y = random.choice(candidates)
                c = count_conflicts(current_B, y)
                if c == 0:
                    new_B = current_B | (1 << (y - 1))
                    new_size = popcount(new_B)
                    current_B = new_B
                    current_size = new_size
                    no_improve_count = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_B = current_B
                else:
                    no_improve_count += 1
            else:
                # Remove: remove x from S (to escape local minima)
                sl = bits_to_list(current_B)
                if not sl:
                    break
                x = random.choice(sl)
                new_B = current_B & ~(1 << (x - 1))
                # Always valid after removal
                current_B = new_B
                current_size = popcount(new_B)
                no_improve_count += 1
            
            T *= cooling
        
        # After local search, try to greedily add more
        if time.time() < deadline - 2:
            changed = True
            while changed and time.time() < deadline - 2:
                changed = False
                for x in all_values:
                    if not (current_B & (1 << (x - 1))):
                        if count_conflicts(current_B, x) == 0:
                            current_B |= (1 << (x - 1))
                            current_size = popcount(current_B)
                            changed = True
                            if current_size > best_size:
                                best_size = current_size
                                best_B = current_B
    
    # Final validation
    if not is_valid(best_B):
        # Fallback to upper half
        best_B = init_B
        best_size = len(init_set)
    
    return bits_to_list(best_B)
