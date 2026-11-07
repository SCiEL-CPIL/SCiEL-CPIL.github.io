import time
import random

P = {'n': 128}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = P['n']
    max_val = n
    
    # Precompute: for each x, bitmask of y such that x+y <= max_val
    # sum_mask[x] has bit (y-1) set if x+y <= max_val
    sum_mask = [0] * (max_val + 1)
    for x in range(1, max_val + 1):
        m = 0
        for y in range(1, max_val + 1):
            if x + y <= max_val:
                m |= (1 << (y - 1))
        sum_mask[x] = m
    
    # Precompute pairs_sum[s] = list of (a,b) with a<b, a+b=s, s<=max_val
    pairs_sum = [[] for _ in range(max_val + 1)]
    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            s = a + b
            if s <= max_val:
                pairs_sum[s].append((a, b))
    
    def is_valid_addition(x, mask):
        """Check if adding x to valid set 'mask' keeps it valid."""
        if mask & (1 << (x - 1)):
            return True
        
        # Case 1: x is a member. Need b,c in S, b!=c, b!=x, c!=x,
        # such that b+c in S, x+b in S, x+c in S.
        # A = {b in S | x+b in S}
        # Need b,c in A, b!=c, such that b+c in S.
        
        # Compute A: b in S and (x+b) in S
        # b_bit in A iff bit(b-1) in mask AND bit(x+b-1) in mask
        # x+b ranges from x+1 to x+128, only valid if <= max_val
        A_bits = 0
        # For each b in S, check if x+b in S
        # b must satisfy x+b <= max_val, so b <= max_val - x
        m = mask
        while m:
            lsb = m & -m
            b_idx = lsb.bit_length() - 1
            b_val = b_idx + 1
            if x + b_val <= max_val and (mask & (1 << (x + b_val - 1))):
                A_bits |= lsb
            m ^= lsb
        
        # Now check if there exist b,c in A_bits, b!=c, with b+c in S
        m_a = A_bits
        while m_a:
            lsb_b = m_a & -m_a
            b_idx = lsb_b.bit_length() - 1
            b_val = b_idx + 1
            m_a ^= lsb_b
            
            # For this b, need c in A_bits, c != b, with b+c in S
            # c must satisfy b+c <= max_val
            # Check: is there c in (A_bits & sum_mask[b]) with c != b and (b+c) in S?
            # c in A_bits means c in S and x+c in S
            # We need: c in A_bits, c != b, b+c <= max_val, and (b+c) in S
            
            # candidates_c = A_bits & sum_mask[b] & ~(1 << b_idx)
            candidates_c = A_bits & sum_mask[b_val] & ~(1 << b_idx)
            while candidates_c:
                lsb_c = candidates_c & -candidates_c
                c_val = lsb_c.bit_length()
                # c_val is 1-indexed
                sum_bc = b_val + c_val
                if sum_bc <= max_val and (mask & (1 << (sum_bc - 1))):
                    return False
                candidates_c ^= lsb_c
        
        # Case 2: x is a sum. a+b=x, a,b in S, a!=b.
        # Need c in S, c!=a, c!=b, with a+c in S and b+c in S.
        for a, b in pairs_sum[x]:
            if not (mask & (1 << (a - 1))):
                continue
            if not (mask & (1 << (b - 1))):
                continue
            
            # Need c in S \ {a,b} with (a+c) in S and (b+c) in S
            # c in S, a+c in S => c in S & sum_mask[a]
            # c in S, b+c in S => c in S & sum_mask[b]
            # c != a, c != b
            # candidates = mask & sum_mask[a] & sum_mask[b] & ~(1<<(a-1)) & ~(1<<(b-1))
            candidates = mask & sum_mask[a] & sum_mask[b] & ~(1 << (a - 1)) & ~(1 << (b - 1))
            if candidates:
                return False
        
        return True
    
    # Initial solution: upper half
    best_mask = 0
    for i in range(n // 2, n):
        best_mask |= (1 << i)
    best_size = n - n // 2
    
    current_mask = best_mask
    current_size = best_size
    
    # Local search with simulated annealing
    temp = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        if time.time() - start_time > 1:
            temp *= cooling
        
        # Try to add a random unused element
        unused = [i for i in range(1, n + 1) if not (current_mask & (1 << (i - 1)))]
        
        if unused:
            random.shuffle(unused)
            added = False
            for x in unused[:20]:  # Limit checks per iteration
                if is_valid_addition(x, current_mask):
                    current_mask |= (1 << (x - 1))
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    added = True
                    break
            if added:
                continue
        
        # Try 1-out, 2-in swap
        elements = [i for i in range(1, n + 1) if (current_mask & (1 << (i - 1)))]
        if elements and unused:
            x_out = random.choice(elements)
            mask_trial = current_mask & ~(1 << (x_out - 1))
            
            valid_x1 = []
            for x in unused:
                if is_valid_addition(x, mask_trial):
                    valid_x1.append(x)
            
            if len(valid_x1) >= 2:
                random.shuffle(valid_x1)
                found = False
                for i in range(min(len(valid_x1), 30)):
                    x1 = valid_x1[i]
                    mask_1 = mask_trial | (1 << (x1 - 1))
                    for j in range(i + 1, min(len(valid_x1), 30)):
                        x2 = valid_x1[j]
                        if is_valid_addition(x2, mask_1):
                            current_mask = mask_1 | (1 << (x2 - 1))
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_mask = current_mask
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue
        
        # Perturbation: remove 1-2 random elements
        if elements:
            num_remove = random.randint(1, 2)
            for _ in range(num_remove):
                if elements:
                    x_out = random.choice(elements)
                    current_mask &= ~(1 << (x_out - 1))
                    current_size -= 1
                    elements.remove(x_out)
    
    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
    
    return sorted(result)
