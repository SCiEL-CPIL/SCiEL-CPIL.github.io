import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        a = np.array(A, dtype=np.int64)
        sumset = a[:, None] + a[None, :]
        prodset = a[:, None] * a[None, :]
        s_size = len(np.unique(sumset))
        p_size = len(np.unique(prodset))
        return max(s_size, p_size)

    def build_structured_set(ap_start, ap_step, ap_len, gp_start, gp_ratio, gp_len):
        """Build a set as union of AP and GP, ensuring distinctness and bounds."""
        vals = set()
        # Arithmetic progression
        for i in range(ap_len):
            v = ap_start + i * ap_step
            if 1 <= v <= max_val:
                vals.add(v)
        # Geometric progression
        for i in range(gp_len):
            v = int(round(gp_start * (gp_ratio ** i)))
            if 1 <= v <= max_val:
                vals.add(v)
        
        if len(vals) < n:
            # Fill remaining with small integers not already in set
            for v in range(1, max_val + 1):
                if v not in vals:
                    vals.add(v)
                if len(vals) == n:
                    break
        
        result = sorted(list(vals))[:n]
        if len(result) != n:
            return None
        return result

    def refine_structured(start_set, max_iter=20000):
        """Refine a structured set via coordinate descent on parameters."""
        current = start_set[:]
        current_score = evaluate(current)
        local_best = current[:]
        local_best_score = current_score
        
        T = 5.0
        cooling = 0.999
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try modifying one element
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            
            # Generate candidate based on structure hints
            r = random.random()
            if r < 0.3:
                # Keep near current value (fine-tune)
                delta = random.randint(-10, 10)
                new_val = old_val + delta
            elif r < 0.5:
                # Try smooth number
                base = old_val
                mult = random.choice([2, 3, 5, 7, 1/2, 1/3, 1/5, 1/7])
                new_val = int(base * mult)
            elif r < 0.7:
                # Try geometric neighbor
                base = current[0] if len(current) > 0 else 1
                ratio = random.choice([2, 3, 4, 5, 1/2, 1/3])
                exp = random.randint(0, 8)
                new_val = int(round(base * (ratio ** exp)))
            elif r < 0.85:
                # Try arithmetic neighbor of first element
                base = current[0] if len(current) > 0 else 1
                step = random.randint(1, 50)
                k = random.randint(0, 20)
                new_val = base + k * step
            else:
                new_val = random.randint(1, max_val)
            
            new_val = max(1, min(max_val, new_val))
            
            if new_val in current:
                continue
                
            temp_set = current[:]
            temp_set[idx] = new_val
            temp_set.sort()
            
            new_score = evaluate(temp_set)
            
            if new_score < current_score:
                current = temp_set
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best = current[:]
            else:
                delta = new_score - current_score
                prob = np.exp(-delta / T) if T > 0 else 0
                if random.random() < prob:
                    current = temp_set
                    current_score = new_score
            
            T *= cooling

        return local_best, local_best_score

    # Generate structured candidates
    candidates = []
    
    # 1. Pure multiplicative grid 2^i * 3^j
    grid_vals = []
    for i in range(6):
        for j in range(6):
            v = (2 ** i) * (3 ** j)
            if v <= max_val:
                grid_vals.append(v)
    grid_vals = sorted(set(grid_vals))[:n]
    if len(grid_vals) == n:
        candidates.append(grid_vals)
    
    # 2. AP + GP union with various parameters
    ap_starts = [1, 2, 3, 5, 10]
    ap_steps = [1, 2, 3, 5, 10, 20]
    gp_starts = [1, 2, 3, 5]
    gp_ratios = [2, 3, 4, 5]
    
    for ap_s in ap_starts:
        for ap_st in ap_steps:
            for gp_s in gp_starts:
                for gp_r in gp_ratios:
                    # Try different splits
                    for ap_len in range(5, 16):
                        gp_len = n - ap_len
                        if gp_len < 3:
                            continue
                        s = build_structured_set(ap_s, ap_st, ap_len, gp_s, gp_r, gp_len)
                        if s is not None and len(set(s)) == n:
                            candidates.append(s)
    
    # 3. Small smooth numbers
    smooth_5 = []
    for x in range(1, max_val + 1):
        t = x
        for p in [2, 3, 5]:
            while t % p == 0:
                t //= p
        if t == 1:
            smooth_5.append(x)
            if len(smooth_5) == n:
                break
    if len(smooth_5) == n:
        candidates.append(smooth_5)
    
    # 4. Random
    for _ in range(5):
        cand = sorted(random.sample(range(1, max_val + 1), n))
        candidates.append(cand)

    # Run refinement on candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        
        if len(set(cand)) != n or max(cand) > max_val or min(cand) < 1:
            continue
            
        res_set, res_score = refine_structured(cand)
        
        if res_score < best_score:
            best_score = res_score
            best_set = res_set[:]

    # Fallback
    if best_set is None:
        best_set = list(range(1, 21))
        
    return best_set
