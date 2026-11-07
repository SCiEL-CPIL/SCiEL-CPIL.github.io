import time

def _dfs(k, used, masks, idx):
    if k == 12:
        return True
    i = idx[k]
    m = masks[i] & ~used
    while m:
        lsb = m & -m
        if _dfs(k + 1, used | lsb, masks, idx):
            return True
        m ^= lsb
    return False

def _check_match_masks(masks, idx):
    return _dfs(0, 0, masks, idx)

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = 12
    max_L = 144
    period = 27720  # lcm(1, 2, ..., 12)
    
    # Precompute f_values for one period
    f_values = [0] * period
    
    # Precompute f for one period
    for r in range(period):
        if time.time() > deadline:
            f_values[r] = 12 # Minimum possible
            continue
        
        # Precompute div_mask for this r
        # div_mask[i] corresponds to divisor i+1
        div_masks = [0] * 12
        for i in range(12):
            divisor = i + 1
            mask = 0
            for j in range(max_L):
                if (r + 1 + j) % divisor == 0:
                    mask |= (1 << j)
            div_masks[i] = mask
        
        found_L = max_L
        for L in range(12, max_L + 1):
            # Create masks for this L
            limit_mask = (1 << L) - 1
            masks = [dm & limit_mask for dm in div_masks]
            
            # Quick check: any empty?
            valid = True
            for m in masks:
                if m == 0:
                    valid = False
                    break
            if not valid:
                continue
                
            # Sort indices by popcount
            idx = list(range(12))
            idx.sort(key=lambda i: masks[i].bit_count())
            
            if _check_match_masks(masks, idx):
                found_L = L
                break
                
        f_values[r] = found_L

    # Find max f
    max_f = max(f_values)
    
    # Find smallest m in [0, 1000000] with f(m) == max_f
    for m in range(1000001):
        if f_values[m % period] == max_f:
            return int(m)
            
    return int(0)
