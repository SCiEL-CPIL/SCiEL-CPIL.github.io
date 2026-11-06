import time
from fractions import Fraction

# ... (keep existing helpers) ...

def _try_bb_search():
    # Precompute conflicts
    bad_masks = {a: [] for a in range(1, 25)}
    for a in range(1, 25):
        target = Fraction(1, a)
        others = [x for x in range(1, 25) if x != a]
        n = len(others)
        mid = n // 2
        left, right = others[:mid], others[mid:]
        
        R = {Fraction(0): [0]}
        for v in right:
            bit = 1 << (v - 1)
            new_R = {k: list(v_list) for k, v_list in R.items()}
            for s, masks in R.items():
                ns = s + Fraction(1, v)
                if ns in new_R:
                    new_R[ns].extend(m | bit for m in masks)
                else:
                    new_R[ns] = [m | bit for m in masks]
            R = new_R
            
        L = {Fraction(0): [0]}
        for v in left:
            bit = 1 << (v - 1)
            new_L = {k: list(v_list) for k, v_list in L.items()}
            for s, masks in L.items():
                ns = s + Fraction(1, v)
                if ns in new_L:
                    new_L[ns].extend(m | bit for m in masks)
                else:
                    new_L[ns] = [m | bit for m in masks]
                
                rem = target - ns
                if rem in R:
                    for m_l in masks:
                        for m_r in R[rem]:
                            bad_masks[a].append(m_l | m_r)
            L = new_L
            
        if target in R:
            bad_masks[a].extend(R[target])
            
    for a in range(1, 25):
        bad_masks[a] = list(set(bad_masks[a]))
        
    # Precompute required masks for fast check
    # req_masks[a][x] = list of m in bad_masks[a] such that x in m
    req_masks = {a: {x: [] for x in range(1, 25)} for a in range(1, 25)}
    for a in range(1, 25):
        for m in bad_masks[a]:
            # Find bits set in m
            temp = m
            while temp:
                lsb = temp & -temp
                x = lsb.bit_length()
                req_masks[a][x].append(m)
                temp ^= lsb
                
    best_size = 19 # From _BASE
    best_mask = 0
    for v in [5, 6] + list(range(8, 25)):
        best_mask |= 1 << (v - 1)
        
    def is_compatible(s_mask, x):
        bit_x = 1 << (x - 1)
        
        # 1. Check a=x
        for m in bad_masks[x]:
            if (m & s_mask) == m:
                return False
                
        # 2. Check a in s_mask
        temp = s_mask
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length()
            for m in req_masks[a][x]:
                # m contains x
                # Check if m \ {x} is subset of s_mask
                if (m & s_mask) == (m ^ bit_x):
                    return False
            temp ^= lsb
            
        return True

    def search(s_mask, c_mask):
        nonlocal best_size, best_mask
        
        if _deadline_hit():
            return
            
        if s_mask.bit_count() + c_mask.bit_count() <= best_size:
            return
            
        if c_mask == 0:
            if s_mask.bit_count() > best_size:
                best_size = s_mask.bit_count()
                best_mask = s_mask
            return
        
        # Pick highest index in c_mask
        lsb = c_mask & -c_mask
        x = lsb.bit_length()
        
        # Include x
        if is_compatible(s_mask, x):
            new_s = s_mask | lsb
            new_c = c_mask ^ lsb
            
            # Filter new_c
            filtered_c = 0
            temp_c = new_c
            while temp_c:
                lsb_c = temp_c & -temp_c
                y = lsb_c.bit_length()
                if is_compatible(new_s, y):
                    filtered_c |= lsb_c
                temp_c ^= lsb_c
                
            search(new_s, filtered_c)
            
        # Exclude x
        search(s_mask, c_mask ^ lsb)

    search(0, (1 << 24) - 1)
    
    return [i+1 for i in range(24) if (best_mask >> i) & 1]

