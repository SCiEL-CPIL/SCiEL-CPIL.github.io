import time
import math

def construct():
    start_time = time.time()
    limit = 85  # Leave 5s for overhead
    
    N = 12
    period = 27720
    max_L = 144

    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def compute_f(m):
        # Compute the least L >= 1 such that there's a perfect matching
        
        for L in range(1, max_L + 1):
            # Build masks for each i
            # mask[i] has bit j set if j is valid for i in [0, L)
            # j corresponds to offset 0..L-1? 
            # b_i in (m, m+L] => b_i = m + x, 1 <= x <= L.
            # Condition: i | (m+x) => x % i == (-m) % i.
            # Let's use x in 1..L.
            # Bit j (0-indexed) corresponds to x = j+1.
            
            masks = [0] * (N + 1)
            valid_indices = []
            counts = [0] * (N + 1)
            
            feasible = True
            for i in range(1, N + 1):
                ri = (-m) % i
                # x in 1..L such that x % i == ri
                # If ri == 0, x is multiple of i.
                # x = ri + k*i.
                # If ri == 0, smallest positive x is i (if i<=L) or 0 is not allowed.
                # Wait, x >= 1.
                # If ri == 0: x = i, 2i, ...
                # If ri > 0: x = ri, ri+i, ...
                
                mask = 0
                count = 0
                # Start x
                if ri == 0:
                    start_x = i
                else:
                    start_x = ri
                    
                if start_x <= L:
                    for x in range(start_x, L + 1, i):
                        # x is in 1..L
                        # bit index is x-1
                        mask |= (1 << (x - 1))
                        count += 1
                else:
                    feasible = False
                    break
                
                masks[i] = mask
                counts[i] = count
                if count > 0:
                    valid_indices.append(i)
            
            if not feasible:
                continue
                
            if len(valid_indices) < N:
                continue
                
            # Sort indices by count (ascending) to help MRV
            # We need to pass the sorted order to backtrack
            # But MRV re-evaluates dynamically. Initial sort helps first step.
            
            indices = sorted(valid_indices, key=lambda x: counts[x])
            
            # If any count is 0, we already handled it.
            
            def backtrack(used_mask, remaining_indices):
                if not remaining_indices:
                    return True
                
                # Find index with minimum remaining values
                best_idx = -1
                best_count = L + 1
                best_avail = 0
                for idx in remaining_indices:
                    avail = masks[idx] & (~used_mask)
                    count = popcount(avail)
                    if count < best_count:
                        best_count = count
                        best_idx = idx
                        best_avail = avail
                    if count == 0:
                        return False # No options for this idx
                
                # Try all options for best_idx
                avail = best_avail
                while avail:
                    # Get lowest set bit
                    lsb = avail & (-avail)
                    avail ^= lsb
                    
                    # Assign this position to best_idx
                    # Remove best_idx from remaining
                    # To avoid list copying, we can pass a tuple or use a visited set, 
                    # but list of 12 is small. Copying is O(12).
                    new_remaining = [x for x in remaining_indices if x != best_idx]
                    
                    if backtrack(used_mask | lsb, new_remaining):
                        return True
                
                return False
            
            if backtrack(0, indices):
                return L
        
        return max_L

    best_m = 0
    best_f = 0
    
    # First pass: quick scan to find a good candidate and estimate cost
    # We will scan all m in period, but skip expensive checks if we have time pressure?
    # Actually, let's just do a full scan but optimize the inner loop.
    
    # Optimization: Precompute (-m) % i for all i and m?
    # (-m) % i depends on m.
    # m ranges 0..27719. i ranges 1..12.
    # We can precompute residues[m][i] = (-m) % i.
    
    residues = [[0] * (N + 1) for _ in range(period)]
    for m in range(period):
        for i in range(1, N + 1):
            residues[m][i] = (-m) % i

    # Also, we can precompute the masks for each L and m?
    # mask[m][i][L] is too large (27720 * 12 * 144).
    # But we can precompute the "start_x" and step for each m, i.
    # The mask for a given L is just bits at positions start_x, start_x+i, ... <= L.
    # Since L is small (<=144), we can compute mask quickly.
    
    # Let's optimize the mask generation in compute_f.
    # Instead of looping x in range(start_x, L+1, i), we can use a lookup table or bit operations.
    # For fixed i and start_x, the bits are at (start_x-1), (start_x-1)+i, ...
    # We can precompute for each i and residue r (0..i-1) a "pattern" of bits?
    # No, because L varies.
    
    # However, we can precompute for each i and L, and residue r, the mask.
    # mask_table[i][L][r]
    # Size: 12 * 144 * 12 = 20736 integers. Very small.
    
    mask_table = [[[0] * i for i in range(1, N + 1)] for L in range(max_L + 1)]
    # Wait, dimensions: mask_table[L][i][r]
    # L in 0..144, i in 1..12, r in 0..i-1
    
    mask_table = [None] * (max_L + 1)
    for L in range(max_L + 1):
        mask_table[L] = [None] * (N + 1)
        for i in range(1, N + 1):
            mask_table[L][i] = [0] * i
            for r in range(i):
                mask = 0
                count = 0
                if r == 0:
                    start_x = i
                else:
                    start_x = r
                if start_x <= L:
                    for x in range(start_x, L + 1, i):
                        mask |= (1 << (x - 1))
                        count += 1
                mask_table[L][i][r] = mask
                # We also need counts. Let's make a count_table.
    
    count_table = [None] * (max_L + 1)
    for L in range(max_L + 1):
        count_table[L] = [None] * (N + 1)
        for i in range(1, N + 1):
            count_table[L][i] = [0] * i
            for r in range(i):
                mask = mask_table[L][i][r]
                count_table[L][i][r] = popcount(mask)

    # Now compute_f is much faster
    def compute_f(m):
        res_m = residues[m]
        for L in range(1, max_L + 1):
            masks = [0] * (N + 1)
            valid_indices = []
            counts = [0] * (N + 1)
            
            feasible = True
            for i in range(1, N + 1):
                r = res_m[i]
                mask = mask_table[L][i][r]
                count = count_table[L][i][r]
                
                if count == 0:
                    feasible = False
                    break
                
                masks[i] = mask
                counts[i] = count
                valid_indices.append(i)
            
            if not feasible:
                continue
            
            indices = sorted(valid_indices, key=lambda x: counts[x])
            
            def backtrack(used_mask, remaining_indices):
                if not remaining_indices:
                    return True
                
                best_idx = -1
                best_count = L + 1
                best_avail = 0
                for idx in remaining_indices:
                    avail = masks[idx] & (~used_mask)
                    count = popcount(avail)
                    if count < best_count:
                        best_count = count
                        best_idx = idx
                        best_avail = avail
                    if count == 0:
                        return False
                
                avail = best_avail
                while avail:
                    lsb = avail & (-avail)
                    avail ^= lsb
                    new_remaining = [x for x in remaining_indices if x != best_idx]
                    if backtrack(used_mask | lsb, new_remaining):
                        return True
                return False
            
            if backtrack(0, indices):
                return L
        
        return max_L

    best_m = 0
    best_f = 0
    
    for m in range(period):
        if time.time() - start_time > limit:
            break
        
        f_val = compute_f(m)
        
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return best_m
