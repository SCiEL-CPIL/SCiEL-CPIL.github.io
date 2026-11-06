import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    
    # Precompute all 8-vertex subsets
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    # Precompute partitions of 8 indices into two sets of 4
    partitions = []
    all_8 = (1 << 8) - 1
    for x_combo in itertools.combinations(range(8), 4):
        x_mask = 0
        for i in x_combo:
            x_mask |= (1 << i)
        y_mask = all_8 ^ x_mask
        partitions.append((x_mask, y_mask))
        
    # Precompute bit counts for 0..255
    bit_counts = [0] * 256
    for i in range(256):
        bit_counts[i] = bin(i).count('1')
        
    # Precompute masks for each subset and partition?
    # subsets_8[i] is tuple of 8 vertices.
    # partitions[j] is (x_mask, y_mask) on 8 bits.
    # We can precompute global masks.
    # But 12870 * 70 * 2 integers is ~1.8M integers. ~15MB. OK.
    # Let's precompute to speed up has_q3.
    
    precomp_masks = []
    for s in subsets_8:
        s_masks = []
        for x_mask, y_mask in partitions:
            mask_x = 0
            mask_y = 0
            for i in range(8):
                if (x_mask >> i) & 1:
                    mask_x |= (1 << s[i])
                else:
                    mask_y |= (1 << s[i])
            s_masks.append((mask_x, mask_y))
        precomp_masks.append(s_masks)
        
    def has_q3(adj_masks):
        for i, s in enumerate(subsets_8):
            # Optional: quick edge count check
            # mask_s = ... (not precomputed, but can be derived or precomputed)
            # Let's precompute mask_s too.
            pass
            
    # Let's just use the loop with precomputed masks.
    # We also need mask_s for edge count check.
    mask_s_list = []
    for s in subsets_8:
        m = 0
        for v in s:
            m |= (1 << v)
        mask_s_list.append(m)
            
    def has_q3(adj_masks):
        for i in range(len(subsets_8)):
            mask_s = mask_s_list[i]
            deg_sum = 0
            for v in subsets_8[i]:
                deg_sum += bit_counts[adj_masks[v] & mask_s]
            if deg_sum // 2 < 12:
                continue
                
            s_masks = precomp_masks[i]
            for mask_x, mask_y in s_masks:
                valid = True
                # Check degrees
                # We need to iterate through the 8 vertices in s
                # and check if they are in X or Y.
                # But we have mask_x and mask_y.
                # For v in s:
                #   if v in X: check deg in Y
                #   else: check deg in X
                
                # To avoid checking "if v in X" using bit ops on 16-bit mask for each v,
                # we can just iterate 8 times.
                s = subsets_8[i]
                for k in range(8):
                    v = s[k]
                    if (mask_x >> v) & 1:
                         # v is in X
                         cnt = bit_counts[adj_masks[v] & mask_y]
                         if cnt < 3:
                             valid = False
                             break
                    else:
                         cnt = bit_counts[adj_masks[v] & mask_x]
                         if cnt < 3:
                             valid = False
                             break
                if valid:
                    return True
        return False
