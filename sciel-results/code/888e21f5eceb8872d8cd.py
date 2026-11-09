"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 12
    total_masks = 1 << n
    all_masks = list(range(total_masks))
    
    # Greedy construction
    chosen = []
    
    # To speed up, we can precompute the list of masks sorted by size or something?
    # Not strictly necessary for 4096 items.
    
    for _ in range(40):
        best_gain = -1
        best_mask = -1
        
        for m in all_masks:
            if m in chosen:
                continue
            
            gain = 0
            for c in chosen:
                # Check strict subset relations
                if m != c:
                    if (m & c) == m:
                        # m is subset of c
                        gain += 1
                    elif (m & c) == c:
                        # c is subset of m
                        gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_mask = m
        
        chosen.append(best_mask)
        
    # Convert masks to lists of integers
    result = []
    for m in chosen:
        s = []
        for i in range(n):
            if (m >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
