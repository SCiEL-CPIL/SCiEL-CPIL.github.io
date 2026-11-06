"""Rewritten program: bitmask + steepest-ascent local search for max 3-APs with no 4-AP."""
import time
import random

N = 256
DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Precompute 3-AP masks: for each (a, d) with a+3d <= N, the mask of {a, a+d, a+2d}
    # We'll represent the set as a Python int bitmask (bit i set means i+1 is in set, or bit i-1 for i in 1..N)
    # Let's use bit position i-1 for element i, so bit 0 = element 1, bit 255 = element 256
    
    # Precompute all 3-APs as bitmasks
    ap3_masks = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            a3 = a + 2 * d
            if a3 <= N:
                mask = (1 << (a - 1)) | (1 << (a + d - 1)) | (1 << (a3 - 1))
                ap3_masks.append(mask)
    
    # Precompute all 4-APs as bitmasks
    ap4_masks = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            a4 = a + 3 * d
            if a4 <= N:
                mask = (1 << (a - 1)) | (1 << (a + d - 1)) | (1 << (a + 2 * d - 1)) | (1 << (a4 - 1))
                ap4_masks.append(mask)
    
    def count_3aps(mask):
        """Count 3-APs fully contained in the set represented by mask."""
        count = 0
        for ap_mask in ap3_masks:
            if (mask & ap_mask) == ap_mask:
                count += 1
        return count
    
    def has_4ap(mask):
        """Check if mask contains any 4-AP."""
        for ap_mask in ap4_masks:
            if (mask & ap_mask) == ap_mask:
                return True
        return False
    
    # Greedy construction: add elements that maximize 3-APs while avoiding 4-APs
    # Start with a random greedy seed
    def greedy_seed():
        """Build a set by greedily adding elements that create 3-APs without creating 4-APs."""
        mask = 0
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        
        for x in elements:
            bit = 1 << (x - 1)
            new_mask = mask | bit
            if has_4ap(new_mask):
                continue
            # Count how many new 3-APs this creates
            # A 3-AP is "new" if it wasn't fully in mask before
            new_3aps = 0
            for ap_mask in ap3_masks:
                if (ap_mask & bit) == bit:  # x is in this AP
                    if (mask & ap_mask) != ap_mask and (new_mask & ap_mask) == ap_mask:
                        new_3aps += 1
            # Add if it creates at least one new 3-AP, or always add to build density
            mask = new_mask
        
        return mask
    
    # Local search with swaps
    def local_search(mask, max_iters=5000):
        """Steepest-ascent local search: try adding, removing, or swapping elements."""
        best_mask = mask
        best_score = count_3aps(mask)
        
        # Precompute for each element x, which 3-APs and 4-APs it participates in
        # For efficiency, we'll compute gains/losses on the fly but use bit operations
        
        for iteration in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            best_delta = 0
            best_move = None  # (type, x, y) where type in {'add', 'remove', 'swap'}
            
            # Try adding each element not in mask
            for x in range(1, N + 1):
                bit = 1 << (x - 1)
                if (best_mask & bit) == bit:
                    continue
                new_mask = best_mask | bit
                if has_4ap(new_mask):
                    continue
                # Compute new 3-APs created by adding x
                new_3aps = 0
                for ap_mask in ap3_masks:
                    if (ap_mask & bit) == bit:
                        if (new_mask & ap_mask) == ap_mask and (best_mask & ap_mask) != ap_mask:
                            new_3aps += 1
                if new_3aps > best_delta:
                    best_delta = new_3aps
                    best_move = ('add', x, None)
            
            # Try removing each element in mask
            for x in range(1, N + 1):
                bit = 1 << (x - 1)
                if (best_mask & bit) != bit:
                    continue
                new_mask = best_mask & ~bit
                # Removing can only decrease or keep 3-AP count, but might allow future adds
                # For now, only remove if it doesn't decrease score (unlikely to help)
                # Skip removals in this phase for speed
                pass
            
            # Try swaps: remove x, add y
            for x in range(1, N + 1):
                bit_x = 1 << (x - 1)
                if (best_mask & bit_x) != bit_x:
                    continue
                for y in range(1, N + 1):
                    bit_y = 1 << (y - 1)
                    if (best_mask & bit_y) == bit_y:
                        continue
                    new_mask = (best_mask & ~bit_x) | bit_y
                    if has_4ap(new_mask):
                        continue
                    # Compute delta in 3-AP count
                    old_3aps = 0
                    new_3aps = 0
                    for ap_mask in ap3_masks:
                        old_full = (best_mask & ap_mask) == ap_mask
                        new_full = (new_mask & ap_mask) == ap_mask
                        if old_full and not new_full:
                            old_3aps += 1
                        if new_full and not old_full:
                            new_3aps += 1
                    delta = new_3aps - old_3aps
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', x, y)
            
            if best_move is not None and best_delta > 0:
                typ, x, y = best_move
                if typ == 'add':
                    best_mask |= (1 << (x - 1))
                elif typ == 'swap':
                    best_mask = (best_mask & ~(1 << (x - 1))) | (1 << (y - 1))
                best_score += best_delta
                improved = True
            
            if not improved:
                break
        
        return best_mask, best_score
    
    # Multi-start local search
    best_mask_overall = 0
    best_score_overall = 0
    
    # First, try a few different greedy seeds
    num_seeds = 5
    for s in range(num_seeds):
        if time.time() - start_time > DEADLINE - 5:
            break
        mask = greedy_seed()
        score = count_3aps(mask)
        if score > best_score_overall:
            best_score_overall = score
            best_mask_overall = mask
    
    # Run local search from the best seed
    if time.time() - start_time < DEADLINE - 10:
        best_mask_overall, best_score_overall = local_search(best_mask_overall, max_iters=2000)
    
    # Convert mask to sorted list
    result = []
    for i in range(N):
        if (best_mask_overall >> i) & 1:
            result.append(i + 1)
    
    if not result:
        # Fallback: return a small valid set
        result = [1, 2, 4]
    
    return sorted(result)
