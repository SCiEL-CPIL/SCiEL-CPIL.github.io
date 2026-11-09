import random
import itertools
import time

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples and their bitmasks
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                mask = (1 << a) | (1 << b) | (1 << c)
                p_ab = (a, b)
                p_ac = (a, c)
                p_bc = (b, c)
                all_triples.append((mask, p_ab, p_ac, p_bc, (a,b,c)))
                
    # Total triples: 560
    # Max possible linear packing size: 37
    
    best_alpha = n + 1
    best_triples = []
    
    # Precompute popcount for all masks up to 2^16
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Function to calculate independence number using precomputed blocked array
    def calc_alpha_fast(masks):
        # Build a "blocked" array: blocked[s] = 1 if subset s contains any triple
        # For each triple mask t, mark all supersets of t as blocked
        blocked = bytearray(1 << n)
        for t in masks:
            # Find all supersets of t
            free_bits = []
            for i in range(n):
                if not (t & (1 << i)):
                    free_bits.append(i)
            num_free = len(free_bits)
            for sub in range(1 << num_free):
                s = t
                for j in range(num_free):
                    if sub & (1 << j):
                        s |= (1 << free_bits[j])
                blocked[s] = 1
        
        max_ind = 0
        for s in range(1 << n):
            if blocked[s] == 0:
                cnt = popcount[s]
                if cnt > max_ind:
                    max_ind = cnt
        return max_ind
    
    # Greedy packing generator
    def generate_packing():
        random.shuffle(all_triples)
        used_pairs = set()
        triples = []
        masks = []
        
        for mask, p1, p2, p3, pts in all_triples:
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
            triples.append(pts)
            masks.append(mask)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            if len(triples) == 37:
                break
        
        return triples, masks, len(triples)
    
    # Local search: try to improve alpha by swapping triples
    def local_search(triples, masks, alpha):
        """Try to reduce alpha by swapping one triple for another."""
        used_pairs = set()
        for pts in triples:
            a, b, c = pts
            used_pairs.add((a, b))
            used_pairs.add((a, c))
            used_pairs.add((b, c))
        
        current_alpha = alpha
        improved = True
        max_iter = 50  # Limit local search iterations to save time
        
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            # Try each triple in the packing
            for i in range(len(triples)):
                a, b, c = triples[i]
                # Remove this triple
                new_used_pairs = set(used_pairs)
                new_used_pairs.discard((a, b))
                new_used_pairs.discard((a, c))
                new_used_pairs.discard((b, c))
                
                # Try to add a different triple
                for j in range(len(all_triples)):
                    mask, p1, p2, p3, pts = all_triples[j]
                    if pts == triples[i]:
                        continue
                    if p1 in new_used_pairs or p2 in new_used_pairs or p3 in new_used_pairs:
                        continue
                    
                    # Compute new alpha
                    new_triples = triples[:]
                    new_triples[i] = pts
                    new_masks = []
                    for t in new_triples:
                        m = 0
                        for p in t:
                            m |= (1 << p)
                        new_masks.append(m)
                    
                    new_alpha = calc_alpha_fast(new_masks)
                    
                    if new_alpha < current_alpha:
                        triples = new_triples
                        masks = new_masks
                        current_alpha = new_alpha
                        # Update used_pairs
                        used_pairs = set()
                        for pts in triples:
                            x, y, z = pts
                            used_pairs.add((x, y))
                            used_pairs.add((x, z))
                            used_pairs.add((y, z))
                        improved = True
                        break
                if improved:
                    break
        
        return triples, masks, current_alpha
    
    # Run search
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        triples, masks, size = generate_packing()
        
        if size < 37:
            continue
            
        alpha = calc_alpha_fast(masks)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            
            # Try local search to improve further
            if time.time() - start_time < timeout - 5:
                triples_ls, masks_ls, alpha_ls = local_search(triples, masks, alpha)
                if alpha_ls < best_alpha:
                    best_alpha = alpha_ls
                    best_triples = triples_ls
                
            if best_alpha <= 4:
                break
    
    # If no packing of size 37 found (unlikely), return the best found or seed
    if not best_triples:
        # Fallback to seed
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_triples

# EVOLVE-BLOCK-START
# The construct function is defined above; this block is kept for structural compatibility
# No additional code needed here as the main logic is in construct()
# EVOLVE-BLOCK-END
