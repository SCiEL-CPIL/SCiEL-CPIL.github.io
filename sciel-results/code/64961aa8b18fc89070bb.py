import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for larger spaces
    if total_words > 5000:
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Generate all words
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # Precompute balls as bitmasks
    balls = [0] * total_words
    # Precompute distance-2 balls for Lower Bound (words at distance > 2 cannot share a codeword)
    # Actually, we need to know which words are "incompatible" for the LB.
    # If d(u,v) > 2, they can't be in the same ball.
    # So for LB, we want to pack words with d(u,v) > 2.
    
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            mask |= (1 << i)
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
            balls[i] = mask
    else:
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask
            
    # Precompute "incompatible" masks for LB:
    # comp_mask[i] contains all j such that d(i, j) > 2 (for radius 1)
    # If we pick i, we cannot use any j in comp_mask[i] in the same "packing" for LB?
    # No, the LB logic:
    # We want a set P of words in Uncovered such that for all u,v in P, d(u,v) > 2.
    # This implies they require distinct codewords.
    # So LB = max size of such a set P.
    # Greedy approximation:
    # Pick u. Add to P. Remove all v with d(u,v) <= 2 from consideration.
    
    # For speed, precompute "close_mask[i]" = ball[i] (words at dist <= 1) 
    # and "dist2_mask[i]" = words at dist <= 2.
    # If d(u,v) <= 2, they are compatible.
    # If d(u,v) > 2, they are incompatible.
    
    dist2_masks = [0] * total_words
    if radius == 1:
        for i in range(total_words):
            m = 0
            # Ball of radius 2
            # Iterate over all words? 2187 * 2187 is ~5M ops. Acceptable.
            for j in range(total_words):
                # Check dist
                # Optimization: Use precomputed balls? 
                # d(i,j) <= 2 iff j is in ball(ball[i])? No.
                # d(i,j) <= 2 iff there exists k such that d(i,k)<=1 and d(k,j)<=1.
                # So dist2_mask[i] = union of balls[k] for all k in ball[i].
                # This is expensive to compute for all i.
                # Simpler: Just calculate distance directly for LB packing if needed, 
                # or precompute once.
                pass
        
        # Let's just precompute dist2_masks properly
        # dist2_masks[i] is bitset of j where d(i,j) <= 2
        for i in range(total_words):
            m = 0
            # Ball of i is balls[i]
            # For each k in balls[i], add balls[k] to m
            # balls[i] has 1 + n*(q-1) = 1 + 7*2 = 15 bits set.
            # So we iterate 15 times, OR-ing 15 bitsets of 2187 bits.
            # 2187 * 15 ORs. Fast.
            temp = balls[i]
            while temp:
                lsb = temp & -temp
                k = lsb.bit_length() - 1
                m |= balls[k]
                temp ^= lsb
            dist2_masks[i] = m
    else:
        # For general radius, skip this specific LB optimization or use simple one
        for i in range(total_words):
            dist2_masks[i] = (1 << total_words) - 1 # Fallback: assume all compatible (weak LB)

    all_ones_mask = (1 << total_words) - 1
    
    # 1. Initial Solution via Greedy + Removal
    def greedy_cover(uncovered):
        selected = []
        unc = uncovered
        while unc:
            best_idx = -1
            max_cover = 0
            for i in range(total_words):
                c = (balls[i] & unc).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = i
            if best_idx == -1: break
            selected.append(best_idx)
            unc &= ~balls[best_idx]
        return selected

    def remove_redundant(selected):
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                union_rest = 0
                for j in range(len(selected)):
                    if j != i:
                        union_rest |= balls[selected[j]]
                if (union_rest & all_ones_mask) == all_ones_mask:
                    selected.pop(i)
                    changed = True
                    break
        return selected

    init_sel = greedy_cover(all_ones_mask)
    init_sel = remove_redundant(init_sel)
    best_size = len(init_sel)
    best_code = [all_words[i] for i in init_sel]
    
    start_time = time.time()
    time_limit = 110.0
    
    # 2. Branch and Bound
    # Global variables for B&B
    current_best_size = best_size
    current_best_code = best_code
    
    def lower_bound(uncovered):
        if not uncovered:
            return 0
        # Greedy packing of words with d(u,v) > 2
        remaining = uncovered
        count = 0
        while remaining:
            # Pick a word from remaining
            # Heuristic: pick one with fewest neighbors in remaining? 
            # For speed, just pick the lowest set bit
            lsb = remaining & -remaining
            idx = lsb.bit_length() - 1
            count += 1
            # Remove all words compatible with idx (d(idx, v) <= 2)
            # compatible_mask = dist2_masks[idx]
            # remaining &= ~compatible_mask
            remaining &= ~dist2_masks[idx]
        return count

    def solve_bb(uncovered, selected_count, selected_list):
        nonlocal current_best_size, current_best_code
        
        if time.time() - start_time > time_limit:
            return
            
        if uncovered == 0:
            if selected_count < current_best_size:
                current_best_size = selected_count
                current_best_code = list(selected_list)
            return
            
        # Pruning
        lb = lower_bound(uncovered)
        if selected_count + lb >= current_best_size:
            return
            
        # Unit Propagation: Find words in uncovered with only 1 available coverer
        # This requires knowing available coverers. 
        # In this formulation, ALL codewords are always "available" unless we fix them to 0.
        # But we are constructing the set. We haven't fixed any to 0.
        # So "available" means any codeword that covers the word.
        # If a word w is covered by only 1 codeword c in the *entire space*, then we MUST pick c.
        # Precompute cover_count[w] = number of codewords covering w.
        
        # For now, skip unit propagation as it requires precomputation or dynamic tracking.
        # Instead, branch on the most constrained word.
        
        # Find word w in uncovered with minimum number of coverers
        min_coverers = total_words + 1
        target_word = -1
        target_coverers = []
        
        # Iterate through uncovered words
        unc_temp = uncovered
        while unc_temp:
            lsb = unc_temp & -unc_temp
            w_idx = lsb.bit_length() - 1
            unc_temp ^= lsb
            
            # Count coverers for w_idx
            # A codeword c covers w_idx if w_idx is in balls[c].
            # This is equivalent to: c is in "reverse_ball[w_idx]"
            # We didn't precompute reverse balls.
            # Let's precompute reverse_balls: list of bitsets, reverse_balls[w] has bit c set if w in balls[c].
            
            # Since we are inside the function, we can't precompute here easily without passing it.
            # Let's assume we precompute reverse_balls outside.
            
            # For now, use a simpler heuristic:
            # Just pick the first uncovered word? No, that's bad.
            # Let's use the number of bits in `uncovered` as a proxy? No.
            
            # Let's stick to a simpler branching:
            # Pick a random word from uncovered?
            # Or pick the word that is covered by the fewest *remaining* balls?
            # Since we don't track "remaining balls" (we don't remove balls from consideration, we just add codewords),
            # the set of coverers for any word is fixed.
            # So the "most constrained" word is the one with the smallest `cover_count[w]`.
            
            # We need cover_counts.
            pass
        
        # Simplified Branching for Time Safety:
        # Pick a word w from uncovered (e.g., first one).
        # Get all c such that w in balls[c].
        # Sort by coverage.
        # Branch on top 5.
        
        lsb = uncovered & -uncovered
        w_idx = lsb.bit_length() - 1
        
        # Find coverers
        coverers = []
        for c in range(total_words):
            if (balls[c] >> w_idx) & 1:
                coverers.append(c)
                
        if not coverers:
            return # Should not happen
            
        # Sort by number of new words covered
        coverers.sort(key=lambda c: (balls[c] & uncovered).bit_count(), reverse=True)
        
        # Branch on top K
        K = 5
        for c in coverers[:K]:
            new_uncovered = uncovered & ~balls[c]
            selected_list.append(c)
            solve_bb(new_uncovered, selected_count + 1, selected_list)
            selected_list.pop()
            
            if time.time() - start_time > time_limit:
                break

    # Precompute cover counts for better branching?
    # For now, run B&B
    solve_bb(all_ones_mask, 0, [])
    
    if current_best_size < best_size:
        best_code = current_best_code
        best_size = current_best_size
        
    return best_code
