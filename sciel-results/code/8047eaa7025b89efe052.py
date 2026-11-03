import numpy as np
import itertools
import random
import time
import sys

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for larger spaces or general radius
    if total_words > 5000 or radius > 2:
        # Heuristic fallback
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Bitset-based Branch and Bound for Minimum Dominating Set
    # Vertices: 0 .. total_words-1
    # Edge in G_cover: dist(u, v) <= radius
    # We want Min Dominating Set in G_cover.
    
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # Precompute balls (neighbors including self)
    balls = [0] * total_words
    
    # Precompute close_masks (neighbors within 2*radius) for Lower Bound
    close_masks = [0] * total_words
    
    # Helper to get index from word
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    # Precompute distances? Or just generate neighbors.
    # For q=3, n=7, R=1:
    # Neighbors of w: w itself + change 1 pos to 2 other vals.
    # Total 1 + 7*2 = 15 neighbors.
    
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
        
        # For close_masks (dist <= 2*radius)
        # If radius=1, dist <= 2.
        # This is more expensive to compute directly.
        # close_mask[i] = union of balls[j] for all j in balls[i].
        # Since balls[i] is small, we can compute it after balls are done.
        
    # Compute close_masks
    for i in range(total_words):
        cm = 0
        # Iterate bits in balls[i]
        bi = balls[i]
        while bi:
            lsb = bi & -bi
            j = lsb.bit_length() - 1
            cm |= balls[j]
            bi ^= lsb
        close_masks[i] = cm

    all_ones_mask = (1 << total_words) - 1
    
    best_size = total_words + 1
    best_code_indices = []
    
    start_time = time.time()
    time_limit = 115.0
    timeout_flag = [False]

    def greedy_packing_lb(uncovered):
        """
        Lower bound: Max number of vertices in 'uncovered' that are pairwise at distance > 2*radius.
        Such vertices must be covered by distinct centers.
        """
        remaining = uncovered
        count = 0
        while remaining:
            # Pick lowest set bit
            lsb = remaining & -remaining
            u = lsb.bit_length() - 1
            count += 1
            # Remove u and all v that are "close" to u (dist <= 2*radius)
            # These v *could* be covered by the same center as u, so we don't need distinct centers for them in this LB step.
            remaining &= ~close_masks[u]
        return count

    def dfs(uncovered, count, selected_indices):
        nonlocal best_size, best_code_indices, timeout_flag
        
        if timeout_flag[0]:
            return
            
        if time.time() - start_time > time_limit:
            timeout_flag[0] = True
            return

        if uncovered == 0:
            if count < best_size:
                best_size = count
                best_code_indices = selected_indices[:]
            return

        # Lower Bound
        lb = greedy_packing_lb(uncovered)
        if count + lb >= best_size:
            return

        # Pick a vertex u from uncovered
        # Heuristic: Pick u that minimizes the number of candidates?
        # Candidates for covering u are centers c in balls[u].
        # All u have same number of candidates in full space.
        # In pruned space, maybe some u have fewer "effective" candidates?
        # Let's just pick the first one (lowest index) for simplicity, or random.
        # Random can help escape symmetries.
        # But for B&B, deterministic is often better for reproducibility, random for exploration.
        # Let's pick the u with the smallest "degree" in the remaining uncovered graph?
        # Too expensive.
        # Let's pick the first one.
        lsb = uncovered & -uncovered
        u = lsb.bit_length() - 1
        
        # Candidates: centers c such that u is in balls[c] <=> c is in balls[u]
        candidates_mask = balls[u]
        
        # Convert to list and sort by coverage
        cands = []
        cm = candidates_mask
        while cm:
            lsb_c = cm & -cm
            c = lsb_c.bit_length() - 1
            cands.append(c)
            cm ^= lsb_c
            
        # Sort by how much of 'uncovered' they cover (descending)
        # This helps find good solutions early, improving the bound
        cands.sort(key=lambda c: (uncovered & balls[c]).bit_count(), reverse=True)
        
        # Dominance Pruning:
        # If balls[c1] & uncovered is subset of balls[c2] & uncovered, skip c1.
        # We do this pairwise or by keeping a list of non-dominated candidates.
        filtered_cands = []
        for c in cands:
            cov_c = uncovered & balls[c]
            dominated = False
            for c2 in filtered_cands:
                cov_c2 = uncovered & balls[c2]
                # If cov_c is subset of cov_c2, c is dominated
                if (cov_c | cov_c2) == cov_c2:
                    dominated = True
                    break
            if not dominated:
                filtered_cands.append(c)
        
        for c in filtered_cands:
            new_uncovered = uncovered & ~balls[c]
            selected_indices.append(c)
            dfs(new_uncovered, count + 1, selected_indices)
            selected_indices.pop()
            
            # Pruning: if we already found a solution of size count+1, 
            # and we are exploring siblings, we might prune?
            # The bound check at the top handles this.

    dfs(all_ones_mask, 0, [])
    
    if best_code_indices:
        return [all_words[i] for i in best_code_indices]
    else:
        # Fallback if B&B failed to find a complete solution (should not happen if time is enough)
        # Return a greedy solution
        uncovered = all_ones_mask
        selected = []
        while uncovered:
            lsb = uncovered & -uncovered
            u = lsb.bit_length() - 1
            # Pick best center for u
            best_c = -1
            max_cov = 0
            cm = balls[u]
            while cm:
                lsb_c = cm & -cm
                c = lsb_c.bit_length() - 1
                cov = (uncovered & balls[c]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_c = c
                cm ^= lsb_c
            selected.append(best_c)
            uncovered &= ~balls[best_c]
        return [all_words[i] for i in selected]
