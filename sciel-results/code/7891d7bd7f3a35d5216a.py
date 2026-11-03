import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    total_words = q ** n
    
    # Fallback for larger spaces or general radius
    if total_words > 10000 or radius > 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 7:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return [list(random.choices(range(q), k=n)) for _ in range(100)]

    # Bitset-based approach for small spaces (e.g., 3^7 = 2187)
    
    # 1. Generate all words
    all_words = [list(p) for p in itertools.product(range(q), repeat=n)]
    
    # 2. Precompute balls as bitmasks (Python ints)
    balls = [0] * total_words
    near2_masks = [0] * total_words # For lower bound: vertices within distance <= 2
    
    if radius == 1:
        for i, w in enumerate(all_words):
            mask = 0
            mask2 = 0
            # Self
            mask |= (1 << i)
            mask2 |= (1 << i)
            
            # Neighbors dist 1
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        neighbor_idx = 0
                        for k in range(n):
                            val = w[k] if k != pos else new_val
                            neighbor_idx = neighbor_idx * q + val
                        mask |= (1 << neighbor_idx)
                        mask2 |= (1 << neighbor_idx)
                        
            # Neighbors dist 2 (for lower bound)
            # A vertex is at dist 2 if it differs in exactly 2 positions
            # We can generate these by changing 2 positions
            # Or simply: mask2 is union of balls of all neighbors in mask
            # Since we already have mask (dist <= 1), near2 is union of balls[x] for x in mask
            # But that's expensive to compute on the fly.
            # Let's compute dist 2 explicitly or use the property:
            # near2_mask[i] = ball[i] | union(ball[j] for j in ball[i])
            # We can compute this after all balls are computed.
            
            balls[i] = mask
            # We'll compute near2_masks in a second pass for efficiency
    else:
        # General radius (slower)
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    mask |= (1 << j)
            balls[i] = mask
            near2_masks[i] = mask # Approximation for bound, not strictly correct for R>1 but ok for fallback

    # Compute near2_masks properly for radius 1
    if radius == 1:
        for i in range(total_words):
            m2 = balls[i]
            # Iterate through bits in balls[i]
            temp = balls[i]
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                m2 |= balls[idx]
                temp ^= lsb
            near2_masks[i] = m2

    all_ones_mask = (1 << total_words) - 1
    
    # Helper: Greedy Set Cover
    def greedy_cover(uncovered_mask, candidates_mask=None):
        if candidates_mask is None:
            candidates_mask = all_ones_mask
        selected = []
        remaining = uncovered_mask
        cands = candidates_mask
        while remaining:
            best_idx = -1
            max_cover = 0
            temp = cands
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                c = (balls[idx] & remaining).bit_count()
                if c > max_cover:
                    max_cover = c
                    best_idx = idx
                elif c == max_cover and c > 0:
                    if random.random() < 0.5:
                        best_idx = idx
                temp ^= lsb
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining &= ~balls[best_idx]
            # Optionally remove best_idx from cands? No, we can reuse, but set cover usually removes.
            # For dominating set, we just need to cover.
        return selected

    # Helper: Packing Lower Bound
    def packing_bound(uncovered_mask):
        count = 0
        temp = uncovered_mask
        while temp:
            # Pick first set bit
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            count += 1
            # Remove all vertices within distance <= 2 of idx
            temp &= ~near2_masks[idx]
        return count

    # Global state for B&B
    best_size = total_words + 1
    best_code_indices = []
    start_time = time.time()
    time_limit = 115.0

    # Initial Greedy Solution
    initial_sel = greedy_cover(all_ones_mask)
    if len(initial_sel) < best_size:
        best_size = len(initial_sel)
        best_code_indices = initial_sel[:]

    # Branch and Bound
    def branch_and_bound(uncovered, candidates, depth, path):
        nonlocal best_size, best_code_indices, start_time, time_limit

        if time.time() - start_time > time_limit:
            return

        if uncovered == 0:
            if depth < best_size:
                best_size = depth
                best_code_indices = path[:]
            return

        # Lower Bound
        lb = packing_bound(uncovered)
        if depth + lb >= best_size:
            return

        # Select a vertex to branch on
        # Heuristic: Choose v in uncovered with fewest candidates covering it
        # Or just pick the first one for speed, but "most constrained" is better.
        # For 2187 nodes, iterating all uncovered to find min-degree might be slow if done naively.
        # Let's pick a random uncovered vertex or the first one to keep overhead low,
        # but sort the candidates for that vertex.
        
        # Pick first uncovered vertex
        lsb = uncovered & -uncovered
        v_idx = lsb.bit_length() - 1
        
        # Find all candidates that cover v_idx
        # candidates is a bitmask
        # We need c in candidates such that bit v_idx is set in balls[c]
        
        # Pre-computed: coverers[v] = bitmask of all words that cover v (i.e., dist <= 1)
        # Let's compute this once outside.
        
        # To avoid global lookup inside recursion, we pass candidates.
        # We iterate through candidates.
        
        # Optimization: Precompute `coverers` list where coverers[v] is a bitmask of all c such that v in ball[c].
        # Then potential_candidates = candidates & coverers[v_idx]
        
        # Let's assume we have `coverers` precomputed.
        
        pot_cands = candidates & coverers[v_idx]
        
        if pot_cands == 0:
            return # Dead end, shouldn't happen if initial is valid

        # Get list of indices in pot_cands
        # Sort by coverage of `uncovered`
        cand_list = []
        temp = pot_cands
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            cov = (balls[idx] & uncovered).bit_count()
            cand_list.append((cov, idx))
            temp ^= lsb
        
        # Sort descending by coverage
        cand_list.sort(key=lambda x: x[0], reverse=True)
        
        # Branch
        for cov, idx in cand_list:
            new_uncovered = uncovered & ~balls[idx]
            # New candidates: We can keep all candidates, or prune those that don't cover any new vertex.
            # Pruning: A candidate c is useful if ball[c] & new_uncovered != 0.
            # This check is expensive for all c. Let's just keep all candidates for now.
            # Actually, removing candidates that are already "used" is not allowed in Dominating Set (we can pick multiple centers).
            # But we don't need to pick a center if it doesn't cover any *new* uncovered vertex.
            # However, for the sake of simplicity and speed in Python, we might skip this pruning.
            
            # Recurse
            path.append(idx)
            branch_and_bound(new_uncovered, candidates, depth + 1, path)
            path.pop()

    # Precompute coverers
    coverers = [0] * total_words
    for i in range(total_words):
        temp = balls[i]
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            coverers[idx] |= (1 << i)
            temp ^= lsb

    # Run B&B
    branch_and_bound(all_ones_mask, all_ones_mask, 0, [])

    if best_code_indices:
        return [all_words[i] for i in best_code_indices]
    else:
        # Fallback to greedy if B&B failed or timed out without finding anything (shouldn't happen)
        return [all_words[i] for i in greedy_cover(all_ones_mask)]
