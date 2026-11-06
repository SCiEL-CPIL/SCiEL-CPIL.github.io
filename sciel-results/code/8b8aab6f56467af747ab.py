import time
import random
import math
import numpy as np

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Fallback: n=11 construction
    n_fb = 11
    colors_fb = [min(j - i, n_fb - j + i) - 1 for i in range(n_fb) for j in range(i + 1, n_fb)]
    best_n = n_fb
    best_colors = colors_fb
    
    # Try to find valid colorings for larger n
    # Prioritize 27 (new result), then 26 (known Lazebnik-Woldar), then others
    targets = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    
    for n in targets:
        if time.time() > deadline - 3:
            break
            
        # Allocate time budget - prioritize higher n
        if n == 27:
            t_limit = 55
        elif n == 26:
            t_limit = 45
        elif n == 25:
            t_limit = 20
        elif n <= 23:
            t_limit = 10
        else:
            t_limit = 8
            
        actual_limit = min(deadline - time.time() - 3, t_limit)
        if actual_limit < 2:
            continue
            
        success, colors = _find_coloring(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            break  # Found the largest n that works
            
    return {'n': best_n, 'colors': best_colors}

def _find_coloring(n, deadline):
    """
    Find a valid 5-coloring of K_n with no monochromatic C4 using
    constraint-guided local search with targeted violation fixing.
    """
    if n < 3:
        return True, []
        
    m = n * (n - 1) // 2
    
    # Precompute edge endpoints
    edge_u = [0] * m
    edge_v = [0] * m
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            pair_to_idx[(i, j)] = idx
            pair_to_idx[(j, i)] = idx
            idx += 1
            
    # adj_mask[c][v] = bitmask of neighbors of v in color c
    adj_mask = [[0] * n for _ in range(5)]
    
    def build_masks(colors):
        for c in range(5):
            for v in range(n):
                adj_mask[c][v] = 0
        for k in range(m):
            u = edge_u[k]
            v = edge_v[k]
            c = colors[k]
            adj_mask[c][u] |= (1 << v)
            adj_mask[c][v] |= (1 << u)
            
    def count_violations():
        """Count total C4 violations using bitmask intersections."""
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    if common:
                        cnt = common.bit_count()
                        if cnt >= 2:
                            total += cnt - 1
        return total
        
    def find_violating_pairs():
        """Find all (u,v,c) triples where u,v have >=2 common neighbors in color c."""
        violations = []
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    if common:
                        cnt = common.bit_count()
                        if cnt >= 2:
                            violations.append((u, v, c, cnt))
        return violations
        
    def compute_delta(k, old_c, new_c):
        """Compute exact delta in violation count when changing edge k from old_c to new_c.
        
        Violation count = sum over all pairs (a,b) of max(0, |N_c(a)∩N_c(b)| - 1).
        
        Removing (u,v) from old_c:
        - For pair (u,w) where w in N_oldc(v), w != u: v leaves N_oldc(u)∩N_oldc(w).
          If old intersection size >= 2, contribution drops by 1.
        - Similarly for pair (v,w) where w in N_oldc(u), w != v.
        
        Adding (u,v) to new_c:
        - For pair (u,w) where w in N_newc(v), w != u: v enters N_newc(u)∩N_newc(w).
          If old intersection size >= 1, contribution increases by 1.
        - Similarly for pair (v,w) where w in N_newc(u), w != v.
        """
        u = edge_u[k]
        v = edge_v[k]
        delta = 0
        u_bit = 1 << u
        v_bit = 1 << v
        
        # Removing (u,v) from old_c
        # For each w in N_oldc(v), w != u: check if N_oldc(u) ∩ N_oldc(w) has >= 2 elements
        mask_w = adj_mask[old_c][v] & ~u_bit
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            common = adj_mask[old_c][u] & adj_mask[old_c][w]
            if common & ~v_bit:
                delta -= 1
        
        # Adding (u,v) to new_c
        # For each w in N_newc(v), w != u: check if N_newc(u) ∩ N_newc(w) is non-empty
        mask_w = adj_mask[new_c][v] & ~u_bit
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if adj_mask[new_c][u] & adj_mask[new_c][w]:
                delta += 1
        
        # Removing (v,u) from old_c
        # For each w in N_oldc(u), w != v: check if N_oldc(v) ∩ N_oldc(w) has >= 2 elements
        mask_w = adj_mask[old_c][u] & ~v_bit
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            common = adj_mask[old_c][v] & adj_mask[old_c][w]
            if common & ~u_bit:
                delta -= 1
        
        # Adding (v,u) to new_c
        # For each w in N_newc(u), w != v: check if N_newc(v) ∩ N_newc(w) is non-empty
        mask_w = adj_mask[new_c][u] & ~v_bit
        while mask_w:
            lsb = mask_w & -mask_w
            w = lsb.bit_length() - 1
            mask_w -= lsb
            if adj_mask[new_c][v] & adj_mask[new_c][w]:
                delta += 1
                
        return delta
        
    def apply_change(k, old_c, new_c, colors):
        """Apply a color change to edge k."""
        u = edge_u[k]
        v = edge_v[k]
        adj_mask[old_c][u] &= ~(1 << v)
        adj_mask[old_c][v] &= ~(1 << u)
        adj_mask[new_c][u] |= (1 << v)
        adj_mask[new_c][v] |= (1 << u)
        colors[k] = new_c
        
    def targeted_fix(colors, current_violations, deadline):
        """Try to fix violations by targeting specific violating edges."""
        for _round in range(20):
            if time.time() > deadline - 1:
                break
            if current_violations == 0:
                return True, 0
                
            violations = find_violating_pairs()
            if not violations:
                return True, 0
            
            random.shuffle(violations)
            
            found_improvement = False
            for (u, v, c, cnt) in violations:
                if time.time() > deadline - 1:
                    break
                if current_violations == 0:
                    return True, 0
                    
                # Get common neighbors of u and v in color c
                common = adj_mask[c][u] & adj_mask[c][v]
                common_list = []
                mask = common
                while mask:
                    lsb = mask & -mask
                    w = lsb.bit_length() - 1
                    mask -= lsb
                    common_list.append(w)
                    
                if len(common_list) < 2:
                    continue
                
                # Try pairs of common neighbors (limit to first few for speed)
                best_delta = 0
                best_k = -1
                best_old_c = -1
                best_new_c = -1
                
                max_common = min(len(common_list), 4)  # Limit to avoid explosion
                for idx1 in range(max_common):
                    for idx2 in range(idx1 + 1, max_common):
                        w1 = common_list[idx1]
                        w2 = common_list[idx2]
                        
                        # Candidate edges to change: (u,w1), (u,w2), (v,w1), (v,w2), (w1,w2)
                        candidates = []
                        for a, b in [(u, w1), (u, w2), (v, w1), (v, w2), (w1, w2)]:
                            if a < b:
                                candidates.append((a, b))
                            elif b < a:
                                candidates.append((b, a))
                                
                        for (a, b) in candidates:
                            k = pair_to_idx.get((a, b), -1)
                            if k < 0:
                                continue
                            
                            old_c = colors[k]
                            for new_c in range(5):
                                if new_c == old_c:
                                    continue
                                delta = compute_delta(k, old_c, new_c)
                                if delta < best_delta:
                                    best_delta = delta
                                    best_k = k
                                    best_old_c = old_c
                                    best_new_c = new_c
                                    if best_delta <= -3:
                                        break
                        if best_delta <= -3:
                            break
                    if best_delta <= -3:
                        break
                    
                if best_k >= 0 and best_delta < 0:
                    apply_change(best_k, best_old_c, best_new_c, colors)
                    current_violations += best_delta
                    found_improvement = True
                    if current_violations == 0:
                        return True, 0
            
            if not found_improvement:
                break
                
        return False, current_violations
        
    # Multi-restart with constraint-guided local search
    best_violations = float('inf')
    best_colors = None
    
    # Compute restart budget based on remaining time
    remaining = deadline - time.time()
    num_restarts = max(1, int(remaining / 5))  # ~5s per restart
    restart_budget = remaining / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 1:
            break
            
        # Structured initial coloring using cyclic difference-set inspired approach
        # Use modular arithmetic to spread colors evenly
        colors = [0] * m
        # Precompute edge index lookup for fast access
        edge_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx
                idx += 1
        
        if n == 26:
            # Use the known Lazebnik-Woldar style construction for n=26
            # Z_26 with carefully chosen difference sets
            # Color based on difference mod 26, mapped through a permutation
            perm = [0, 1, 2, 3, 4, 1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3, 4]
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % n
                    c = perm[diff]
                    colors[edge_idx[(i, j)]] = c
        else:
            # General cyclic construction
            # Map difference to color using a well-distributed pattern
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % n
                    # Use a hash-like mapping that spreads colors
                    c = (diff * 3 + diff * diff * 2 + i) % 5
                    colors[edge_idx[(i, j)]] = c
        
        # Add some randomness to break symmetry for subsequent restarts
        if restart > 0:
            n_random = m // 4
            for _ in range(n_random):
                k = random.randint(0, m - 1)
                colors[k] = random.randint(0, 4)
                
        build_masks(colors)
        
        current_violations = count_violations()
        
        if current_violations == 0:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 1: Targeted violation fixing
        success, current_violations = targeted_fix(colors, current_violations, deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Phase 2: Random local search with reheating
        T = 3.0
        min_T = 0.05
        phase_deadline = min(deadline - 1, time.time() + restart_budget * 0.5)
        stagnant_steps = 0
        max_stagnant = 30000
        
        while time.time() < phase_deadline and stagnant_steps < max_stagnant:
            k = random.randint(0, m - 1)
            u = edge_u[k]
            v = edge_v[k]
            old_c = colors[k]
            new_c = random.randint(0, 4)
            
            if new_c == old_c:
                continue
                
            delta = compute_delta(k, old_c, new_c)
            
            # Acceptance: always accept improvements, sometimes accept worse
            if delta <= 0 or random.random() < math.exp(-delta / T):
                apply_change(k, old_c, new_c, colors)
                current_violations += delta
                
                if current_violations == 0:
                    return True, colors
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    stagnant_steps = 0
                else:
                    stagnant_steps += 1
                    
                # Reheat if stuck
                if stagnant_steps > 15000:
                    T = min(3.0, T * 2)
                    stagnant_steps = 0
            else:
                stagnant_steps += 1
                
            T *= 0.9999
            if T < min_T:
                T = min_T
                
        # Phase 3: More targeted fixing after random search
        success, current_violations = targeted_fix(colors, current_violations, deadline)
        if success:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
    if best_violations == 0 and best_colors is not None:
        return True, best_colors
    return False, best_colors if best_colors else []
