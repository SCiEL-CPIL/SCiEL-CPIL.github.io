import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    positions = list(range(1, n + 1))
    
    # Precompute all nonconstant 4-term APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n):
            if a + 3 * d <= n:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
            else:
                break
    
    num_aps = len(aps)
    
    # For each position, list of AP indices it belongs to
    pos_to_aps = {i: [] for i in positions}
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Precompute for fast constraint checking: for each AP, store positions as tuple
    ap_positions = [tuple(ap) for ap in aps]
    
    # For incremental checking: for each position, which APs does it appear in and at what index?
    # pos_ap_info[pos] = list of (ap_idx, position_in_ap)
    pos_ap_info = {i: [] for i in positions}
    for idx, ap in enumerate(aps):
        for pos_idx, pos in enumerate(ap):
            pos_ap_info[pos].append((idx, pos_idx))
    
    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        cnt = 0
        for ap in ap_positions:
            if len(set(colors[p] for p in ap)) < 3:
                cnt += 1
        return cnt
    
    def compute_color_usage_penalty(colors, k, temperature=2.0):
        """
        Smooth surrogate penalty for number of distinct colors used.
        Uses log-sum-exp style: for each color c, compute soft count,
        then penalty encourages fewer colors with non-zero soft count.
        """
        # For each color in 0..k-1, compute a soft indicator of whether it's used
        # soft_use[c] = exp(sum_{p: colors[p]==c} 1/temperature) / (k * exp(1/temperature))
        # But simpler: penalty = sum_c softmax_indicator
        # Even simpler: use a smooth count of distinct colors
        # Let's use: penalty = sum_{c=0}^{k-1} (1 - exp(-count_c / temperature))
        # This is 0 when count_c=0, approaches 1 when count_c is large
        counts = [0] * k
        for p in positions:
            c = colors[p]
            if 0 <= c < k:
                counts[c] += 1
        penalty = 0.0
        for c in range(k):
            if counts[c] > 0:
                penalty += 1.0 - math.exp(-counts[c] / temperature)
        return penalty
    
    def try_local_search(k, max_iters=80000, seed=0, use_penalty=False):
        rng = random.Random(seed)
        
        # Greedy initialization with lookahead
        colors = {}
        for p in positions:
            best_c = None
            best_v = float('inf')
            for c in range(k):
                colors[p] = c
                v = 0
                for ap_i, _ in pos_ap_info[p]:
                    ap = ap_positions[ap_i]
                    vals = set()
                    for q in ap:
                        if q in colors:
                            vals.add(colors[q])
                    if len(vals) < 3:
                        v += 1
                if v < best_v:
                    best_v = v
                    best_c = c
                if v == 0:
                    break
            colors[p] = best_c
        
        # Compute initial violations
        violated = [False] * num_aps
        for idx in range(num_aps):
            ap = ap_positions[idx]
            vals = set(colors[p] for p in ap)
            violated[idx] = len(vals) < 3
        num_violations = sum(violated)
        
        if num_violations == 0:
            return colors
        
        # Track best solution found
        best_violations = num_violations
        best_colors = dict(colors)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return best_colors if best_violations == 0 else None
            
            if num_violations == 0:
                return colors
            
            # Pick a random violated AP
            v_indices = [i for i in range(num_aps) if violated[i]]
            if not v_indices:
                return colors
            ap_idx = v_indices[rng.randint(0, len(v_indices) - 1)]
            ap = ap_positions[ap_idx]
            
            # Try to fix by recoloring one position in this AP
            pos_order = list(ap)
            rng.shuffle(pos_order)
            
            improved = False
            for pos in pos_order:
                old_color = colors[pos]
                for c in range(k):
                    if c == old_color:
                        continue
                    colors[pos] = c
                    # Update violations for APs containing pos
                    delta = 0
                    for ap_i, _ in pos_ap_info[pos]:
                        was_v = violated[ap_i]
                        vals = set(colors[q] for q in ap_positions[ap_i])
                        is_v = len(vals) < 3
                        violated[ap_i] = is_v
                        if is_v and not was_v:
                            delta += 1
                        elif not is_v and was_v:
                            delta -= 1
                    new_total = num_violations + delta
                    if new_total < num_violations:
                        num_violations = new_total
                        improved = True
                        break
                    else:
                        # Revert
                        colors[pos] = old_color
                        for ap_i, _ in pos_ap_info[pos]:
                            vals = set(colors[q] for q in ap_positions[ap_i])
                            violated[ap_i] = len(vals) < 3
                        num_violations = sum(violated)
                        break
                if improved:
                    break
            
            if not improved:
                # Random kick: try multiple random changes
                for _ in range(3):
                    pos = pos_order[rng.randint(0, len(pos_order) - 1)]
                    new_c = rng.randint(0, k - 1)
                    colors[pos] = new_c
                    for ap_i, _ in pos_ap_info[pos]:
                        vals = set(colors[q] for q in ap_positions[ap_i])
                        violated[ap_i] = len(vals) < 3
                    num_violations = sum(violated)
                    if num_violations == 0:
                        return colors
                    if num_violations < best_violations:
                        best_violations = num_violations
                        best_colors = dict(colors)
        
        return None
    
    # Try k=3 with multiple seeds
    for seed in range(200):
        if time.time() > deadline:
            break
        result = try_local_search(3, max_iters=30000, seed=seed)
        if result is not None:
            return [result[p] for p in positions]
    
    # Try k=4
    for seed in range(100):
        if time.time() > deadline:
            break
        result = try_local_search(4, max_iters=30000, seed=seed)
        if result is not None:
            return [result[p] for p in positions]
    
    # Try k=5
    for seed in range(50):
        if time.time() > deadline:
            break
        result = try_local_search(5, max_iters=30000, seed=seed)
        if result is not None:
            return [result[p] for p in positions]
    
    # Fallback: deterministic greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3 + 1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2 + 1))):
                colors.append(color)
                break
    return colors
