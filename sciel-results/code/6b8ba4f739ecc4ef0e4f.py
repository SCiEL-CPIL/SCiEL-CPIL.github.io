"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s limit
    
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    all_taps = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            all_taps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_taps = len(all_taps)
    
    # Map each position to the indices of taps it participates in
    pos_taps = [[] for _ in range(N)]
    for i, tap in enumerate(all_taps):
        for p in tap:
            pos_taps[p].append(i)
    
    def is_valid_coloring(colors):
        """Check if every 4-term progression uses at least 3 distinct colors."""
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True
    
    def get_distinct_count(colors):
        return len(set(colors))
    
    def count_violations(colors):
        """Count number of 4-term APs with fewer than 3 distinct colors."""
        violations = 0
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                violations += 1
        return violations
    
    def compute_energy(colors, weight=1000):
        """
        Compute energy: sum of squared violations + large weight * number of distinct colors.
        This allows temporarily invalid colorings during SA.
        """
        violations = 0
        for a, b, c, d in all_taps:
            distinct = len({colors[a], colors[b], colors[c], colors[d]})
            if distinct < 3:
                violations += 1
        num_colors = len(set(colors))
        return violations * violations + weight * num_colors
    
    def hamming_weight_coloring():
        """
        Deterministic construction using Hamming weight of binary representation.
        Position i gets color = popcount(i), which ranges from 0 to 7.
        """
        colors = [bin(i).count('1') for i in range(N)]
        if is_valid_coloring(colors):
            return colors
        return None
    
    def greedy_construct(max_colors):
        """
        Greedy left-to-right construction using bitmask for fast color selection.
        """
        colors = [-1] * N
        
        for pos in range(N):
            forbidden_mask = 0
            stuck = False
            
            for ti in pos_taps[pos]:
                a, b, c, d = all_taps[ti]
                pts = [a, b, c, d]
                others = [p for p in pts if p != pos]
                
                assigned_colors = []
                all_assigned = True
                for p in others:
                    if colors[p] == -1:
                        all_assigned = False
                        break
                    assigned_colors.append(colors[p])
                
                if not all_assigned:
                    continue
                
                s = set(assigned_colors)
                if len(s) == 1:
                    stuck = True
                    break
                elif len(s) == 2:
                    for c in s:
                        forbidden_mask |= (1 << c)
            
            if stuck:
                return None
            
            found = False
            for c in range(max_colors):
                if not (forbidden_mask & (1 << c)):
                    colors[pos] = c
                    found = True
                    break
            
            if not found:
                return None
        
        if not is_valid_coloring(colors):
            return None
        
        return colors
    
    def is_local_valid(pos, c, new_colors):
        """Check if assigning c to pos is valid with respect to all taps where pos is involved."""
        for ti in pos_taps[pos]:
            a, b, c_tap, d_tap = all_taps[ti]
            pts = [a, b, c_tap, d_tap]
            others = [p for p in pts if p != pos]
            
            vals = [c]
            for p in others:
                vals.append(new_colors[p])
            
            if len(set(vals)) < 3:
                return False
        return True
    
    def simulated_annealing(initial_colors, max_colors, time_limit=20.0):
        """
        Simulated annealing to find valid coloring with at most max_colors colors.
        Energy = violations^2 + 1000 * num_distinct_colors
        Allows temporarily invalid states to escape local minima.
        """
        start_t = time.time()
        colors = initial_colors[:]
        current_energy = compute_energy(colors)
        best_colors = colors[:]
        best_energy = current_energy
        
        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        num_iterations = 50000
        cooling_rate = (T_end / T_start) ** (1.0 / num_iterations)
        
        # Precompute tap indices for fast violation counting
        # For each position, store which taps it's in and the other 3 positions
        pos_tap_info = [[] for _ in range(N)]
        for ti, tap in enumerate(all_taps):
            for p in tap:
                others = [q for q in tap if q != p]
                pos_tap_info[p].append((ti, others))
        
        def compute_violation_delta(pos, old_color, new_color, colors):
            """Compute change in violation count when changing pos from old_color to new_color."""
            delta = 0
            for ti, others in pos_tap_info[pos]:
                # Old state
                old_vals = [old_color] + [colors[p] for p in others]
                old_distinct = len(set(old_vals))
                old_viol = 1 if old_distinct < 3 else 0
                
                # New state
                new_vals = [new_color] + [colors[p] for p in others]
                new_distinct = len(set(new_vals))
                new_viol = 1 if new_distinct < 3 else 0
                
                delta += new_viol - old_viol
            return delta
        
        T = T_start
        for it in range(num_iterations):
            if time.time() - start_t > time_limit:
                break
            
            # Choose random position and new color
            pos = random.randint(0, N - 1)
            new_color = random.randint(0, max_colors - 1)
            old_color = colors[pos]
            
            if new_color == old_color:
                T *= cooling_rate
                continue
            
            # Compute energy delta
            old_viol = count_violations(colors)
            colors[pos] = new_color
            new_viol = count_violations(colors)
            colors[pos] = old_color
            
            old_num_colors = len(set(colors))
            colors[pos] = new_color
            new_num_colors = len(set(colors))
            colors[pos] = old_color
            
            old_energy = old_viol * old_viol + 1000 * old_num_colors
            new_energy = new_viol * new_viol + 1000 * new_num_colors
            delta_E = new_energy - old_energy
            
            # Accept or reject
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                colors[pos] = new_color
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
                    # If we found a valid coloring, check if we can reduce colors
                    if best_energy == 1000 * len(set(best_colors)):
                        # Valid coloring found
                        pass
            
            T *= cooling_rate
        
        return best_colors, best_energy
    
    def try_remove_color(colors, target_color, time_limit=3.0):
        """
        Attempts to remove target_color by recoloring its occurrences to other existing colors.
        """
        start_t = time.time()
        positions = [i for i, c in enumerate(colors) if c == target_color]
        if not positions:
            return None
        
        available_colors = [c for c in set(colors) if c != target_color]
        if not available_colors:
            return None
        
        new_colors = colors[:]
        positions.sort(key=lambda i: len(pos_taps[i]), reverse=True)
        
        def backtrack(idx):
            if time.time() - start_t > time_limit:
                return False
            
            if idx == len(positions):
                return True
            
            pos = positions[idx]
            
            for c in available_colors:
                if c == new_colors[pos]:
                    continue
                
                new_colors[pos] = c
                
                if is_local_valid(pos, c, new_colors):
                    if backtrack(idx + 1):
                        return True
                
                new_colors[pos] = target_color
            
            return False
        
        if backtrack(0):
            return new_colors
        return None
    
    def try_merge_pair_sa(colors, color1, color2, time_limit=5.0):
        """
        Use SA to merge color2 into color1 or other colors.
        """
        start_t = time.time()
        
        # Create initial state where color2 positions are "free"
        positions_c2 = [i for i, c in enumerate(colors) if c == color2]
        if not positions_c2:
            return None
        
        # Use SA with max_colors = number of distinct colors (we want to eliminate color2)
        distinct_colors = set(colors)
        max_c = max(distinct_colors) + 1
        
        # Initialize: keep current colors
        init_colors = colors[:]
        
        # Run SA with constraint that we prefer to eliminate color2
        # We'll use a modified energy that heavily penalizes having color2
        result_colors, result_energy = simulated_annealing(init_colors, max_c, time_limit)
        
        # Check if color2 was eliminated
        if color2 not in set(result_colors):
            if is_valid_coloring(result_colors):
                return result_colors
        
        return None
    
    def local_search_polish(colors, time_limit=10.0):
        """
        Try to reduce color count by merging pairs of colors.
        """
        start_t = time.time()
        current = colors[:]
        
        while time.time() - start_t < time_limit:
            colors_in_use = list(set(current))
            if len(colors_in_use) <= 2:
                break
            
            colors_in_use.sort(key=lambda c: current.count(c))
            
            improved = False
            for i in range(len(colors_in_use)):
                for j in range(i + 1, len(colors_in_use)):
                    if time.time() - start_t >= time_limit:
                        break
                    
                    c1, c2 = colors_in_use[i], colors_in_use[j]
                    
                    # Try to merge c2 into c1 using SA
                    new_colors = try_merge_pair_sa(current, c1, c2, time_limit=3.0)
                    if new_colors is not None and len(set(new_colors)) < len(set(current)):
                        current = new_colors
                        improved = True
                        break
                    
                    # Try to merge c1 into c2
                    new_colors = try_merge_pair_sa(current, c2, c1, time_limit=3.0)
                    if new_colors is not None and len(set(new_colors)) < len(set(current)):
                        current = new_colors
                        improved = True
                        break
                
                if improved or time.time() - start_t >= time_limit:
                    break
            
            if not improved:
                break
        
        return current
    
    def color_count(colors):
        return len(set(colors))
    
    # 1. Generate initial coloring
    initial_colors = hamming_weight_coloring()
    
    if initial_colors is None:
        for k in range(3, 15):
            initial_colors = greedy_construct(k)
            if initial_colors is not None:
                break
    
    if initial_colors is None:
        initial_colors = [i % 4 for i in range(N)]
        if not is_valid_coloring(initial_colors):
            initial_colors = list(range(N))
    
    current_colors = initial_colors
    current_count = get_distinct_count(current_colors)
    
    # 2. Use SA to try to find valid coloring with fewer colors
    if time.time() < deadline - 20:
        # Try SA with decreasing max_colors
        for max_c in range(3, min(current_count + 1, 8)):
            if time.time() > deadline - 15:
                break
            
            # Create random initial coloring with max_c colors
            sa_init = [random.randint(0, max_c - 1) for _ in range(N)]
            
            sa_result, sa_energy = simulated_annealing(sa_init, max_c, time_limit=15.0)
            
            # Check if we got a valid coloring
            if is_valid_coloring(sa_result):
                sa_count = len(set(sa_result))
                if sa_count < current_count:
                    current_colors = sa_result
                    current_count = sa_count
                    break
                elif sa_count == current_count:
                    # Same count, but maybe better structure for further reduction
                    if count_violations(sa_result) == 0:
                        current_colors = sa_result
    
    # 3. Local search to reduce color count further
    if time.time() < deadline - 10:
        # Try removing individual colors
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            colors_in_use = list(set(current_colors))
            colors_in_use.sort(key=lambda c: current_colors.count(c))
            
            for c in colors_in_use:
                if time.time() > deadline - 8:
                    break
                
                new_colors = try_remove_color(current_colors, c, time_limit=3.0)
                if new_colors is not None:
                    new_count = get_distinct_count(new_colors)
                    if new_count < current_count:
                        current_colors = new_colors
                        current_count = new_count
                        improved = True
                        break
        
        # Try merging pairs with SA
        if time.time() < deadline - 5:
            polished = local_search_polish(current_colors, time_limit=8.0)
            if get_distinct_count(polished) < current_count:
                current_colors = polished
                current_count = get_distinct_count(current_colors)
    
    # Final validation
    if not is_valid_coloring(current_colors):
        current_colors = [i % 4 for i in range(N)]
    
    return current_colors
# EVOLVE-BLOCK-END
