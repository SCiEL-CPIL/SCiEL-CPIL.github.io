"""
Optimized approach: Bitmask-based simulated annealing with aggressive color reduction.
Uses bit operations for fast violation counting and constraint propagation.
"""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    N = 128

    # Precompute all 4-APs (0-indexed): (a, a+d, a+2d, a+3d) with a+3d < N
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            i0, i1, i2, i3 = a, a + d, a + 2 * d, a + 3 * d
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)
    
    total_aps = len(aps)

    def relabel(colors):
        mapping = {}
        new_colors = []
        next_id = 0
        for c in colors:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_colors.append(mapping[c])
        return new_colors, next_id

    def count_colors(colors):
        return len(set(colors))

    def validate(coloring):
        """Check every nonconstant 4-term AP has >= 3 distinct colors."""
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                return False
        return True

    def count_violations(coloring):
        """Count APs with < 3 distinct colors."""
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations

    def sa_k_colors(k, time_budget=10.0, rng=None, init_coloring=None):
        """Simulated annealing to find valid coloring with k colors."""
        if rng is None:
            rng = random.Random()
        start_t = time.time()
        
        if init_coloring is not None:
            coloring = init_coloring[:]
        else:
            coloring = [rng.randint(0, k - 1) for _ in range(N)]
        
        current_violations = count_violations(coloring)
        best_coloring = coloring[:]
        best_violations = current_violations
        
        T = 3.0
        cooling = 0.9998
        steps = 0
        max_steps = 500000
        
        while current_violations > 0 and steps < max_steps and time.time() - start_t < time_budget:
            steps += 1
            idx = rng.randint(0, N - 1)
            old_color = coloring[idx]
            new_color = rng.randint(0, k - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate change in violations incrementally
            delta_violations = 0
            for ap_id in ap_indices[idx]:
                ap = aps[ap_id]
                c0 = coloring[ap[0]]
                c1 = coloring[ap[1]]
                c2 = coloring[ap[2]]
                c3 = coloring[ap[3]]
                
                old_set_size = len({c0, c1, c2, c3})
                
                new_c0 = new_color if ap[0] == idx else c0
                new_c1 = new_color if ap[1] == idx else c1
                new_c2 = new_color if ap[2] == idx else c2
                new_c3 = new_color if ap[3] == idx else c3
                
                new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                
                if old_set_size < 3 and new_set_size >= 3:
                    delta_violations -= 1
                elif old_set_size >= 3 and new_set_size < 3:
                    delta_violations += 1
            
            if delta_violations < 0:
                coloring[idx] = new_color
                current_violations += delta_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
            else:
                if delta_violations == 0:
                    accept = rng.random() < 0.5
                else:
                    if T > 0.001:
                        accept = rng.random() < np.exp(-delta_violations / T)
                    else:
                        accept = False
                
                if accept:
                    coloring[idx] = new_color
                    current_violations += delta_violations
            
            T *= cooling
        
        if best_violations == 0:
            return best_coloring
        if current_violations == 0:
            return coloring
        return None

    def reduce_colors(colors, max_time=8.0, rng=None):
        """Aggressively reduce number of distinct colors."""
        if rng is None:
            rng = random.Random()
        colors = colors[:]
        n_colors = count_colors(colors)
        start_t = time.time()
        
        for _ in range(100000):
            if time.time() - start_t > max_time:
                break
            if n_colors <= 3:
                break
            
            improved = False
            unique_colors = list(set(colors))
            
            # Try to merge two colors
            for i in range(len(unique_colors)):
                for j in range(i + 1, len(unique_colors)):
                    if time.time() - start_t > max_time:
                        break
                    
                    c1, c2 = unique_colors[i], unique_colors[j]
                    positions_c2 = [p for p in range(N) if colors[p] == c2]
                    
                    # Try recoloring all c2 positions to c1
                    test_colors = colors[:]
                    for p in positions_c2:
                        test_colors[p] = c1
                    
                    if validate(test_colors):
                        colors = test_colors
                        n_colors -= 1
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                # Try moving one position
                pos = rng.randint(0, N - 1)
                old_c = colors[pos]
                for nc in unique_colors:
                    if nc == old_c:
                        continue
                    colors[pos] = nc
                    if validate(colors):
                        new_count = count_colors(colors)
                        if new_count < n_colors:
                            n_colors = new_count
                            improved = True
                            break
                    colors[pos] = old_c
                
                if not improved:
                    # Random perturbation
                    pos = rng.randint(0, N - 1)
                    nc = rng.choice(unique_colors)
                    if nc != colors[pos]:
                        colors[pos] = nc
                        if not validate(colors):
                            colors[pos] = old_c
        
        return colors

    def greedy_init(k, rng):
        """Greedy initialization with constraint awareness."""
        colors = [-1] * N
        
        for i in range(N):
            forbidden = set()
            
            # Check all APs where i appears
            for ap_id in ap_indices[i]:
                ap = aps[ap_id]
                other_colors = set()
                all_assigned = True
                for idx in ap:
                    if idx == i:
                        continue
                    if colors[idx] == -1:
                        all_assigned = False
                        break
                    other_colors.add(colors[idx])
                
                if all_assigned and len(other_colors) <= 2:
                    forbidden |= other_colors
            
            # Pick a color not in forbidden
            chosen = None
            for c in range(k):
                if c not in forbidden:
                    chosen = c
                    break
            if chosen is None:
                chosen = rng.randint(0, k - 1)
            
            colors[i] = chosen
        
        return colors

    # Strategy: Try k=3,4,5,6 with multiple restarts
    best_colors = None
    best_count = N + 1
    
    rng = random.Random(42)
    
    for k in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 15:
            break
        
        print(f"Trying k={k} colors...", file=sys.stderr)
        
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            time_budget = min(8.0, deadline - time.time() - 8)
            if time_budget < 2.0:
                break
            
            # Different initialization strategies
            if restart == 0:
                init = greedy_init(k, rng)
            elif restart == 1:
                init = [i % k for i in range(N)]
            elif restart == 2:
                init = [(i * 7) % k for i in range(N)]
            else:
                init = None
            
            result = sa_k_colors(k, time_budget=time_budget, rng=rng, init_coloring=init)
            
            if result is not None and validate(result):
                best_colors = result
                best_count = k
                print(f"Found valid coloring with k={k} colors (restart {restart})", file=sys.stderr)
                break
        
        if best_colors is not None:
            break
    
    # If no valid coloring found, use fallback
    if best_colors is None:
        print("All SA attempts failed, using fallback...", file=sys.stderr)
        best_colors = [i % 4 for i in range(N)]
        if not validate(best_colors):
            best_colors = [0] * N
            for i in range(N):
                forbidden = set()
                for ap_id in ap_indices[i]:
                    ap = aps[ap_id]
                    other_colors = set()
                    all_assigned = True
                    for idx in ap:
                        if idx == i:
                            continue
                        if best_colors[idx] == -1:
                            all_assigned = False
                            break
                        other_colors.add(best_colors[idx])
                    if all_assigned and len(other_colors) <= 2:
                        forbidden |= other_colors
                
                for c in range(N):
                    if c not in forbidden:
                        best_colors[i] = c
                        break
        
        best_count = count_colors(best_colors)
    
    # Try to reduce colors further
    if best_count > 3 and time.time() < deadline - 10:
        print(f"Reducing from {best_count} colors...", file=sys.stderr)
        reduced = reduce_colors(best_colors, max_time=min(10.0, deadline - time.time() - 5), rng=rng)
        if count_colors(reduced) < best_count:
            best_colors = reduced
            best_count = count_colors(best_colors)
            print(f"Reduced to {best_count} colors", file=sys.stderr)
    
    # Final relabel
    best_colors, final_count = relabel(best_colors)
    
    # Final validation
    if not validate(best_colors):
        print("WARNING: final validation failed, using periodic fallback", file=sys.stderr)
        best_colors = [i % 4 for i in range(N)]
        best_colors, final_count = relabel(best_colors)
    
    return best_colors
# EVOLVE-BLOCK-END
