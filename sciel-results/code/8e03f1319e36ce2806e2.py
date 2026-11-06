import time
import sys
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute all 4-cycles (3 per 4-vertex subset)
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        cycles.append(((a,c), (c,b), (b,d), (d,a)))

    # Convert to edge indices
    cycle_indices = np.array([tuple(edge_idx[u][v] for u, v in c) for c in cycles], dtype=np.int32)

    # For each edge, which cycles does it belong to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, c_idx in enumerate(cycle_indices):
        for e_idx in c_idx:
            edge_to_cycles[e_idx].append(i)

    def get_seed_colors(seed=0):
        """Generate a valid coloring via recursive partitioning with randomization."""
        colors = [0] * NUM_EDGES
        color_id = 0
        rng = random.Random(seed)
        
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        
        process_partition(list(range(N)))
        return colors

    def check_validity(colors):
        """Check if coloring is valid (no rainbow 4-cycle)."""
        c = np.array(colors, dtype=np.int32)
        for ci in cycle_indices:
            c0, c1, c2, c3 = c[ci[0]], c[ci[1]], c[ci[2]], c[ci[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # Try multiple initial seeds and pick the best
    best_colors = None
    best_score = 0
    for s in range(5):
        if time.time() > deadline - 10.0:
            break
        colors = get_seed_colors(s)
        if check_validity(colors):
            sc = count_colors(colors)
            if sc > best_score:
                best_score = sc
                best_colors = colors.copy()

    if best_colors is None:
        best_colors = get_seed_colors(0)
        best_score = count_colors(best_colors)

    current_colors = best_colors.copy()
    current_distinct = best_score

    # Track frequency
    freq = [0] * MAX_COLORS
    for c in current_colors:
        freq[c] += 1

    def try_single_edge_flip(colors, freq, e, new_color):
        """Check if flipping edge e to new_color is valid. Returns True if valid."""
        old_color = colors[e]
        if new_color == old_color:
            return True
        valid = True
        for c_idx in edge_to_cycles[e]:
            ci = cycle_indices[c_idx]
            c0 = colors[ci[0]] if ci[0] != e else new_color
            c1 = colors[ci[1]] if ci[1] != e else new_color
            c2 = colors[ci[2]] if ci[2] != e else new_color
            c3 = colors[ci[3]] if ci[3] != e else new_color
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                valid = False
                break
        return valid

    def apply_single_edge_flip(colors, freq, e, new_color):
        """Apply flip and update freq. Returns (old_color, distinct_change)."""
        old_color = colors[e]
        if new_color == old_color:
            return old_color, 0
        old_freq = freq[old_color]
        new_freq = freq[new_color]
        freq[old_color] -= 1
        freq[new_color] += 1
        colors[e] = new_color
        delta = 0
        if old_freq == 1:
            delta -= 1
        if new_freq == 0:
            delta += 1
        return old_color, delta

    def undo_single_edge_flip(colors, freq, e, old_color, new_color):
        """Undo flip and update freq."""
        if old_color == new_color:
            return
        freq[old_color] += 1
        freq[new_color] -= 1
        colors[e] = old_color

    def local_search(colors, freq, distinct, max_steps, seed):
        """Perform local search with single-edge flips."""
        rng = random.Random(seed)
        best_local = distinct
        best_local_colors = colors.copy()
        T = 3.0
        T_min = 0.1
        cooling = 0.9999
        no_improve = 0
        max_no_improve = 20000

        for _ in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            e = rng.randint(0, NUM_EDGES - 1)
            old_color = colors[e]

            # Choose new color: prefer unused or rare
            if rng.random() < 0.5:
                unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
                if unused:
                    new_color = rng.choice(unused)
                else:
                    new_color = rng.randint(0, MAX_COLORS - 1)
            else:
                new_color = rng.randint(0, MAX_COLORS - 1)

            if new_color == old_color:
                continue

            if not try_single_edge_flip(colors, freq, e, new_color):
                no_improve += 1
                continue

            old_color_val, delta = apply_single_edge_flip(colors, freq, e, new_color)
            new_distinct = distinct + delta

            if new_distinct >= distinct:
                accept = True
            else:
                delta_val = new_distinct - distinct
                accept = rng.random() < np.exp(delta_val / T)

            if accept:
                distinct = new_distinct
                if new_distinct > best_local:
                    best_local = new_distinct
                    best_local_colors = colors.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                undo_single_edge_flip(colors, freq, e, old_color_val, new_color)
                no_improve += 1

            T *= cooling
            if T < T_min:
                T = T_min

            if no_improve > max_no_improve:
                break

        return best_local, best_local_colors

    # Multi-start local search with restarts
    overall_best = best_score
    overall_best_colors = current_colors.copy()
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Start from current best or a fresh seed
        if r == 0:
            start_colors = current_colors.copy()
            start_freq = freq.copy()
            start_distinct = current_distinct
        else:
            # Perturb current best
            start_colors = overall_best_colors.copy()
            start_freq = [0] * MAX_COLORS
            for c in start_colors:
                start_freq[c] += 1
            start_distinct = count_colors(start_colors)
            
            # Apply some random valid perturbations
            rng_perturb = random.Random(r * 1000 + 42)
            for _ in range(10):
                pe = rng_perturb.randint(0, NUM_EDGES - 1)
                pc = rng_perturb.randint(0, MAX_COLORS - 1)
                if pc != start_colors[pe] and try_single_edge_flip(start_colors, start_freq, pe, pc):
                    old_c = start_colors[pe]
                    apply_single_edge_flip(start_colors, start_freq, pe, pc)
        
        # Run local search
        local_best, local_colors = local_search(start_colors, start_freq, start_distinct, 200000, r * 777 + 123)
        
        if local_best > overall_best:
            overall_best = local_best
            overall_best_colors = local_colors.copy()
            # Update current state for next restart
            current_colors = local_colors.copy()
            current_distinct = local_best
            freq = [0] * MAX_COLORS
            for c in current_colors:
                freq[c] += 1
    
    return overall_best_colors
