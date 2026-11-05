import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}

    # Precompute K4s and their edges
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]

    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)
        for e in e_idx:
            edge_to_k4s[e].append(len(k4s) - 1)

    num_k4s = len(k4s)

    # For fast delta calculation
    edge_k4_info = []
    for e in range(num_edges):
        info = []
        for k4_idx in edge_to_k4s[e]:
            e_list = k4_edges[k4_idx]
            others = [x for x in e_list if x != e]
            info.append((k4_idx, others))
        edge_k4_info.append(info)

    def count_bad_k4s(colors):
        bad = 0
        for e_list in k4_edges:
            c0 = colors[e_list[0]]
            c1 = colors[e_list[1]]
            c2 = colors[e_list[2]]
            c3 = colors[e_list[3]]
            c4 = colors[e_list[4]]
            c5 = colors[e_list[5]]
            s = {c0, c1, c2, c3, c4, c5}
            if len(s) < 5:
                bad += 1
        return bad

    def delta_bad(e_idx, old_color, new_color, colors):
        delta = 0
        for k4_idx, others in edge_k4_info[e_idx]:
            c1 = colors[others[0]]
            c2 = colors[others[1]]
            c3 = colors[others[2]]
            c4 = colors[others[3]]
            c5 = colors[others[4]]
            s_before = {old_color, c1, c2, c3, c4, c5}
            bad_before = 1 if len(s_before) < 5 else 0
            s_after = {new_color, c1, c2, c3, c4, c5}
            bad_after = 1 if len(s_after) < 5 else 0
            delta += bad_after - bad_before
        return delta

    def verify_valid(colors_list):
        for e_list in k4_edges:
            cols = set([colors_list[e] for e in e_list])
            if len(cols) < 5:
                return False
        return True

    # Baseline: (i+j) % 17
    def get_baseline():
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    baseline = get_baseline()
    if not verify_valid(baseline):
        return [random.randint(0, 119) for _ in range(num_edges)]

    best_colors_used = 17
    best_result = baseline.copy()

    # Strategy: Try to reduce colors from 17 down to 13
    # Use a different approach: start with the 17-color solution and try to merge colors
    # Or use a smarter initialization based on the structure of K16
    
    # For target C in [13, 14, 15]:
    # Use multiple strategies:
    # 1. Random restarts with SA
    # 2. Start from the 17-color solution and try to recolor edges to use fewer colors
    # 3. Use a constraint-satisfaction approach: focus on the most violated K4s
    
    for target_C in [13, 14, 15]:
        if time.time() > deadline - 10:
            break

        # Try to find a valid coloring with target_C colors
        found = False

        # Strategy 1: Start from baseline (17 colors) and try to reduce
        # For each edge, try to change its color to one in [0, target_C-1]
        # Use SA to find a valid assignment
        
        # First, let's try a few random restarts
        num_restarts = 8 if target_C == 13 else 15
        max_iters = 3000

        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break

            # Initialize: random assignment
            colors = [random.randint(0, target_C - 1) for _ in range(num_edges)]
            current_bad = count_bad_k4s(colors)
            best_bad_in_run = current_bad
            best_colors_in_run = colors.copy()

            if current_bad == 0:
                found = True
                break

            # SA with adaptive cooling
            T = 2.0
            cooling = 0.9995
            T_min = 0.05
            no_improve = 0

            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break

                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, target_C - 1)
                if new_color == old_color:
                    continue

                delta = delta_bad(e_idx, old_color, new_color, colors)

                if delta < 0:
                    colors[e_idx] = new_color
                    current_bad += delta
                    no_improve = 0
                elif delta == 0:
                    if random.random() < 0.3:
                        colors[e_idx] = new_color
                        no_improve = 0
                else:
                    if random.random() < min(1.0, np.exp(-delta / max(T, 0.01))):
                        colors[e_idx] = new_color
                        current_bad += delta
                    no_improve += 1

                if current_bad < best_bad_in_run:
                    best_bad_in_run = current_bad
                    best_colors_in_run = colors.copy()

                if current_bad == 0:
                    break

                # Adapt temperature
                if no_improve > 200:
                    T = max(T_min, T * 0.95)
                    no_improve = 0
                else:
                    T = max(T_min, T * cooling)

            if best_bad_in_run == 0:
                found = True
                break

        if found:
            if target_C < best_colors_used:
                best_colors_used = target_C
                best_result = best_colors_in_run.copy()
            break

        # Strategy 2: If random restarts failed, try a greedy repair from baseline
        # Start with the 17-color solution and try to recolor each edge to use only target_C colors
        colors = baseline.copy()
        
        # For each edge, if its color >= target_C, try to change it
        # Use a local search that prioritizes edges with high colors
        current_bad = count_bad_k4s(colors)
        
        # Count how many edges need recoloring
        num_high = sum(1 for c in colors if c >= target_C)
        
        # Try to recolor each high-color edge
        for _ in range(2000):  # Max 2000 iterations
            if time.time() > deadline - 5:
                break
            if current_bad == 0:
                break
            
            # Pick a random edge with color >= target_C
            high_edges = [e for e, c in enumerate(colors) if c >= target_C]
            if not high_edges:
                # All colors are in range, but we still have bad K4s
                # Try random recoloring
                e_idx = random.randint(0, num_edges - 1)
            else:
                e_idx = random.choice(high_edges)
            
            old_color = colors[e_idx]
            new_color = random.randint(0, target_C - 1)
            if new_color == old_color:
                continue
            
            delta = delta_bad(e_idx, old_color, new_color, colors)
            
            if delta <= 0 or random.random() < np.exp(-delta / 1.0):
                colors[e_idx] = new_color
                current_bad += delta
            
            if current_bad == 0:
                break

        if current_bad == 0:
            if target_C < best_colors_used:
                best_colors_used = target_C
                best_result = colors.copy()
            break

    # Final verification
    if verify_valid(best_result):
        return best_result
    else:
        return baseline
