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

    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)

    num_k4s = len(k4s)

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

    # Strategy: Vertex-labeling approach
    # Assign labels to vertices, derive edge colors from labels
    # Try different color derivation functions
    
    def try_vertex_labeling(target_C, num_restarts=10, max_iters=5000):
        """Try to find a valid coloring using vertex labels"""
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None, None
            
            # Initialize vertex labels
            labels = [random.randint(0, target_C - 1) for _ in range(n)]
            
            # Derive edge colors from vertex labels
            # Use a function that maximizes diversity in K4s
            # Try: (label[i] * label[j]) % target_C, or (label[i] + label[j]) % target_C, etc.
            
            # Let's try multiple derivation functions
            best_bad = float('inf')
            best_labels = None
            best_derive_func = None
            
            derive_funcs = [
                lambda i, j, l: (l[i] * l[j]) % target_C,
                lambda i, j, l: (l[i] + l[j]) % target_C,
                lambda i, j, l: (l[i] * (j + 1) + l[j] * (i + 1)) % target_C,
                lambda i, j, l: (l[i] ^ l[j]) % target_C,
            ]
            
            for df in derive_funcs:
                # Compute colors from labels
                colors = [df(i, j, labels) for i, j in edges]
                
                # Count bad K4s
                bad = 0
                for e_list in k4_edges:
                    cols = set([colors[e] for e in e_list])
                    if len(cols) < 5:
                        bad += 1
                
                if bad < best_bad:
                    best_bad = bad
                    best_labels = labels.copy()
                    best_derive_func = df
                
                if bad == 0:
                    return colors, target_C
            
            # Local search on vertex labels
            current_labels = best_labels.copy()
            current_derive = best_derive_func
            current_colors = [current_derive(i, j, current_labels) for i, j in edges]
            current_bad = best_bad
            
            if current_bad == 0:
                return current_colors, target_C
            
            T = 2.0
            cooling = 0.9995
            T_min = 0.05
            no_improve = 0
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                
                # Pick a random vertex to relabel
                v = random.randint(0, n - 1)
                old_label = current_labels[v]
                new_label = random.randint(0, target_C - 1)
                if new_label == old_label:
                    continue
                
                # Compute delta: how does changing this vertex's label affect bad K4 count?
                # A vertex v is in K4s with 3 other vertices chosen from the other 15
                # Edges affected: all edges incident to v (15 edges)
                # K4s affected: those containing v (C(15,3) = 455 K4s)
                
                # For efficiency, only consider K4s that contain v
                # Precompute: for each vertex, which K4s contain it
                # But for now, just recompute affected K4s
                
                delta = 0
                # Find all K4s containing vertex v
                for k4_idx, quad in enumerate(k4s):
                    if v not in quad:
                        continue
                    
                    a, b, c, d = quad
                    e_idx = [
                        edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
                        edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
                    ]
                    
                    # Get current colors for these edges
                    # Some edges may be affected by the label change (those incident to v)
                    old_colors = []
                    new_colors = []
                    for e in e_idx:
                        u, w = edges[e]
                        old_c = current_colors[e]
                        # If edge is incident to v, its color changes
                        if u == v or w == v:
                            other = w if u == v else u
                            new_c = current_derive(v, other, current_labels) if v < other else current_derive(other, v, current_labels)
                            # Actually, we need to compute with new label
                            # Let's compute properly
                        else:
                            new_c = old_c
                        
                        old_colors.append(old_c)
                        new_colors.append(new_c)
                    
                    # This is getting complex. Let's use a simpler approach:
                    # Just recompute all colors and count bad K4s
                    # But that's O(num_k4s) per iteration, which might be too slow.
                    
                    # Alternative: use a cheaper heuristic
                    pass
                
                # For simplicity, let's use a different approach:
                # Just try random label changes and evaluate
                current_labels[v] = new_label
                new_colors = [current_derive(i, j, current_labels) for i, j in edges]
                
                new_bad = 0
                for e_list in k4_edges:
                    cols = set([new_colors[e] for e in e_list])
                    if len(cols) < 5:
                        new_bad += 1
                
                if new_bad < current_bad:
                    current_bad = new_bad
                    current_colors = new_colors
                    no_improve = 0
                elif new_bad == current_bad:
                    if random.random() < 0.3:
                        current_colors = new_colors
                        no_improve = 0
                else:
                    # Revert
                    current_labels[v] = old_label
                    if random.random() < min(1.0, np.exp(-(new_bad - current_bad) / max(T, 0.01))):
                        current_labels[v] = new_label
                        current_colors = new_colors
                        current_bad = new_bad
                    no_improve += 1
                
                if current_bad == 0:
                    return current_colors, target_C
                
                if no_improve > 200:
                    T = max(T_min, T * 0.95)
                    no_improve = 0
                else:
                    T = max(T_min, T * cooling)
        
        return None, None

    # Try vertex labeling for target_C in [13, 14, 15]
    for target_C in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
        
        colors, c_used = try_vertex_labeling(target_C, num_restarts=5, max_iters=3000)
        
        if colors is not None and c_used is not None:
            if verify_valid(colors):
                if c_used < best_colors_used:
                    best_colors_used = c_used
                    best_result = colors.copy()
                break
    
    # If vertex labeling didn't find anything better, fall back to original SA approach
    if best_colors_used == 17:
        # Original SA approach on edge colors
        edge_k4_info = []
        edge_to_k4s = [[] for _ in range(num_edges)]
        for k4_idx, e_list in enumerate(k4_edges):
            for e in e_list:
                edge_to_k4s[e].append(k4_idx)
        
        for e in range(num_edges):
            info = []
            for k4_idx in edge_to_k4s[e]:
                e_list = k4_edges[k4_idx]
                others = [x for x in e_list if x != e]
                info.append((k4_idx, others))
            edge_k4_info.append(info)
        
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
        
        for target_C in [13, 14, 15]:
            if time.time() > deadline - 5:
                break
            
            num_restarts = 8 if target_C == 13 else 15
            max_iters = 3000
            
            for r in range(num_restarts):
                if time.time() > deadline - 3:
                    break
                
                colors = [random.randint(0, target_C - 1) for _ in range(num_edges)]
                current_bad = sum(1 for e_list in k4_edges if len(set(colors[e] for e in e_list)) < 5)
                
                if current_bad == 0:
                    if target_C < best_colors_used:
                        best_colors_used = target_C
                        best_result = colors.copy()
                    break
                
                T = 2.0
                cooling = 0.9995
                T_min = 0.05
                no_improve = 0
                
                for it in range(max_iters):
                    if time.time() > deadline - 2:
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
                    
                    if current_bad == 0:
                        break
                    
                    if no_improve > 200:
                        T = max(T_min, T * 0.95)
                        no_improve = 0
                    else:
                        T = max(T_min, T * cooling)
                
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
