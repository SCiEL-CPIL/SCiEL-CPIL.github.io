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

    # Precompute edge_k4_info for fast delta
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

    # Step 1: Construct a valid 14-coloring
    # Split into V1 (0-7) and V2 (8-15)
    V1 = list(range(8))
    V2 = list(range(8, 16))
    
    # Proper edge coloring for K8 (V1)
    # Use a simple greedy algorithm or a known pattern
    # For K8, we can use colors 0-6.
    # A known proper edge coloring for K_{2m} involves 1-factors.
    # We will just use a randomized greedy proper coloring for V1 and V2.
    
    def get_proper_edge_coloring(vertices, num_colors):
        """
        Returns a list of colors for edges among 'vertices' using 0..num_colors-1
        such that no two adjacent edges have the same color.
        """
        v_edges = []
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                v_edges.append((vertices[i], vertices[j]))
        
        # Map local index to global edge index
        local_to_global = {}
        for idx, (u, v) in enumerate(v_edges):
            local_to_global[(u, v)] = edge_index[(u, v)]
            
        colors = [0] * len(v_edges)
        
        # Greedy with backtracking if needed, but for K8 greedy usually works with 7 colors
        # To be safe, we use a fixed known 1-factorization based coloring.
        # For K8, vertices 0..7.
        # Color of (i,j) is (i+j) % 7 if i!=j? No, that failed.
        # Let's use the "circle method" for 1-factorization.
        # Fix vertex 0.
        # Pairs: (1,7), (2,6), (3,5) -> Color 0? No.
        
        # Simpler: Just try random proper coloring.
        # Since K8 is small, we can do a quick search.
        # But we need it to be fast.
        
        # Let's use a deterministic construction for K8.
        # Vertices: 0,1,2,3,4,5,6,7
        # Edges:
        # (0,1): 0, (0,2): 1, (0,3): 2, (0,4): 3, (0,5): 4, (0,6): 5, (0,7): 6
        # (1,2): 2, (1,3): 3, (1,4): 4, (1,5): 5, (1,6): 6, (1,7): 0
        # (2,3): 4, (2,4): 5, (2,5): 6, (2,6): 0, (2,7): 1
        # (3,4): 6, (3,5): 0, (3,6): 1, (3,7): 2
        # (4,5): 1, (4,6): 2, (4,7): 3
        # (5,6): 3, (5,7): 4
        # (6,7): 5
        
        # This is a proper edge coloring.
        # We can generate this programmatically.
        # For K_{2m}, the edge (i,j) can be colored by (i+j) mod (2m-1) if we permute vertices?
        
        # Let's just hardcode the logic for K8 proper coloring.
        # It's easier to just use a random assignment and check, retrying if not proper.
        # Probability of random assignment being proper is low, but we can use greedy.
        
        def greedy_proper_coloring(v_list, num_c):
            # v_list is list of global vertex ids
            # Returns dict (u,v) -> color
            col_map = {}
            # Sort edges to make it deterministic?
            # Just iterate
            for i in range(len(v_list)):
                for j in range(i+1, len(v_list)):
                    u, v = v_list[i], v_list[j]
                    used = set()
                    # Check neighbors of u
                    for w in v_list:
                        if w != u and w != v:
                            if (min(u,w), max(u,w)) in col_map:
                                used.add(col_map[(min(u,w), max(u,w))])
                            elif (min(w,u), max(w,u)) in col_map: # Should be same key
                                pass
                    # Check neighbors of v
                    for w in v_list:
                        if w != u and w != v:
                            if (min(v,w), max(v,w)) in col_map:
                                used.add(col_map[(min(v,w), max(v,w))])
                    
                    # Find smallest available color
                    for c in range(num_c):
                        if c not in used:
                            col_map[(u,v)] = c
                            break
                    else:
                        # Should not happen for K8 with 7 colors
                        col_map[(u,v)] = 0
            return col_map

        # Greedy might fail if not ordered correctly, but for K8 it's sparse enough?
        # Actually, K8 is dense.
        # Let's use the explicit mapping derived from 1-factorization.
        # Reference: https://math.stackexchange.com/questions/465565/proper-edge-coloring-of-k-8
        
        # For K8, we can use:
        # Color(i,j) = (i+j) % 7 if i+j < 7 or ...?
        # Let's just use the hardcoded list from above, mapped to local indices.
        
        # Local indices 0..7 correspond to v_list[0]..v_list[7]
        # Hardcoded colors for local edges (i,j) with i<j:
        # (0,1):0, (0,2):1, (0,3):2, (0,4):3, (0,5):4, (0,6):5, (0,7):6
        # (1,2):2, (1,3):3, (1,4):4, (1,5):5, (1,6):6, (1,7):0
        # (2,3):4, (2,4):5, (2,5):6, (2,6):0, (2,7):1
        # (3,4):6, (3,5):0, (3,6):1, (3,7):2
        # (4,5):1, (4,6):2, (4,7):3
        # (5,6):3, (5,7):4
        # (6,7):5
        
        hardcoded = {
            (0,1):0, (0,2):1, (0,3):2, (0,4):3, (0,5):4, (0,6):5, (0,7):6,
            (1,2):2, (1,3):3, (1,4):4, (1,5):5, (1,6):6, (1,7):0,
            (2,3):4, (2,4):5, (2,5):6, (2,6):0, (2,7):1,
            (3,4):6, (3,5):0, (3,6):1, (3,7):2,
            (4,5):1, (4,6):2, (4,7):3,
            (5,6):3, (5,7):4,
            (6,7):5
        }
        
        for li, u in enumerate(v_list):
            for lj, v in enumerate(v_list):
                if li < lj:
                    global_u, global_v = u, v
                    if global_u > global_v:
                        global_u, global_v = global_v, global_u
                    # The hardcoded map is for local indices 0..7
                    # We need to map local (li, lj) to hardcoded
                    c = hardcoded[(li, lj)]
                    local_to_global_val = edge_index[(global_u, global_v)]
                    # We return a list of colors for the edges in v_list
                    # But we need to fill the global array.
                    # Let's just return the dict (global_u, global_v) -> color
                    # Wait, the function returns colors for the edges.
                    # I will restructure to return a list of colors for the global edges array.
        
        # This is getting messy. Let's just fill the global colors array directly.
        return None # Placeholder

    # Better approach: Just create the 14-coloring directly in the global array.
    
    def create_14_coloring():
        colors = [0] * num_edges
        V1 = list(range(8))
        V2 = list(range(8, 16))
        
        # Intra V1
        # Use hardcoded proper coloring for K8 mapped to 0..7
        hardcoded = {
            (0,1):0, (0,2):1, (0,3):2, (0,4):3, (0,5):4, (0,6):5, (0,7):6,
            (1,2):2, (1,3):3, (1,4):4, (1,5):5, (1,6):6, (1,7):0,
            (2,3):4, (2,4):5, (2,5):6, (2,6):0, (2,7):1,
            (3,4):6, (3,5):0, (3,6):1, (3,7):2,
            (4,5):1, (4,6):2, (4,7):3,
            (5,6):3, (5,7):4,
            (6,7):5
        }
        
        for i in range(8):
            for j in range(i+1, 8):
                c = hardcoded[(i,j)]
                idx = edge_index[(i,j)]
                colors[idx] = c
        
        # Intra V2
        # Same coloring, shifted by 0? No, we can use same colors 0..6
        for i in range(8):
            for j in range(i+1, 8):
                c = hardcoded[(i,j)]
                u, v = 8+i, 8+j
                idx = edge_index[(u,v)]
                colors[idx] = c
        
        # Inter V1-V2
        # Use colors 7..13
        # Random assignment
        for _ in range(1000): # Try up to 1000 times
            for u in V1:
                for v in V2:
                    idx = edge_index[(u,v)]
                    colors[idx] = random.randint(7, 13)
            
            if verify_valid(colors):
                return colors
            if time.time() > deadline - 5:
                break
        
        # If failed, return a valid 17-coloring (baseline)
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

    best_colors = create_14_coloring()
    best_count = 14
    
    # Step 2: Try to reduce to 13
    if time.time() < deadline - 10:
        # Start from 14-coloring, but map colors > 12 to random 0..12
        # Or just random restart with 13 colors, but use the 14-coloring as a guide?
        # Simpler: Simulated Annealing with 13 colors.
        
        target_C = 13
        restarts = 10
        max_iterations = 5000
        
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random 13 colors
            colors = [random.randint(0, target_C - 1) for _ in range(num_edges)]
            current_bad = count_bad_k4s(colors)
            best_bad_in_run = current_bad
            best_colors_in_run = colors.copy()
            
            if current_bad == 0:
                best_count = target_C
                best_colors = colors.copy()
                break
            
            T = 1.0
            cooling = 0.999
            
            for it in range(max_iterations):
                if time.time() > deadline - 5:
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
                elif delta == 0:
                    if random.random() < 0.5:
                        colors[e_idx] = new_color
                else:
                    if random.random() < np.exp(-delta / max(T, 0.01)):
                        colors[e_idx] = new_color
                        current_bad += delta
                
                if current_bad < best_bad_in_run:
                    best_bad_in_run = current_bad
                    best_colors_in_run = colors.copy()
                
                if current_bad == 0:
                    break
                
                T *= cooling
            
            if best_bad_in_run == 0:
                best_count = target_C
                best_colors = best_colors_in_run.copy()
                break

    # Final verification
    if verify_valid(best_colors):
        # Count distinct colors used
        used = set(best_colors)
        return best_colors
    else:
        # Fallback to baseline
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
