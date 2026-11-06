import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Pre-compute K4s and their edges
    k4s = []
    k4_edges = [] # list of list of edge indices
    for comb in itertools.combinations(range(n), 4):
        v0, v1, v2, v3 = comb
        e_indices = [
            edge_index[(v0, v1)],
            edge_index[(v0, v2)],
            edge_index[(v0, v3)],
            edge_index[(v1, v2)],
            edge_index[(v1, v3)],
            edge_index[(v2, v3)]
        ]
        k4s.append(comb)
        k4_edges.append(e_indices)
        
    num_k4s = len(k4s)
    
    # For each edge, list of K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)
            
    def generate_1_factorization_colors(perm):
        """
        Generates colors based on a 1-factorization of K16 using the given vertex permutation.
        perm is a list of 16 integers, a permutation of 0..15.
        Standard vertex s corresponds to actual vertex perm[s].
        """
        colors = [0] * num_edges
        
        for k in range(15):
            # Edge (15, k) in standard -> (perm[15], perm[k]) in actual
            u_std, v_std = 15, k
            u_act, v_act = perm[u_std], perm[v_std]
            if u_act > v_act:
                u_act, v_act = v_act, u_act
            e_idx = edge_index[(u_act, v_act)]
            colors[e_idx] = k
            
            for i in range(1, 8):
                u_std = (k + i) % 15
                v_std = (k - i) % 15
                if u_std > v_std:
                    u_std, v_std = v_std, u_std
                u_act, v_act = perm[u_std], perm[v_std]
                if u_act > v_act:
                    u_act, v_act = v_act, u_act
                e_idx = edge_index[(u_act, v_act)]
                colors[e_idx] = k
        return colors

    def compact_colors(colors):
        unique_c = sorted(set(colors))
        if len(unique_c) <= 1:
            return colors
        mapping = {old: new for new, old in enumerate(unique_c)}
        new_colors = [mapping[c] for c in colors]
        return new_colors

    def can_merge(c_from, c_to, colors):
        """
        Check if all edges of color c_from can be changed to c_to.
        Returns True if safe.
        """
        # Find all edges with color c_from
        edges_to_change = [e for e in range(num_edges) if colors[e] == c_from]
        
        for e_idx in edges_to_change:
            for k4_idx in edge_to_k4s[e_idx]:
                e_list = k4_edges[k4_idx]
                # Get colors of other 5 edges
                other_colors = set()
                for e in e_list:
                    if e != e_idx:
                        other_colors.add(colors[e])
                
                if len(other_colors) == 4 and c_to in other_colors:
                    return False
        return True

    def local_search(initial_colors, time_limit):
        current_colors = initial_colors[:]
        current_colors = compact_colors(current_colors)
        num_unique = max(current_colors) + 1 if current_colors else 0
        best_local = current_colors[:]
        best_local_num = num_unique
        
        iter_count = 0
        no_improve_count = 0
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Compact current colors to keep indices small
            current_colors = compact_colors(current_colors)
            num_unique = max(current_colors) + 1 if current_colors else 0
            
            if num_unique < best_local_num:
                best_local_num = num_unique
                best_local = current_colors[:]
                
            if best_local_num == 13:
                break # Hit lower bound
            
            improved = False
            # Try to merge
            if num_unique > 1:
                colors_in_use = list(set(current_colors))
                random.shuffle(colors_in_use)
                
                # Try a few random merges
                for _ in range(100):
                    if len(colors_in_use) < 2:
                        break
                    c_from = random.choice(colors_in_use)
                    c_to = random.choice(colors_in_use)
                    if c_from == c_to:
                        continue
                    
                    if can_merge(c_from, c_to, current_colors):
                        # Perform merge
                        for e in range(num_edges):
                            if current_colors[e] == c_from:
                                current_colors[e] = c_to
                        improved = True
                        break
                
                if improved:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # Kick: Try to change a single edge to a "new" color 
                    e_idx = random.randint(0, num_edges - 1)
                    new_color = num_unique
                    old_c = current_colors[e_idx]
                    
                    valid_kick = True
                    for k4_idx in edge_to_k4s[e_idx]:
                        e_list = k4_edges[k4_idx]
                        other_colors = set()
                        for e in e_list:
                            if e != e_idx:
                                other_colors.add(current_colors[e])
                        
                        if len(other_colors) < 4:
                            valid_kick = False
                            break
                    
                    if valid_kick:
                        current_colors[e_idx] = new_color
                        no_improve_count = 0
                    else:
                        # If kick fails, try a random move to an existing color?
                        if no_improve_count > 50:
                            # Try to move one edge to another color
                            e_idx = random.randint(0, num_edges - 1)
                            old_c = current_colors[e_idx]
                            if len(colors_in_use) > 1:
                                target_color = random.choice(colors_in_use)
                            else:
                                target_color = colors_in_use[0]
                            if target_color != old_c:
                                # Check if valid
                                is_valid = True
                                for k4_idx in edge_to_k4s[e_idx]:
                                    e_list = k4_edges[k4_idx]
                                    other_colors = set()
                                    for e in e_list:
                                        if e != e_idx:
                                            other_colors.add(current_colors[e])
                                    # New size: len(other_colors U {target_color})
                                    if len(other_colors | {target_color}) < 5:
                                        is_valid = False
                                        break
                                if is_valid:
                                    current_colors[e_idx] = target_color
                                    no_improve_count = 0
                                    improved = True # Count as activity
                            
            iter_count += 1
            if iter_count % 1000 == 0:
                pass

        return best_local, best_local_num

    best_colors_global = None
    best_num_global = 16 # Upper bound
    
    # Try multiple random permutations
    # We have ~110s. Let's allocate ~10s per run.
    # 10 runs should be feasible.
    
    num_runs = 10
    for run in range(num_runs):
        if time.time() > deadline - 10.0:
            break
            
        perm = list(range(n))
        random.shuffle(perm)
        
        init_colors = generate_1_factorization_colors(perm)
        
        # Verify seed is valid (should be, but just in case)
        seed_valid = True
        for k4_idx in range(num_k4s):
            e_list = k4_edges[k4_idx]
            if len({init_colors[e] for e in e_list}) < 5:
                seed_valid = False
                break
        
        if not seed_valid:
            # Fallback
            for i, (u, v) in enumerate(edges):
                init_colors[i] = (u + v) % 17
            init_colors = compact_colors(init_colors)
        
        res_colors, res_num = local_search(init_colors, 10.0)
        
        if res_num < best_num_global:
            best_num_global = res_num
            best_colors_global = res_colors[:]
            
        if best_num_global == 13:
            break
            
    if best_colors_global is None:
        # Fallback to first run result or default
        perm = list(range(n))
        init_colors = generate_1_factorization_colors(perm)
        best_colors_global = init_colors
        best_num_global = max(init_colors) + 1

    final_compact = compact_colors(best_colors_global)
    return final_compact
