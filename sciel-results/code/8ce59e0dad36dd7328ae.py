import time
import random
import numpy as np

def construct():
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Map (i, j) to index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - (i * (i + 1) // 2) + (j - i - 1)
        
    # Precompute K4 edge tuples
    k4_edges = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    edges = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d), edge_idx(c, d)
                    ]
                    k4_edges.append(edges)
                    
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)
    num_k4s = len(k4_edges)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    # Convert to arrays for faster access
    edge_to_k4s_arr = [np.array(lst, dtype=np.int32) for lst in edge_to_k4s]
    
    # Precompute vertex to edges mapping for vertex-based search
    vertex_to_edges = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = edge_idx(i, j)
            vertex_to_edges[i].append(idx)
            vertex_to_edges[j].append(idx)
    vertex_to_edges_arr = [np.array(lst, dtype=np.int32) for lst in vertex_to_edges]

    def calculate_penalties(colors):
        """
        Vectorized calculation of penalties for all K4s.
        """
        k4_colors = colors[k4_edges_arr]
        sorted_colors = np.sort(k4_colors, axis=1)
        diffs = np.diff(sorted_colors, axis=1)
        distinct = 1 + np.sum(diffs != 0, axis=1)
        penalties = np.where(distinct < 5, (5 - distinct) ** 2, 0)
        return penalties

    def is_valid(colors):
        return np.sum(calculate_penalties(colors)) == 0

    def count_colors(colors):
        return len(np.unique(colors))

    def fallback():
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]

    def search_vertex_labels(time_limit):
        """
        Search for vertex labels v_i in Z_m such that c_ij = (v_i + v_j) % m
        satisfies the K4 constraint and uses few colors.
        """
        start_time = time.time()
        best_solution = None
        best_color_count = float('inf')
        
        # Try different moduli
        for m in [13, 14, 15, 16]:
            if time.time() - start_time > time_limit:
                break
                
            for restart in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                # Initialize labels
                if restart == 0:
                    labels = np.random.randint(0, m, size=n, dtype=np.int32)
                elif restart == 1:
                    labels = np.arange(n) % m
                else:
                    labels = np.random.randint(0, m, size=n, dtype=np.int32)
                
                # Helper to get colors from labels
                def get_colors(labels):
                    cols = np.zeros(num_edges, dtype=np.int32)
                    for i in range(n):
                        for j in range(i + 1, n):
                            cols[edge_idx(i, j)] = (labels[i] + labels[j]) % m
                    return cols

                colors = get_colors(labels)
                penalties = calculate_penalties(colors)
                cost = np.sum(penalties)
                num_colors = count_colors(colors)
                
                if cost == 0:
                    if num_colors < best_color_count:
                        best_color_count = num_colors
                        best_solution = colors.copy()
                        if best_color_count <= 13:
                            return best_solution.tolist()
                
                # Local search step
                no_improve = 0
                while time.time() - start_time < time_limit:
                    if cost == 0:
                        if num_colors < best_color_count:
                            best_color_count = num_colors
                            best_solution = colors.copy()
                        break # Try another restart or modulus
                    
                    # Pick a random vertex to change
                    v_idx = random.randint(0, n - 1)
                    old_label = labels[v_idx]
                    
                    best_new_label = old_label
                    best_delta = 0
                    best_new_colors = None
                    
                    # Try all new labels
                    for new_label in range(m):
                        if new_label == old_label:
                            continue
                        
                        # Update labels
                        labels[v_idx] = new_label
                        
                        # Recalculate colors and penalties
                        # This is O(120) for colors and O(1820) for penalties
                        new_colors = get_colors(labels)
                        new_penalties = calculate_penalties(new_colors)
                        new_cost = np.sum(new_penalties)
                        delta = new_cost - cost
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_new_label = new_label
                            best_new_colors = new_colors
                            best_new_penalties = new_penalties

                    if best_delta < 0:
                        labels[v_idx] = best_new_label
                        colors = best_new_colors
                        penalties = best_new_penalties
                        cost = cost + best_delta
                        no_improve = 0
                        num_colors = count_colors(colors)
                        
                        if cost == 0 and num_colors < best_color_count:
                            best_color_count = num_colors
                            best_solution = colors.copy()
                            if best_color_count <= 13:
                                return best_solution.tolist()
                    else:
                        no_improve += 1
                        if no_improve > 50:
                            # Random kick
                            for _ in range(5):
                                v_idx = random.randint(0, n - 1)
                                labels[v_idx] = random.randint(0, m - 1)
                            colors = get_colors(labels)
                            penalties = calculate_penalties(colors)
                            cost = np.sum(penalties)
                            num_colors = count_colors(colors)
                            no_improve = 0
                            
                            if cost == 0 and num_colors < best_color_count:
                                best_color_count = num_colors
                                best_solution = colors.copy()
                                if best_color_count <= 13:
                                    return best_solution.tolist()

        if best_solution is not None:
            return best_solution.tolist()
        return None

    def local_search(k, time_limit):
        start_time = time.time()
        
        best_solution = None
        best_cost = float('inf')
        
        init_17 = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
                
            if restart % 2 == 0:
                colors = np.random.randint(0, k, size=num_edges, dtype=np.int32)
            else:
                colors = init_17 % k
                
            k4_penalties = calculate_penalties(colors)
            cur_cost = np.sum(k4_penalties)
            
            if cur_cost == 0:
                return colors.tolist()
                
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_solution = colors.copy()
                
            no_improve_count = 0
            while time.time() - start_time < time_limit:
                if cur_cost == 0:
                    return colors.tolist()
                    
                violating_indices = np.where(k4_penalties > 0)[0]
                if len(violating_indices) == 0:
                    break
                
                k4_idx = violating_indices[random.randint(0, len(violating_indices) - 1)]
                edges = k4_edges_arr[k4_idx]
                e_idx = edges[random.randint(0, 5)]
                old_c = colors[e_idx]
                
                best_new_c = old_c
                best_delta = 0
                
                affected_k4s = edge_to_k4s_arr[e_idx]
                old_affected_penalties = k4_penalties[affected_k4s]
                
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    
                    colors[e_idx] = new_c
                    affected_colors = colors[k4_edges_arr[affected_k4s]]
                    sorted_affected = np.sort(affected_colors, axis=1)
                    diffs_affected = np.diff(sorted_affected, axis=1)
                    distinct_affected = 1 + np.sum(diffs_affected != 0, axis=1)
                    new_affected_penalties = np.where(distinct_affected < 5, (5 - distinct_affected) ** 2, 0)
                    
                    delta = np.sum(new_affected_penalties) - np.sum(old_affected_penalties)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c

                if best_delta < 0:
                    colors[e_idx] = best_new_c
                    affected_colors = colors[k4_edges_arr[affected_k4s]]
                    sorted_affected = np.sort(affected_colors, axis=1)
                    diffs_affected = np.diff(sorted_affected, axis=1)
                    distinct_affected = 1 + np.sum(diffs_affected != 0, axis=1)
                    new_affected_penalties = np.where(distinct_affected < 5, (5 - distinct_affected) ** 2, 0)
                    
                    k4_penalties[affected_k4s] = new_affected_penalties
                    cur_cost += best_delta
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > 100:
                        for _ in range(5):
                            kick_e = random.randint(0, num_edges - 1)
                            colors[kick_e] = random.randint(0, k - 1)
                        
                        k4_penalties = calculate_penalties(colors)
                        cur_cost = np.sum(k4_penalties)
                        no_improve_count = 0
                        
                        if cur_cost < best_cost:
                            best_cost = cur_cost
                            best_solution = colors.copy()

        if best_solution is not None and is_valid(best_solution):
            return best_solution.tolist()
        return None

    time_start = time.time()
    
    # Try vertex labeling first (faster, structural)
    # Allocate 60s to this
    res_labels = search_vertex_labels(60.0)
    if res_labels is not None:
        if is_valid(np.array(res_labels)):
            return res_labels

    # Fallback to edge-based local search
    remaining_time = 100.0 - (time.time() - time_start)
    if remaining_time < 5:
        return fallback()
        
    time_limit_per_k = remaining_time / 3.0
    
    for k in [13, 14, 15]:
        if time.time() - time_start > 100:
            break
        res = local_search(k, time_limit_per_k)
        if res is not None:
            if is_valid(np.array(res)):
                return res

    return fallback()
