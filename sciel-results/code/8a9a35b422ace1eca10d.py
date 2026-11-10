import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices
    edge_idx = {}
    edges = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            edges.append((i, j))
            idx += 1
            
    # Precompute K4s and their edge indices
    k4_edges = []
    edge_to_k4 = [[] for _ in range(num_edges)]
    
    k4_idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e_ab = edge_idx[(a, b)]
                    e_ac = edge_idx[(a, c)]
                    e_ad = edge_idx[(a, d)]
                    e_bc = edge_idx[(b, c)]
                    e_bd = edge_idx[(b, d)]
                    e_cd = edge_idx[(c, d)]
                    k4_edges.append([e_ab, e_ac, e_ad, e_bc, e_bd, e_cd])
                    
                    edge_to_k4[e_ab].append(k4_idx)
                    edge_to_k4[e_ac].append(k4_idx)
                    edge_to_k4[e_ad].append(k4_idx)
                    edge_to_k4[e_bc].append(k4_idx)
                    edge_to_k4[e_bd].append(k4_idx)
                    edge_to_k4[e_cd].append(k4_idx)
                    
                    k4_idx += 1
                    
    num_k4s = len(k4_edges)
    
    # Helper to count distinct colors in a K4
    def count_distinct(k4_idx, colors):
        e_indices = k4_edges[k4_idx]
        c0 = colors[e_indices[0]]
        c1 = colors[e_indices[1]]
        c2 = colors[e_indices[2]]
        c3 = colors[e_indices[3]]
        c4 = colors[e_indices[4]]
        c5 = colors[e_indices[5]]
        return len({c0, c1, c2, c3, c4, c5})
        
    # Calculate total violations
    def calc_violations(colors):
        v = 0
        for k in range(num_k4s):
            if count_distinct(k, colors) < 5:
                v += 1
        return v
        
    # Base solution
    base_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    # base_violations = calc_violations(base_colors) # Known to be 0
    
    best_valid_colors = base_colors.copy()
    best_num_colors = 17
    
    def try_find_solution(K, time_limit):
        """
        Try to find a valid coloring with at most K colors.
        Returns (colors, num_colors_used) or None.
        """
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Random restart
            colors = [random.randint(0, K - 1) for _ in range(num_edges)]
            violations = calc_violations(colors)
            
            if violations == 0:
                num_used = len(set(colors))
                return colors, num_used
            
            # Local search (Simulated Annealing)
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            
            cur_violations = violations
            cur_colors = colors.copy()
            
            while time.time() < end_time and T > min_T:
                # Pick a random edge
                e = random.randint(0, num_edges - 1)
                c_old = cur_colors[e]
                
                # Pick a new color
                c_new = random.randint(0, K - 1)
                if c_new == c_old:
                    continue
                
                # Calculate delta violations
                affected_k4s = edge_to_k4[e]
                delta_v = 0
                for k in affected_k4s:
                    old_distinct = count_distinct(k, cur_colors)
                    cur_colors[e] = c_new
                    new_distinct = count_distinct(k, cur_colors)
                    cur_colors[e] = c_old
                    
                    if old_distinct < 5 and new_distinct >= 5:
                        delta_v -= 1
                    elif old_distinct >= 5 and new_distinct < 5:
                        delta_v += 1
                        
                new_violations = cur_violations + delta_v
                
                # Acceptance
                if new_violations < cur_violations or random.random() < pow(2.71828, -delta_v / T):
                    cur_colors[e] = c_new
                    cur_violations = new_violations
                    
                    if cur_violations == 0:
                        num_used = len(set(cur_colors))
                        return cur_colors, num_used
                        
                T *= cooling
                
        return None

    # Try K=13, 14, 15
    # 13: 20s, 14: 40s, 15: 60s
    
    remaining = deadline - time.time()
    if remaining > 20:
        res = try_find_solution(13, 20)
        if res:
            colors, num = res
            if num < best_num_colors:
                best_valid_colors = colors
                best_num_colors = num
            return colors

    remaining = deadline - time.time()
    if remaining > 40:
        res = try_find_solution(14, 40)
        if res:
            colors, num = res
            if num < best_num_colors:
                best_valid_colors = colors
                best_num_colors = num
            return colors

    remaining = deadline - time.time()
    if remaining > 10:
        res = try_find_solution(15, remaining - 5)
        if res:
            colors, num = res
            if num < best_num_colors:
                best_valid_colors = colors
                best_num_colors = num
            return colors

    return best_valid_colors
