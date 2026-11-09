import random
import time
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute K4s and their edges
    # k4s[i] is a list of 6 edge indices for the i-th K4
    # edge_to_k4s[e] is a list of K4 indices containing edge e
    k4s = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    # Map (u, v) to edge index
    edge_idx = {}
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx[(u, v)] = len(edge_list)
            edge_list.append((u, v))
            
    # Generate all K4s
    for quad in itertools.combinations(range(n), 4):
        edges_in_k4 = []
        for u, v in itertools.combinations(quad, 2):
            idx = edge_idx[(u, v)]
            edges_in_k4.append(idx)
        k4s.append(edges_in_k4)
        
        for e in edges_in_k4:
            edge_to_k4s[e].append(len(k4s) - 1)
            
    def try_greedy(seed):
        random.seed(seed)
        colors = [-1] * num_edges
        # Randomize edge order to escape bad local minima
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        max_color_used = -1
        
        for e in edge_order:
            u, v = edge_list[e]
            # Try colors 0..max_color_used+1 (and a bit more just in case)
            # We can cap the search for new colors if we are doing well, 
            # but let's just try incrementally.
            
            valid = False
            # We only need to try up to the current max color + 1.
            # If none work, we might need a new color, but if a new color is needed,
            # it will definitely be valid for the "4 colors in K4" constraint 
            # provided no K4 is already broken (|S|<4).
            
            for c in range(max_color_used + 2):
                is_valid = True
                for k4_idx in edge_to_k4s[e]:
                    k4_edges = k4s[k4_idx]
                    # Check if all other 5 edges are colored
                    # and determine the set of colors S
                    s_colors = set()
                    all_colored = True
                    for e2 in k4_edges:
                        if e2 == e:
                            continue
                        if colors[e2] == -1:
                            all_colored = False
                            break
                        s_colors.add(colors[e2])
                    
                    if all_colored:
                        size_s = len(s_colors)
                        if size_s < 4:
                            # This K4 is doomed. No color c can fix it to have >=5 colors.
                            # Because max distinct colors would be size_s + 1 <= 4.
                            return None
                        elif size_s == 4:
                            if c in s_colors:
                                is_valid = False
                                break
                        # If size_s >= 5, any c is fine for this K4.
                
                if is_valid:
                    colors[e] = c
                    if c > max_color_used:
                        max_color_used = c
                    valid = True
                    break
            
            if not valid:
                # Should not happen if we allow new colors and no K4 is doomed.
                # But if a K4 was doomed, we returned None already.
                # So this case implies we ran out of colors? No, we go up to max+1.
                # If max+1 is not valid, it means for some K4, size_s=4 and c in s_colors.
                # But if we pick a new color c_new > max_color_used, c_new cannot be in s_colors 
                # (since s_colors contains only old colors).
                # So a new color is always valid unless size_s < 4.
                return None

        return colors, max_color_used + 1

    best_result = None
    best_count = 17 # Seed uses 17
    
    # Try a few seeds
    for seed in range(50):
        if time.time() > deadline:
            break
            
        res = try_greedy(seed)
        if res:
            colors, count = res
            if count < best_count:
                best_count = count
                best_result = colors
                
            if best_count == 13:
                break # Theoretical min

    if best_result is None:
        # Fallback to seed if greedy failed (should not happen)
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    # Convert list to tuple/list as required
    return best_result
