import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    num_edges = 120
    
    # Precompute edge indices
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_map[(i,j)] = idx
            idx += 1
            
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    edges = [
                        edge_map[(a,b)], edge_map[(a,c)], edge_map[(a,d)],
                        edge_map[(b,c)], edge_map[(b,d)], edge_map[(c,d)]
                    ]
                    k4_idx = len(k4_edges)
                    k4_edges.append(edges)
                    for e in edges:
                        edge_to_k4s[e].append(k4_idx)
                        
    def count_distinct(edges, colors):
        return len(set(colors[e] for e in edges))
        
    def calc_violations(colors):
        return sum(1 for edges in k4_edges if count_distinct(edges, colors) < 5)
        
    best_valid_colors = None
    best_valid_count = float('inf')
    
    # Seed solution
    seed_colors = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    seed_v = calc_violations(seed_colors)
    if seed_v == 0:
        best_valid_colors = seed_colors
        best_valid_count = len(set(seed_colors)) # 17
        
    # Try to find better solutions
    for max_c in [13, 14, 15]:
        if time.time() > deadline:
            break
        if best_valid_count <= max_c:
            break # Already found a solution with <= max_c
            
        phase_time = 35.0
        phase_deadline = time.time() + phase_time
        
        # SA Parameters
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        
        # Initialize
        colors = [random.randint(0, max_c - 1) for _ in range(num_edges)]
        cur_v = calc_violations(colors)
        cur_c = len(set(colors))
        cur_cost = cur_v * 10000 + cur_c
        
        while time.time() < phase_deadline:
            e_idx = random.randint(0, num_edges - 1)
            old_c = colors[e_idx]
            new_c = random.randint(0, max_c - 1)
            
            if old_c == new_c:
                continue
                
            affected = edge_to_k4s[e_idx]
            
            # Calculate delta violations
            delta_v = 0
            for k4_idx in affected:
                edges = k4_edges[k4_idx]
                old_cnt = count_distinct(edges, colors)
                # Temporarily change
                colors[e_idx] = new_c
                new_cnt = count_distinct(edges, colors)
                # Revert for next iteration's old calculation? 
                # No, we need to revert if we reject.
                # But we are calculating delta for this specific change.
                # We need to be careful: the "old_cnt" must be with old_c.
                # The "new_cnt" must be with new_c.
                # Since we are iterating through affected K4s, and the state is "old_c" at the start of the loop,
                # we must revert the change after calculating new_cnt for this K4?
                # No, the change is global to the edge.
                # So:
                # 1. State is old_c.
                # 2. For each K4, calc old_cnt (state old_c).
                # 3. Change state to new_c.
                # 4. For each K4, calc new_cnt (state new_c).
                # 5. Revert state to old_c (if rejecting) or keep (if accepting).
                
                # My previous logic was slightly flawed in implementation detail.
                # Let's do it cleanly:
                pass
                
            # Clean implementation:
            old_violations = [1 if count_distinct(k4_edges[k4_idx], colors) < 5 else 0 for k4_idx in affected]
            
            colors[e_idx] = new_c
            new_violations = [1 if count_distinct(k4_edges[k4_idx], colors) < 5 else 0 for k4_idx in affected]
            
            delta_v = sum(new_violations) - sum(old_violations)
            
            new_v = cur_v + delta_v
            new_c_set = set(colors)
            new_c_count = len(new_c_set)
            new_cost = new_v * 10000 + new_c_count
            
            if new_cost < cur_cost or random.random() < pow(2.718, -(new_cost - cur_cost) / T):
                cur_v = new_v
                cur_cost = new_cost
                # colors already updated to new_c
                
                if cur_v == 0:
                    # Valid solution
                    if new_c_count < best_valid_count:
                        best_valid_count = new_c_count
                        best_valid_colors = colors[:]
                        # We can break early if we reached max_c?
                        # No, we want to minimize further within this max_c?
                        # Actually, if we are in max_c=13 phase, and we find valid, it's optimal.
                        # If max_c=14, we might find 13 or 14.
                        # We track best_valid_count.
            else:
                colors[e_idx] = old_c
                
            T = max(T * cooling, T_min)
            
    if best_valid_colors is None:
        return seed_colors
        
    return best_valid_colors
