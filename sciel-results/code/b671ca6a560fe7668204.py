import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # Try to find a solution for n from 34 down to 1
    # We prioritize higher n.
    # If we find a valid coloring for n, we return it immediately?
    # The task says "Maximize n". So we should try 34, 33...
    # If 34 is solvable, we return it.
    
    for n in range(34, 0, -1):
        if time.time() > deadline - 5:
            break
        
        # Try to solve for this n
        # We can try multiple random seeds/restarts
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            result = solve_sa(n, time_limit=5)
            if result is not None:
                best_n = n
                best_colors = result
                break
        
        if best_n == n:
            break
            
    if best_n == 0:
        # Fallback to a known small solution if SA fails completely
        # The seed program provided n=9.
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        return {'n': n, 'colors': colors}
        
    return {'n': best_n, 'colors': best_colors}

def solve_sa(n, time_limit=5):
    """
    Uses Simulated Annealing to find a valid coloring for n vertices.
    Returns list of colors or None.
    """
    start_time = time.time()
    
    # Precompute edges
    edges = list(combinations(range(n), 3))
    num_edges = len(edges)
    
    # Map edge tuple to index
    edge_index = {e: i for i, e in enumerate(edges)}
    
    # Precompute K4s and K5s
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    # Map each edge to the K4s and K5s it belongs to
    # And also store the other edges in those cliques to check validity
    # For a K4 (a,b,c,d), the edges are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
    # For a K5 (a,b,c,d,e), the edges are all 10 combinations.
    
    edge_to_k4s = [[] for _ in range(num_edges)]
    edge_to_k5s = [[] for _ in range(num_edges)]
    
    # For each K4, identify its 4 edges
    # Store as list of (k4_idx, edge_idx_in_k4)
    # Actually, we just need to know which K4s an edge is part of.
    # To check if a K4 is all red, we need the colors of its 4 edges.
    
    # Let's store for each K4, the indices of its 4 edges.
    k4_edges = []
    for k4 in k4s:
        a, b, c, d = k4
        e1 = edge_index[(a,b,c)]
        e2 = edge_index[(a,b,d)]
        e3 = edge_index[(a,c,d)]
        e4 = edge_index[(b,c,d)]
        k4_edges.append([e1, e2, e3, e4])
        
        # Update edge_to_k4s
        for e in [e1, e2, e3, e4]:
            edge_to_k4s[e].append(len(k4_edges) - 1)
            
    k5_edges = []
    for k5 in k5s:
        edges_in_k5 = [edge_index[e] for e in combinations(k5, 3)]
        k5_edges.append(edges_in_k5)
        
        for e in edges_in_k5:
            edge_to_k5s[e].append(len(k5_edges) - 1)
            
    # Initial random coloring
    colors = [random.randint(0, 1) for _ in range(num_edges)]
    
    # Calculate initial violations
    # red_k4_violation[i] = 1 if K4 i is all red
    red_k4_violation = [0] * len(k4s)
    blue_k5_violation = [0] * len(k5s)
    
    total_violations = 0
    
    def update_violations():
        nonlocal total_violations
        total_violations = 0
        for i, e_list in enumerate(k4_edges):
            if all(colors[e] == 0 for e in e_list):
                red_k4_violation[i] = 1
                total_violations += 1
            else:
                red_k4_violation[i] = 0
                
        for i, e_list in enumerate(k5_edges):
            if all(colors[e] == 1 for e in e_list):
                blue_k5_violation[i] = 1
                total_violations += 1
            else:
                blue_k5_violation[i] = 0
                
    update_violations()
    
    if total_violations == 0:
        return colors.copy()
        
    T = 1.0
    cooling = 0.99
    best_colors = colors.copy()
    best_violations = total_violations
    
    while time.time() < start_time + time_limit:
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        new_color = 1 - old_color
        
        delta_v = 0
        
        # Check K4s containing e_idx
        for k4_idx in edge_to_k4s[e_idx]:
            e_list = k4_edges[k4_idx]
            if old_color == 0:
                # Was Red, becoming Blue.
                # If it was a violation (all red), it stops being one.
                if red_k4_violation[k4_idx]:
                    delta_v -= 1
                    # It will no longer be a violation
                    # We don't need to check if it becomes a violation because it's no longer all red
            else:
                # Was Blue, becoming Red.
                # It was not a red violation.
                # Check if it becomes one: all other 3 edges must be Red.
                is_all_red = True
                for e in e_list:
                    if e == e_idx:
                        continue
                    if colors[e] != 0:
                        is_all_red = False
                        break
                if is_all_red:
                    delta_v += 1
                    
        # Check K5s containing e_idx
        for k5_idx in edge_to_k5s[e_idx]:
            e_list = k5_edges[k5_idx]
            if old_color == 1:
                # Was Blue, becoming Red.
                # If it was a violation (all blue), it stops being one.
                if blue_k5_violation[k5_idx]:
                    delta_v -= 1
            else:
                # Was Red, becoming Blue.
                # Check if it becomes one: all other 9 edges must be Blue.
                is_all_blue = True
                for e in e_list:
                    if e == e_idx:
                        continue
                    if colors[e] != 1:
                        is_all_blue = False
                        break
                if is_all_blue:
                    delta_v += 1
                    
        new_v = total_violations + delta_v
        
        # Acceptance probability
        if delta_v <= 0 or random.random() < pow(2.718, -delta_v / T):
            # Accept
            colors[e_idx] = new_color
            
            # Update violation states
            for k4_idx in edge_to_k4s[e_idx]:
                e_list = k4_edges[k4_idx]
                if new_color == 0:
                    # Now Red. Check if all red.
                    if all(colors[e] == 0 for e in e_list):
                        red_k4_violation[k4_idx] = 1
                    else:
                        red_k4_violation[k4_idx] = 0
                else:
                    # Now Blue. Cannot be red violation.
                    red_k4_violation[k4_idx] = 0
                    
            for k5_idx in edge_to_k5s[e_idx]:
                e_list = k5_edges[k5_idx]
                if new_color == 1:
                    # Now Blue. Check if all blue.
                    if all(colors[e] == 1 for e in e_list):
                        blue_k5_violation[k5_idx] = 1
                    else:
                        blue_k5_violation[k5_idx] = 0
                else:
                    # Now Red. Cannot be blue violation.
                    blue_k5_violation[k5_idx] = 0
                    
            total_violations = new_v
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors.copy()
                
            if total_violations == 0:
                return colors.copy()
                
        T *= cooling
        
    # If we didn't find 0, return None (or best if it's 0? No, must be valid)
    if best_violations == 0:
        return best_colors
    return None
