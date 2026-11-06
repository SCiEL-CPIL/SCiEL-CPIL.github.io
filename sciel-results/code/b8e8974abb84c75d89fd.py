import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 20s margin for safety
    
    n = 20
    n_colors = 20
    
    # Precompute all 4-cycles and the edges in them
    # A 4-cycle is a set of 4 edges {u-v, v-w, w-x, x-u} for distinct u,v,w,x
    # We need to check: do these 4 edges have distinct colors?
    
    # For efficiency, let's think about the structure.
    # A 4-cycle violation exists if there is a 4-cycle where two edges share a color.
    
    # Let's use a different representation for speed.
    # adj[u][v] = color, or -1 if no edge.
    # colors is an array of size n*n, flattened.
    
    # Precompute all 4-cycles as tuples of 4 edge indices (u*n+v, v*n+w, w*n+x, x*n) with u<v<w<x? No, cycles can be in any order.
    # A 4-cycle is defined by 4 distinct vertices a,b,c,d.
    # The edges are (a,b), (b,c), (c,d), (d,a).
    # We need to check if the colors of these 4 edges are all distinct.
    
    # Number of 4-cycles in K20 is C(20,4) * 3 = 4845 * 3 = 14535.
    # This is a small number. We can store them.
    
    # Let's store 4-cycles as lists of 4 edge keys.
    # edge key for (u,v) with u<v is u*n+v.
    
    four_cycles = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # 3 ways to form a 4-cycle from 4 vertices:
                    # a-b-c-d-a
                    # a-b-d-c-a
                    # a-c-b-d-a
                    # Actually, any 4 distinct vertices form exactly 3 distinct 4-cycles.
                    # Let's just generate them systematically.
                    # Cycle 1: a-b-c-d
                    edges1 = [(a,b), (b,c), (c,d), (d,a)]
                    four_cycles.append(edges1)
                    # Cycle 2: a-b-d-c
                    edges2 = [(a,b), (b,d), (d,c), (c,a)]
                    four_cycles.append(edges2)
                    # Cycle 3: a-c-b-d
                    edges3 = [(a,c), (c,b), (b,d), (d,a)]
                    four_cycles.append(edges3)
    
    # Convert to edge indices for faster lookup
    four_cycles_idx = []
    for cycle in four_cycles:
        idxs = []
        for u, v in cycle:
            if u > v:
                u, v = v, u
            idxs.append(u * n + v)
        four_cycles_idx.append(tuple(idxs))
    
    # Initialize with K20 and random colors
    def edge_idx(u, v):
        if u > v:
            u, v = v, u
        return u * n + v
    
    colors = np.zeros(n*n, dtype=np.int8)
    for u in range(n):
        for v in range(u+1, n):
            colors[edge_idx(u, v)] = random.randint(0, n_colors - 1)
    
    # violation_count: number of 4-cycles that are violated
    # For each 4-cycle, it's violated if any two of its edges have the same color.
    
    def count_violations():
        count = 0
        for cycle in four_cycles_idx:
            c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
            # Check if all distinct
            if len(set([c0, c1, c2, c3])) < 4:
                count += 1
        return count
    
    current_violations = count_violations()
    best_violations = current_violations
    best_colors = colors.copy()
    best_edge_count = n * (n-1) // 2
    
    # Local search
    # For each edge, try changing its color to all possible colors.
    # Calculate delta in violations.
    
    # To make this faster, we can precompute for each edge which 4-cycles it belongs to.
    edge_to_cycles = [[] for _ in range(n*n)]
    for i, cycle in enumerate(four_cycles_idx):
        for idx in cycle:
            edge_to_cycles[idx].append(i)
    
    # Precompute color sets for each 4-cycle for fast update?
    # Let's just recalculate the violation status of affected 4-cycles.
    
    def get_violation(cycle):
        c0, c1, c2, c3 = colors[cycle[0]], colors[cycle[1]], colors[cycle[2]], colors[cycle[3]]
        return len(set([c0, c1, c2, c3])) < 4
    
    # Current violation status for each 4-cycle
    violation_status = np.zeros(len(four_cycles_idx), dtype=bool)
    for i, cycle in enumerate(four_cycles_idx):
        violation_status[i] = get_violation(cycle)
    
    current_violations = np.sum(violation_status)
    
    # Edge colors
    edge_colors = colors.copy()
    
    # For each edge, store its color
    # We will try to change edge colors to reduce violations.
    
    # Also, we can remove edges. If an edge is removed, all 4-cycles containing it become "satisfied" (vacuously? No, they just don't exist anymore).
    # Actually, if an edge is removed, the 4-cycle is no longer a subgraph, so it's not a violation.
    # So removing an edge reduces the number of violations by the number of violated 4-cycles containing that edge.
    
    # Let's track which edges are present.
    present = np.ones(n*n, dtype=bool)
    for u in range(n):
        for v in range(u+1, n):
            present[edge_idx(u, v)] = True
    
    def get_edge_present_count():
        return np.sum(present)
    
    def try_improve():
        nonlocal current_violations, best_violations, best_colors, best_edge_count
        nonlocal edge_colors, violation_status, present
        
        # Find the edge that gives the best improvement if recoloring or removing.
        best_improve = 0
        best_action = None
        best_edge = -1
        best_color = -1
        
        for e in range(n*n):
            if not present[e]:
                continue
            
            # Try removing
            # Number of violated 4-cycles containing e
            remove_improve = 0
            for ci in edge_to_cycles[e]:
                if violation_status[ci]:
                    remove_improve += 1
            
            if remove_improve > best_improve:
                best_improve = remove_improve
                best_action = 'remove'
                best_edge = e
                best_color = -1
            
            # Try recoloring to each color
            current_color = edge_colors[e]
            for c in range(n_colors):
                if c == current_color:
                    continue
                
                # Calculate delta
                delta = 0
                for ci in edge_to_cycles[e]:
                    cycle = four_cycles_idx[ci]
                    old_status = violation_status[ci]
                    
                    # Temporarily change
                    old_c = edge_colors[e]
                    edge_colors[e] = c
                    
                    new_status = get_violation(cycle)
                    edge_colors[e] = old_c
                    
                    if old_status and not new_status:
                        delta -= 1
                    elif not old_status and new_status:
                        delta += 1
                
                if -delta > best_improve:
                    best_improve = -delta
                    best_action = 'recolor'
                    best_edge = e
                    best_color = c
        
        if best_action is None:
            return False
        
        # Apply action
        if best_action == 'remove':
            present[best_edge] = False
            for ci in edge_to_cycles[best_edge]:
                if violation_status[ci]:
                    violation_status[ci] = False
                    current_violations -= 1
        else:  # recolor
            old_color = edge_colors[best_edge]
            edge_colors[best_edge] = best_color
            for ci in edge_to_cycles[best_edge]:
                cycle = four_cycles_idx[ci]
                old_status = violation_status[ci]
                new_status = get_violation(cycle)
                violation_status[ci] = new_status
                if old_status and not new_status:
                    current_violations -= 1
                elif not old_status and new_status:
                    current_violations += 1
        
        # Update best if valid
        if current_violations == 0:
            if get_edge_present_count() > best_edge_count:
                best_edge_count = get_edge_present_count()
                best_colors = edge_colors.copy()
                best_violations = 0
        
        return True
    
    # Run local search
    while time.time() < deadline:
        improved = False
        for _ in range(10):  # Multiple passes per time check
            if not try_improve():
                break
            improved = True
            if current_violations == 0:
                break
        if not improved and current_violations > 0:
            # Stuck, try a random restart or random perturbation
            # Perturb: pick a random present edge and change its color
            present_edges = [e for e in range(n*n) if present[e]]
            if present_edges:
                e = random.choice(present_edges)
                old_color = edge_colors[e]
                new_color = random.randint(0, n_colors - 1)
                while new_color == old_color:
                    new_color = random.randint(0, n_colors - 1)
                edge_colors[e] = new_color
                for ci in edge_to_cycles[e]:
                    cycle = four_cycles_idx[ci]
                    old_status = violation_status[ci]
                    new_status = get_violation(cycle)
                    violation_status[ci] = new_status
                    if old_status and not new_status:
                        current_violations -= 1
                    elif not old_status and new_status:
                        current_violations += 1
                if current_violations == 0:
                    if get_edge_present_count() > best_edge_count:
                        best_edge_count = get_edge_present_count()
                        best_colors = edge_colors.copy()
                        best_violations = 0
    
    # Construct result from best solution
    # If best_violations > 0, we need to make it valid by removing edges
    if best_violations > 0:
        # Fallback: remove edges until valid
        colors_temp = best_colors.copy()
        present_temp = present.copy()
        # This might be slow, but it's a fallback
        # Actually, let's just use the current state if it's valid, otherwise try to fix
        pass
    
    # Let's just return the best valid solution found
    # If none found, return a star
    if best_violations == 0:
        result = []
        for u in range(n):
            for v in range(u+1, n):
                e = edge_idx(u, v)
                if best_colors[e] != -1:
                    result.append([u, v, int(best_colors[e])])
        return result
    else:
        # Fallback: star
        return [[0, v, 0] for v in range(1, n)]
