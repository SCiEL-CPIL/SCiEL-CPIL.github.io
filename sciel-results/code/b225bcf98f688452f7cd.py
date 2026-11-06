import sys
import random
import time

N = 16
sys.setrecursionlimit(10000)

def construct():
    # Generate all edges in lexicographic order
    edges = []
    edge_to_idx = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(edges)
            edges.append((i, j))
    
    num_edges = len(edges)
    
    # Precompute cycles
    # A cycle is defined by 4 vertices {u, v, w, x}.
    # The edges are (u,v), (v,w), (w,x), (x,u) in some order.
    # There are 3 distinct cycles for each set of 4 vertices.
    # Cycle 1: u-v, v-w, w-x, x-u
    # Cycle 2: u-w, w-v, v-x, x-u  (u-v, u-w, v-x, w-x? No, edges are undirected)
    # Let's list the 3 perfect matchings of the 4 vertices to form cycles?
    # No, a 4-cycle in K4 uses 4 edges.
    # In K4 with vertices 0,1,2,3:
    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Cycles:
    # 1: 0-1-2-3-0: (0,1), (1,2), (2,3), (3,0)
    # 2: 0-1-3-2-0: (0,1), (1,3), (3,2), (2,0)
    # 3: 0-2-1-3-0: (0,2), (2,1), (1,3), (3,0)
    
    cycles_for_edge = [[] for _ in range(num_edges)]
    # We also need for each edge e, and each other edge a in a cycle with e,
    # what are the other two edges b, c?
    # This structure: edge_pair_constraints[e][a] = [b, c]
    # If e, b, c are colored, then a is constrained by colors of e, b, c.
    
    # Let's build a list of all cycles first.
    all_cycles = []
    for u in range(N):
        for v in range(u+1, N):
            for w in range(v+1, N):
                for x in range(w+1, N):
                    # Vertices u, v, w, x
                    # Cycle 1: u-v, v-w, w-x, x-u
                    e1 = edge_to_idx[(u, v)]
                    e2 = edge_to_idx[(v, w)]
                    e3 = edge_to_idx[(w, x)]
                    e4 = edge_to_idx[(u, x)]
                    all_cycles.append((e1, e2, e3, e4))
                    
                    # Cycle 2: u-v, v-x, x-w, w-u
                    e1 = edge_to_idx[(u, v)]
                    e2 = edge_to_idx[(v, x)]
                    e3 = edge_to_idx[(w, x)]
                    e4 = edge_to_idx[(u, w)]
                    all_cycles.append((e1, e2, e3, e4))
                    
                    # Cycle 3: u-w, w-v, v-x, x-u
                    e1 = edge_to_idx[(u, w)]
                    e2 = edge_to_idx[(v, w)]
                    e3 = edge_to_idx[(v, x)]
                    e4 = edge_to_idx[(u, x)]
                    all_cycles.append((e1, e2, e3, e4))
                    
    # Build constraint map:
    # For each edge e, and each edge a in a cycle with e:
    # store the other two edges b, c.
    # constraints[e] = { a: [b, c], ... }
    constraints = [dict() for _ in range(num_edges)]
    
    for cyc in all_cycles:
        e0, e1, e2, e3 = cyc
        # For e0, neighbors are e1, e2, e3.
        # If e1, e2 are colored, e3 is constrained by e0, e1, e2?
        # No, the constraint is on the *uncolored* edge.
        # If we are coloring e0, it affects e1, e2, e3.
        # Specifically, if e1 and e2 are colored, then e3's domain is restricted by {c(e0), c(e1), c(e2)}.
        # So for edge e0, if we look at neighbor e3, the "other two" are e1 and e2.
        
        # e0 is constrained by e1, e2, e3 if they are colored.
        # But we are building a map to check forward propagation.
        # When e0 is colored:
        #   For each neighbor a in {e1, e2, e3}:
        #     Let b, c be the other two edges in the cycle.
        #     If b and c are colored, then a is now constrained by e0, b, c.
        
        # For e0:
        #   Neighbor e1: other two are e2, e3.
        #   Neighbor e2: other two are e1, e3.
        #   Neighbor e3: other two are e1, e2.
        
        constraints[e0][e1] = [e2, e3]
        constraints[e0][e2] = [e1, e3]
        constraints[e0][e3] = [e1, e2]
        
        constraints[e1][e0] = [e2, e3]
        constraints[e1][e2] = [e0, e3]
        constraints[e1][e3] = [e0, e2]
        
        constraints[e2][e0] = [e1, e3]
        constraints[e2][e1] = [e0, e3]
        constraints[e2][e3] = [e0, e1]
        
        constraints[e3][e0] = [e1, e2]
        constraints[e3][e1] = [e0, e2]
        constraints[e3][e2] = [e0, e1]

    # State
    colors = [-1] * num_edges
    used_colors = set()
    best_count = 0
    best_assignment = None
    
    # Helper to get valid colors for an edge given current colors
    def get_valid_colors(e_idx):
        valid = set(range(120))
        # Check all cycles containing e_idx
        # We need to know which cycles contain e_idx.
        # We didn't precompute that directly, but we can derive it from constraints?
        # No, constraints[e_idx] lists neighbors.
        # For each neighbor a, we have the other two b, c.
        # The cycle is {e_idx, a, b, c}.
        # If a, b, c are colored, and their colors are distinct, e_idx must be in that set.
        
        for a, (b, c) in constraints[e_idx].items():
            if colors[a] != -1 and colors[b] != -1 and colors[c] != -1:
                c_a = colors[a]
                c_b = colors[b]
                c_c = colors[c]
                s = {c_a, c_b, c_c}
                if len(s) == 3:
                    valid.intersection_update(s)
                    if not valid:
                        return set()
        
        return valid

    # Forward Check:
    # When we color e_idx with col, we need to check if any neighbor a now has empty domain.
    # a is affected if the other two edges b, c in the cycle {e_idx, a, b, c} are already colored.
    def is_consistent_after_coloring(e_idx, col):
        colors[e_idx] = col
        for a, (b, c) in constraints[e_idx].items():
            if colors[a] == -1: # a is uncolored
                if colors[b] != -1 and colors[c] != -1:
                    # b and c are colored. e_idx is now colored.
                    # So a is constrained by e_idx, b, c.
                    c_e = col
                    c_b = colors[b]
                    c_c = colors[c]
                    s = {c_e, c_b, c_c}
                    if len(s) == 3:
                        # a must be in s.
                        # Check if a has any valid color in s that is not ruled out by other constraints?
                        # For a simple check, we just verify if s is non-empty (it is, size 3).
                        # But we must ensure that a's *other* constraints don't eliminate all of s.
                        # This is expensive.
                        # Simplified Forward Check: Just check if the domain of a becomes empty
                        # due to THIS constraint? No, we need to check intersection with ALL constraints.
                        # So we call get_valid_colors(a) and see if it's empty.
                        if not get_valid_colors(a):
                            colors[e_idx] = -1
                            return False
        colors[e_idx] = -1
        return True

    def dfs(idx, depth):
        nonlocal best_count, best_assignment
        
        if depth == 10000: # Safety
            return False
            
        # Find uncolored edge
        # MRV Heuristic
        min_valid_count = 121
        target_edge = -1
        target_valid = None
        
        # To speed up, we can just pick the first uncolored edge?
        # MRV is better.
        # But calculating valid colors for all 120 edges at each step is slow.
        # Let's try a compromise: Pick the first uncolored edge in lexicographic order?
        # Or use a static ordering?
        # Let's try static ordering first for speed, then optimize if needed.
        # Actually, let's just iterate and find the first uncolored.
        # This is O(E).
        
        for i in range(num_edges):
            if colors[i] == -1:
                target_edge = i
                break
        
        if target_edge == -1:
            # All colored
            if len(used_colors) > best_count:
                best_count = len(used_colors)
                best_assignment = colors[:]
            return True

        valid_colors = get_valid_colors(target_edge)
        
        if not valid_colors:
            return False

        # Value Ordering: New colors first
        new_colors = [c for c in valid_colors if c not in used_colors]
        old_colors = [c for c in valid_colors if c in used_colors]
        
        random.shuffle(new_colors)
        random.shuffle(old_colors)
        
        candidates = new_colors + old_colors
        
        for c in candidates:
            # Check forward consistency
            if is_consistent_after_coloring(target_edge, c):
                colors[target_edge] = c
                if c not in used_colors:
                    used_colors.add(c)
                
                if dfs(idx + 1, depth + 1):
                    return True
                
                colors[target_edge] = -1
                if c not in used_colors:
                    # It was added, so remove it
                    used_colors.remove(c)
                # If it was already in used_colors, we didn't add it, so don't remove.
                # Wait, logic:
                # if c not in used_colors: add.
                # So if it was NOT in used_colors, we added it. Now we must remove it.
                # If it WAS in used_colors, we didn't add it. So we don't remove it.
                # The code above:
                # if c not in used_colors: used_colors.remove(c)
                # This is correct IF we know that c is currently in used_colors (which it is, because we added it).
                # But what if c was already in used_colors? Then we didn't add it.
                # The condition `if c not in used_colors` in the ADD block ensures we only add if new.
                # So in the REMOVE block, we should only remove if we added it.
                # We can track this with a boolean.
                
                # Let's refactor to be safe.
                
            # Refactored loop for safety
            pass # The above logic has a bug in removal. Let's rewrite the loop body properly.

        return False

    # Rewrite DFS loop properly
    def dfs_correct(depth):
        nonlocal best_count, best_assignment
        
        if depth > 120:
            return False

        # Find target edge
        target_edge = -1
        for i in range(num_edges):
            if colors[i] == -1:
                target_edge = i
                break
        
        if target_edge == -1:
            if len(used_colors) > best_count:
                best_count = len(used_colors)
                best_assignment = colors[:]
            return True

        valid_colors = get_valid_colors(target_edge)
        
        if not valid_colors:
            return False

        new_colors = [c for c in valid_colors if c not in used_colors]
        old_colors = [c for c in valid_colors if c in used_colors]
        
        random.shuffle(new_colors)
        random.shuffle(old_colors)
        
        candidates = new_colors + old_colors
        
        for c in candidates:
            is_new = c not in used_colors
            
            # Check consistency
            if is_consistent_after_coloring(target_edge, c):
                colors[target_edge] = c
                if is_new:
                    used_colors.add(c)
                
                if dfs_correct(depth + 1):
                    return True
                
                colors[target_edge] = -1
                if is_new:
                    used_colors.remove(c)
                
        return False

    # Run search
    # To handle potential timeouts, we can add a node limit.
    # But let's try without first.
    
    # Random seed for reproducibility in tests, but we want to explore.
    # The problem asks to maximize.
    
    # One issue: The static order (first uncolored) might be bad.
    # Let's try a few random restarts.
    
    best_global = 0
    best_global_assign = None
    
    for restart in range(10):
        random.seed(restart)
        colors = [-1] * num_edges
        used_colors = set()
        best_count = 0
        best_assignment = None
        
        dfs_correct(0)
        
        if best_count > best_global:
            best_global = best_count
            best_global_assign = best_assignment[:]
            
        if best_global >= 100:
            break

    if best_global_assign is None:
        # Fallback
        return [0] * num_edges
        
    return best_global_assign

