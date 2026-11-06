import time
import random
from itertools import combinations
from collections import defaultdict

def is_2_colorable_fast(n, edges):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if there exists a 2-coloring with no monochromatic edge.
    """
    if not edges:
        return True
    
    m = len(edges)
    # Precompute: for each vertex, which edges contain it
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    
    assignment = [-1] * n
    
    # Counters for each edge: how many 0s, 1s, unassigned
    c0 = [0] * m
    c1 = [0] * m
    cu = [5] * m  # unassigned count
    
    def propagate(assign):
        # Returns False if contradiction, True otherwise.
        # Modifies 'assign', c0, c1, cu in place.
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if c0[i] == 5 or c1[i] == 5:
                    return False
                if c0[i] == 4 and cu[i] == 1:
                    # The one unassigned vertex must be 1
                    for v in edges[i]:
                        if assign[v] == -1:
                            assign[v] = 1
                            cu[i] -= 1
                            c1[i] += 1
                            # Update other edges containing v
                            for j in vert_edges[v]:
                                if j != i:
                                    cu[j] -= 1
                                    c1[j] += 1
                            changed = True
                            break
                elif c1[i] == 4 and cu[i] == 1:
                    for v in edges[i]:
                        if assign[v] == -1:
                            assign[v] = 0
                            cu[i] -= 1
                            c0[i] += 1
                            for j in vert_edges[v]:
                                if j != i:
                                    cu[j] -= 1
                                    c0[j] += 1
                            changed = True
                            break
        return True
    
    def solve(assign, c0, c1, cu):
        # Find next unassigned vertex using heuristic:
        # Pick a vertex that appears in an edge with the most assigned vertices (critical edge)
        v = -1
        best_criticality = -1
        for i in range(m):
            if cu[i] == 0:
                continue
            # Criticality: edges with 4 assigned and 1 unassigned are most critical
            crit = 0
            if c0[i] == 4 or c1[i] == 4:
                crit = 100
            elif c0[i] + c1[i] >= 3:
                crit = 50 + (c0[i] + c1[i])
            else:
                crit = c0[i] + c1[i]
            if crit > best_criticality:
                best_criticality = crit
                # Pick the first unassigned vertex in this edge
                for vv in edges[i]:
                    if assign[vv] == -1:
                        v = vv
                        break
        if v == -1:
            return True
        
        # Try 0
        new_assign_0 = assign[:]
        new_c0_0 = c0[:]
        new_c1_0 = c1[:]
        new_cu_0 = cu[:]
        new_assign_0[v] = 0
        for j in vert_edges[v]:
            new_cu_0[j] -= 1
            new_c0_0[j] += 1
        if propagate(new_assign_0, new_c0_0, new_c1_0, new_cu_0):
            if solve(new_assign_0, new_c0_0, new_c1_0, new_cu_0):
                return True
        
        # Try 1
        new_assign_1 = assign[:]
        new_c0_1 = c0[:]
        new_c1_1 = c1[:]
        new_cu_1 = cu[:]
        new_assign_1[v] = 1
        for j in vert_edges[v]:
            new_cu_1[j] -= 1
            new_c1_1[j] += 1
        if propagate(new_assign_1, new_c0_1, new_c1_1, new_cu_1):
            if solve(new_assign_1, new_c0_1, new_c1_1, new_cu_1):
                return True
        
        return False
    
    if not propagate(assignment, c0, c1, cu):
        return False
    
    return solve(assignment, c0, c1, cu)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    # Seed solution: Complete 5-uniform hypergraph on n=9 vertices.
    seed_edges = [list(e) for e in combinations(range(9), 5)]
    best_edges = seed_edges
    best_n = 9
    best_count = len(best_edges)
    
    # Known small non-2-colorable constructions to try
    # Try to find constructions with fewer edges
    
    # Strategy 1: Start with random hypergraphs and use greedy deletion
    # Strategy 2: Try structured constructions (e.g., projective planes, etc.)
    # Strategy 3: Local search / simulated annealing
    
    # First, try to improve from the seed using greedy deletion
    # The seed has 126 edges on n=9. We can try to remove edges.
    
    def greedy_delete(n, edges, time_limit):
        """Try to remove edges while maintaining non-2-colorability."""
        current_edges = set(tuple(e) for e in edges)
        removal_order = list(current_edges)
        random.shuffle(removal_order)
        
        improved = True
        while improved:
            improved = False
            if time.time() > time_limit:
                break
            for edge_tuple in removal_order:
                if time.time() > time_limit:
                    break
                if edge_tuple not in current_edges:
                    continue
                test_edges = current_edges - {edge_tuple}
                test_list = [list(e) for e in test_edges]
                if not is_2_colorable_fast(n, test_list):
                    current_edges = test_edges
                    improved = True
                    break  # Restart removal order after each deletion
        
        return [list(e) for e in current_edges]
    
    # Apply greedy deletion to seed
    if time.time() < deadline - 60:
        deleted = greedy_delete(best_n, best_edges, deadline - 60)
        if len(deleted) < best_count:
            best_count = len(deleted)
            best_edges = deleted
            best_n = best_n
    
    # Try different n values and random starts
    for n in range(9, 15):
        if time.time() > deadline - 10:
            break
        
        restarts = 3
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
            
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            
            # Start with a moderate number of edges
            k = min(len(all_possible), 60 + random.randint(0, 40))
            edges = [list(e) for e in all_possible[:k]]
            
            # Ensure it's non-2-colorable by adding edges
            while is_2_colorable_fast(n, edges):
                if time.time() > deadline - 10:
                    break
                if len(edges) < len(all_possible):
                    edge_set = set(tuple(e) for e in edges)
                    added = False
                    for e in all_possible:
                        if tuple(e) not in edge_set:
                            edges.append(list(e))
                            edge_set.add(tuple(e))
                            added = True
                            break
                    if not added:
                        break
                else:
                    break
            
            if time.time() > deadline - 10:
                break
            
            if is_2_colorable_fast(n, edges):
                continue
            
            # Greedy deletion
            deleted = greedy_delete(n, edges, deadline - 10)
            if not is_2_colorable_fast(n, deleted):
                continue
            if len(deleted) < best_count:
                best_count = len(deleted)
                best_edges = deleted
                best_n = n
    
    # Local search: try to swap edges to reduce count
    if time.time() < deadline - 5 and best_count > 30:
        # Try to find a smaller construction by local search
        # This is complex, so we'll do a simple version
        pass
    
    # Final verification
    if is_2_colorable_fast(best_n, best_edges):
        # Fallback to seed
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
        best_count = len(best_edges)
    
    return {'n': best_n, 'edges': best_edges}
