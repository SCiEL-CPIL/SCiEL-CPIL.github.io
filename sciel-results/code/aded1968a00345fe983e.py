"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, product

def construct():
    deadline = time.time() + 115  # 115s safety margin before 150s limit
    n = 16
    
    # Precompute Q3 structure
    # Q3 vertices: 0..7, edges connect i,j if i^j is a power of 2
    q3_edges = []
    q3_adj = [[] for _ in range(8)]
    for i in range(8):
        for j in range(i+1, 8):
            d = i ^ j
            if d & (d - 1) == 0:  # power of 2
                q3_edges.append((i, j))
                q3_adj[i].append(j)
                q3_adj[j].append(i)
    
    # For checking if edge (u,v) creates a Q3:
    # In Q3, fix edge (0,1). The remaining vertices 2-7 form a specific structure.
    # Q3 with edge 0-1:
    # 0 connects to: 1, 2, 4
    # 1 connects to: 0, 3, 5
    # 2 connects to: 0, 3, 6
    # 3 connects to: 1, 2, 7
    # 4 connects to: 0, 5, 6
    # 5 connects to: 1, 4, 7
    # 6 connects to: 2, 4, 7
    # 7 connects to: 3, 5, 6
    #
    # So given u->0, v->1:
    #   a2: neighbor of u (not v), must connect to a3
    #   a4: neighbor of u (not v, not a2), must connect to a5
    #   a3: neighbor of v (not u), must connect to a2
    #   a5: neighbor of v (not u, not a3), must connect to a4
    #   a6: neighbor of a2 AND a4 (distinct from others)
    #   a7: neighbor of a3 AND a5 (distinct from others, not a6)
    #   a6-a7 must be an edge
    
    def creates_q3(u, v, edge_set):
        """Check if adding edge (u,v) creates a Q3 in the graph."""
        # Candidates for a2: neighbors of u (excluding v)
        neigh_u = set()
        for w in range(n):
            if w == v:
                continue
            if (min(u, w), max(u, w)) in edge_set:
                neigh_u.add(w)
        
        neigh_v = set()
        for w in range(n):
            if w == u:
                continue
            if (min(v, w), max(v, w)) in edge_set:
                neigh_v.add(w)
        
        neigh_u_list = list(neigh_u)
        neigh_v_list = list(neigh_v)
        
        for a2 in neigh_u_list:
            for a4 in neigh_u_list:
                if a4 == a2:
                    continue
                for a3 in neigh_v_list:
                    if a3 == a2 or a3 == a4:
                        continue
                    if (min(a2, a3), max(a2, a3)) not in edge_set:
                        continue
                    for a5 in neigh_v_list:
                        if a5 == a2 or a5 == a3 or a5 == a4:
                            continue
                        if (min(a4, a5), max(a4, a5)) not in edge_set:
                            continue
                        # Find a6: common neighbor of a2 and a4, not in {u,v,a2,a3,a4,a5}
                        used = {u, v, a2, a3, a4, a5}
                        # Common neighbors of a2 and a4
                        cn_a2a4 = set()
                        for w in range(n):
                            if w in used:
                                continue
                            if (min(a2, w), max(a2, w)) in edge_set and (min(a4, w), max(a4, w)) in edge_set:
                                cn_a2a4.add(w)
                        if not cn_a2a4:
                            continue
                        # Common neighbors of a3 and a5
                        cn_a3a5 = set()
                        for w in range(n):
                            if w in used:
                                continue
                            if (min(a3, w), max(a3, w)) in edge_set and (min(a5, w), max(a5, w)) in edge_set:
                                cn_a3a5.add(w)
                        if not cn_a3a5:
                            continue
                        for a6 in cn_a2a4:
                            for a7 in cn_a3a5:
                                if a7 == a6:
                                    continue
                                if (min(a6, a7), max(a6, a7)) in edge_set:
                                    return True
        return False
    
    # Generate all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u+1, n):
            all_edges.append((u, v))
    
    def try_add_edges(initial_set, edge_order):
        """Try to add edges from edge_order to initial_set, checking for Q3."""
        trial_set = initial_set.copy()
        added = 0
        for u, v in edge_order:
            if (u, v) in trial_set:
                continue
            if time.time() > deadline:
                break
            if not creates_q3(u, v, trial_set):
                trial_set.add((u, v))
                added += 1
        return trial_set, added
    
    # Strategy: Try multiple random orderings, keep the best result
    random.seed(42)
    best_result = None
    best_count = -1
    
    # Initial: try all edges in natural order
    trial_set, added = try_add_edges(set(), all_edges)
    best_result = trial_set
    best_count = len(trial_set)
    
    # Try random restarts
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        random_order = all_edges[:]
        random.shuffle(random_order)
        trial_set, _ = try_add_edges(set(), random_order)
        if len(trial_set) > best_count:
            best_count = len(trial_set)
            best_result = trial_set.copy()
        
        # Also try starting from current best and adding remaining
        if time.time() < deadline - 20:
            remaining = [e for e in all_edges if e not in best_result]
            if remaining:
                random.shuffle(remaining)
                trial_set2, _ = try_add_edges(best_result.copy(), remaining)
                if len(trial_set2) > best_count:
                    best_count = len(trial_set2)
                    best_result = trial_set2.copy()
    
    # Convert to list of edges
    result_edges = [[u, v] for u, v in best_result]
    return result_edges
# EVOLVE-BLOCK-END
