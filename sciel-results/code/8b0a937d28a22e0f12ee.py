"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n = 16
    
    # Precompute Q3 structure
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            diff = bin(i ^ j).count('1')
            if diff == 1:
                q3_edges.add((i, j))
    
    # Build Q3 adjacency lists
    q3_adj = [set() for _ in range(8)]
    for (a, b) in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)
    
    def check_q3_free(adj):
        """Check if graph with adjacency matrix adj is Q3-free using backtracking."""
        # Try all 8-vertex subsets
        for subset in itertools.combinations(range(n), 8):
            subset_set = set(subset)
            # Backtracking: assign Q3 vertices to subset vertices
            assignment = [-1] * 8
            used = set()
            
            def backtrack(q3_idx):
                if q3_idx == 8:
                    return True
                for sv in subset:
                    if sv in used:
                        continue
                    # Check compatibility with already assigned neighbors
                    compatible = True
                    for prev_q3 in range(q3_idx):
                        if prev_q3 in q3_adj[q3_idx]:
                            prev_sv = assignment[prev_q3]
                            if not adj[sv][prev_sv]:
                                compatible = False
                                break
                    if compatible:
                        used.add(sv)
                        assignment[q3_idx] = sv
                        if backtrack(q3_idx + 1):
                            return True
                        used.remove(sv)
                        assignment[q3_idx] = -1
                return False
            
            if backtrack(0):
                return False  # Found a Q3 subgraph
        
        return True  # No Q3 subgraph found
    
    def check_q3_free_fast(adj, new_edge):
        """Fast check: only verify if adding new_edge creates a Q3."""
        u, v = new_edge
        # A new Q3 must include the edge (u,v)
        # Find all ways to embed Q3 such that some Q3 edge maps to (u,v)
        # For each Q3 edge (a,b), try mapping a->u, b->v and a->v, b->u
        
        for (a, b) in q3_edges:
            for mapping in [(a, u, b, v), (a, v, b, u)]:
                qa, su, qb, sv = mapping
                # Check that su and sv are adjacent (they are, since we're adding this edge)
                # Now try to extend: assign remaining Q3 vertices
                assignment = {qa: su, qb: sv}
                used = {su, sv}
                
                # Order remaining Q3 vertices
                remaining = [x for x in range(8) if x != qa and x != qb]
                
                def backtrack(idx):
                    if idx == len(remaining):
                        return True
                    q3_v = remaining[idx]
                    for candidate in range(n):
                        if candidate in used:
                            continue
                        # Check compatibility with already assigned neighbors
                        compatible = True
                        for assigned_q3, assigned_sv in assignment.items():
                            if assigned_q3 in q3_adj[q3_v]:
                                if not adj[candidate][assigned_sv]:
                                    compatible = False
                                    break
                        if compatible:
                            used.add(candidate)
                            assignment[q3_v] = candidate
                            if backtrack(idx + 1):
                                return True
                            used.remove(candidate)
                            del assignment[q3_v]
                    return False
                
                if backtrack(0):
                    return False  # Adding this edge creates a Q3
        
        return True  # Adding this edge does not create a Q3
    
    # Start with K_{3,13}: vertices 0,1,2 in part A, vertices 3-15 in part B
    adj = [[0]*n for _ in range(n)]
    edges = []
    
    # Add all edges between part A (0,1,2) and part B (3-15)
    for a in range(3):
        for b in range(3, n):
            adj[a][b] = 1
            adj[b][a] = 1
            edges.append([a, b])
    
    # Now greedily add all other possible edges
    all_possible_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                all_possible_edges.append((i, j))
    
    # Sort edges: prioritize edges that are "easier" to add
    # Heuristic: edges within smaller partitions first (less likely to create Q3)
    # Part A has 3 vertices, Part B has 13 vertices
    # Edges within A: (0,1), (0,2), (1,2) - very safe
    # Edges within B: many, riskier
    # Cross edges already added
    
    # Priority: within-A edges first, then cross (already done), then within-B
    def edge_priority(edge):
        i, j = edge
        in_a = lambda x: x < 3
        if in_a(i) and in_a(j):
            return 0  # Highest priority
        elif not in_a(i) and not in_a(j):
            return 1  # Lower priority
        else:
            return 2  # Shouldn't happen (cross edges already added)
    
    all_possible_edges.sort(key=edge_priority)
    
    best_edges = list(edges)
    best_count = len(edges)
    
    for edge in all_possible_edges:
        if time.time() > deadline - 5:
            break
        
        u, v = edge
        # Try adding this edge
        adj[u][v] = 1
        adj[v][u] = 1
        edges.append([u, v])
        
        # Check if Q3-free
        if check_q3_free_fast(adj, (u, v)):
            best_edges = list(edges)
            best_count = len(edges)
        else:
            # Revert
            adj[u][v] = 0
            adj[v][u] = 0
            edges.pop()
    
    return best_edges
# EVOLVE-BLOCK-END
