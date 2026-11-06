"""Q3-free graph on 16 vertices - maximum edge count via randomized local search."""
import random
import time

def _adj_matrix_from_edges(edges, n=16):
    adj = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    return adj

def _find_q3(adj, n=16):
    """
    Check if graph with adjacency matrix adj contains a subgraph isomorphic to Q3.
    Returns True if Q3 is found, False otherwise.
    """
    # Q3 edges
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
    
    # Precompute neighbors in Q3 for each vertex
    q3_neighbors = [[] for _ in range(8)]
    for u, v in q3_edges:
        q3_neighbors[u].append(v)
        q3_neighbors[v].append(u)

    # Pruning: If fewer than 8 vertices have degree >= 3, no Q3
    degs = [sum(adj[i]) for i in range(n)]
    if sum(1 for d in degs if d >= 3) < 8:
        return False

    # Backtracking
    # Order Q3 vertices to prune early? All have degree 3.
    # Order graph vertices by degree descending to try high-degree nodes first?
    # Or just iterate.
    
    assignment = [-1]*8
    used = [0]*n # 1 if used
    
    def backtrack(idx):
        if idx == 8:
            # Verify all edges
            for u, v in q3_edges:
                if adj[assignment[u]][assignment[v]] == 0:
                    return False
            return True
        
        # Try all unused graph vertices
        for gv in range(n):
            if used[gv]:
                continue
            
            # Check constraints with already assigned Q3 neighbors
            valid = True
            for qv in range(idx):
                if assignment[qv] != -1:
                    # If qv and idx are adjacent in Q3
                    # We need to check if (qv, idx) is in q3_edges
                    # Since we built q3_neighbors, we can check if idx is in q3_neighbors[qv]
                    if idx in q3_neighbors[qv]:
                        if adj[assignment[qv]][gv] == 0:
                            valid = False
                            break
            if not valid:
                continue
            
            assignment[idx] = gv
            used[gv] = 1
            if backtrack(idx + 1):
                return True
            assignment[idx] = -1
            used[gv] = 0
        
        return False

    # Try all starting vertices
    # Optimization: Only start with vertices that have degree >= 3
    candidates = [i for i in range(n) if degs[i] >= 3]
    for first_gv in candidates:
        assignment[0] = first_gv
        used[first_gv] = 1
        if backtrack(1):
            return True
        used[first_gv] = 0
        assignment[0] = -1
        
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n = 16
    # Start with complete graph K16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    current_edges = set(all_edges)
    
    adj = _adj_matrix_from_edges(list(current_edges), n)
    
    # Phase 1: Remove edges to make Q3-free
    # While there is a Q3, remove a random edge from it
    # To do this efficiently, we need to find a Q3 and pick an edge.
    # The current _find_q3 only returns True/False.
    # We need a version that returns the Q3 or just pick a random edge and check if it breaks all Q3s?
    # Simpler: Just remove random edges until Q3-free.
    # This is slow if we have many Q3s.
    
    # Better: Find a Q3, remove one of its edges.
    # Let's modify _find_q3 to return the assignment if found?
    # Or just use the boolean and remove a random edge from the graph?
    # If we remove a random edge, we might not break the specific Q3 we found.
    
    # Let's implement a helper to get a Q3.
    def _get_q3(adj, n=16):
        q3_edges = []
        for i in range(8):
            for j in range(i+1, 8):
                if bin(i ^ j).count('1') == 1:
                    q3_edges.append((i, j))
        
        q3_neighbors = [[] for _ in range(8)]
        for u, v in q3_edges:
            q3_neighbors[u].append(v)
            q3_neighbors[v].append(u)

        degs = [sum(adj[i]) for i in range(n)]
        if sum(1 for d in degs if d >= 3) < 8:
            return None

        assignment = [-1]*8
        used = [0]*n
        
        def backtrack(idx):
            if idx == 8:
                return assignment[:]
            
            for gv in range(n):
                if used[gv]:
                    continue
                
                valid = True
                for qv in range(idx):
                    if idx in q3_neighbors[qv]:
                        if adj[assignment[qv]][gv] == 0:
                            valid = False
                            break
                if not valid:
                    continue
                
                assignment[idx] = gv
                used[gv] = 1
                res = backtrack(idx + 1)
                if res:
                    return res
                assignment[idx] = -1
                used[gv] = 0
            
            return None

        candidates = [i for i in range(n) if degs[i] >= 3]
        random.shuffle(candidates)
        for first_gv in candidates:
            assignment[0] = first_gv
            used[first_gv] = 1
            res = backtrack(1)
            if res:
                return res
            used[first_gv] = 0
            assignment[0] = -1
            
        return None

    # Phase 1: Remove edges
    while time.time() < deadline - 30.0:
        q3 = _get_q3(adj, n)
        if not q3:
            break
        
        # q3 is a list of 8 graph vertices corresponding to Q3 vertices 0..7
        # The edges of the Q3 in the graph are (q3[u], q3[v]) for (u,v) in q3_edges
        # Pick a random edge from this Q3 to remove
        q3_edges_list = []
        for i in range(8):
            for j in range(i+1, 8):
                if bin(i ^ j).count('1') == 1:
                    gu, gv = q3[i], q3[j]
                    q3_edges_list.append((min(gu, gv), max(gu, gv)))
        
        edge_to_remove = random.choice(q3_edges_list)
        u, v = edge_to_remove
        current_edges.remove(edge_to_remove)
        adj[u][v] = 0
        adj[v][u] = 0

    # Phase 2: Local Search
    # Try to add back edges
    missing_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in current_edges:
                missing_edges.append((i, j))
    
    random.shuffle(missing_edges)
    
    # We will iterate through missing edges and try to add them.
    # If adding creates a Q3, we try to remove a different edge from that Q3 (swap).
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        random.shuffle(missing_edges)
        
        for edge in missing_edges:
            if time.time() > deadline - 5.0:
                break
            
            u, v = edge
            if (u, v) in current_edges:
                continue
                
            # Try adding
            adj[u][v] = 1
            adj[v][u] = 1
            
            if not _find_q3(adj, n):
                # Safe to add
                current_edges.add(edge)
                missing_edges.remove(edge)
                improved = True
            else:
                # Creates Q3. Try to fix by swapping.
                # Find a Q3 containing this edge?
                # _get_q3 might return a Q3 not containing (u,v).
                # We need a Q3 that contains (u,v) to ensure we break the conflict we just created.
                # Or just remove any edge from any Q3?
                # If we remove an edge from a different Q3, we might still have the new Q3.
                # So we must remove an edge from a Q3 that contains (u,v).
                
                # Let's find a Q3 containing (u,v).
                # We can modify _get_q3 or just check all Q3s?
                # Simpler: revert the addition for now.
                adj[u][v] = 0
                adj[v][u] = 0
                
                # Try a swap: Remove a random edge from the current graph, add (u,v).
                # This is expensive to check.
                # Let's just skip this edge for now in this pass.
                pass

    # Final cleanup: ensure no Q3
    # The loop above might have left us in a state with Q3 if we didn't handle swaps carefully?
    # No, we only added if `not _find_q3`.
    # So the graph is Q3-free.
    
    result = [[u, v] for u, v in sorted(current_edges)]
    return result
