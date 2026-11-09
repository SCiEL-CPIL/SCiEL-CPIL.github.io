import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 16
    
    # Precompute Q3 structure
    # Q3 vertices are 0..7, edges between i,j if popcount(i^j)==1
    q3_edges = []
    q3_adj = [set() for _ in range(8)]
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges.append((i, j))
                q3_adj[i].add(j)
                q3_adj[j].add(i)
    
    # Initial Graph: 
    # A = {0, 1, 2}, B = {3..15}
    # Edges: All A-B, Triangle in A, Matching in B (3-4, 5-6, ..., 13-14)
    adj = [[0]*n for _ in range(n)]
    edges = []
    
    # Add K3,13
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = 1
            adj[b][a] = 1
            edges.append([a, b])
            
    # Add Triangle in A
    edges.append([0, 1]); adj[0][1]=1; adj[1][0]=1
    edges.append([1, 2]); adj[1][2]=1; adj[2][1]=1
    edges.append([0, 2]); adj[0][2]=1; adj[2][0]=1
    
    # Add Matching in B
    for i in range(3, 15, 2):
        edges.append([i, i+1])
        adj[i][i+1] = 1
        adj[i+1][i] = 1
        
    current_edge_count = len(edges)
    best_edges = list(edges)
    best_count = current_edge_count
    
    # Generate all missing edges
    missing_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 0:
                missing_edges.append((i, j))
                
    # Heuristic sorting: 
    # Edges within B are more likely to create Q3s because B is large and connected to A.
    # Edges within A are already saturated (triangle).
    # So we are mostly adding edges within B or potentially re-adding A-B (but A-B is full).
    # Wait, A-B is full. So missing edges are only within B.
    # Let's verify: 
    # Total possible edges: 16*15/2 = 120
    # Current: 39 (A-B) + 3 (A-A) + 6 (B-B) = 48
    # Missing: 120 - 48 = 72 edges, all within B.
    
    # To maximize edges, we want to add as many B-B edges as possible.
    # We iterate through them.
    
    def check_q3_free_with_new_edge(adj, u, v):
        """
        Returns True if the graph defined by adj (which includes edge u-v) is Q3-free.
        Since we only added u-v, we only need to check if u-v is part of a Q3 subgraph.
        """
        # Try all Q3 edges (a,b) and map them to (u,v)
        for a, b in q3_edges:
            # Mapping 1: a->u, b->v
            if _can_extend_q3(adj, u, v, a, b):
                return False
            # Mapping 2: a->v, b->u
            if _can_extend_q3(adj, v, u, a, b):
                return False
        return True

    def _can_extend_q3(adj, gu, gv, qa, qb):
        """
        Check if we can assign the remaining 6 Q3 vertices to graph vertices
        such that all Q3 edges are satisfied, given that qa is mapped to gu and qb to gv.
        gu and gv must be adjacent in adj (they are, by construction of the call).
        """
        # Remaining Q3 vertices
        remaining_q = [x for x in range(8) if x != qa and x != qb]
        
        # We need to assign these to distinct graph vertices from 0..15, excluding gu, gv.
        # To make it faster, we can use backtracking with forward checking.
        
        assignment = {qa: gu, qb: gv}
        used = {gu, gv}
        
        # Order remaining Q3 vertices by degree in the "unassigned" part? 
        # Or just fixed order. Fixed order is fine for 6 nodes.
        
        def backtrack(idx):
            if idx == len(remaining_q):
                return True
            
            q = remaining_q[idx]
            
            # Candidate graph vertices
            # Must be distinct from used
            # Must be adjacent in graph to all assigned neighbors of q in Q3
            
            candidates = []
            for g in range(n):
                if g in used:
                    continue
                # Check if g is adjacent to all assigned neighbors of q
                valid = True
                for neighbor_q in q3_adj[q]:
                    if neighbor_q in assignment:
                        g_neighbor = assignment[neighbor_q]
                        if adj[g][g_neighbor] == 0:
                            valid = False
                            break
                if valid:
                    candidates.append(g)
            
            # Optimization: If no candidates, return False immediately
            if not candidates:
                return False
                
            for g in candidates:
                used.add(g)
                assignment[q] = g
                
                if backtrack(idx + 1):
                    return True
                    
                used.remove(g)
                del assignment[q]
                
            return False
            
        return backtrack(0)

    # Greedy addition
    # We can try multiple orders or restarts if time allows.
    # For now, one pass.
    
    for u, v in missing_edges:
        if time.time() > deadline:
            break
            
        # Tentatively add
        adj[u][v] = 1
        adj[v][u] = 1
        
        if check_q3_free_with_new_edge(adj, u, v):
            edges.append([u, v])
            best_edges = list(edges)
            best_count = len(edges)
        else:
            # Revert
            adj[u][v] = 0
            adj[v][u] = 0
            
    # If we have time, we can try a local search: 
    # Try removing an edge and adding two others? 
    # Or just repeat the greedy pass with a different order?
    # Given the small size, the greedy pass is likely near-optimal or optimal for this structure.
    
    return best_edges
