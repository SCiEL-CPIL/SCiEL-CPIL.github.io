import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = 16
    # Precompute all possible Q3 embeddings (subgraphs isomorphic to Q3)
    # A Q3 is a cube: 8 vertices, 12 edges.
    # Vertices can be represented as bitmasks 0..7.
    # Edges connect masks differing by exactly one bit.
    q3_edges = []
    for u in range(8):
        for v in range(u+1, 8):
            if (u ^ v).bit_count() == 1:
                q3_edges.append((u, v))
    
    # There are 16! / 8! ways to map Q3 vertices to graph vertices?
    # Actually, we just need to check if a set of 8 vertices induces a Q3.
    # Or better: check if adding an edge creates a Q3.
    
    # A faster check: 
    # A graph G contains Q3 iff there exists a set of 8 vertices V' such that 
    # the subgraph induced by V' is isomorphic to Q3.
    
    # Even faster for incremental addition:
    # When adding edge (u, v), a new Q3 is formed if there exists a set of 6 other vertices
    # such that {u, v} + these 6 form a Q3 containing edge (u, v).
    # In Q3, every edge is part of exactly 2 squares (cycles of length 4).
    # So, if we add (u, v), we check if there are two disjoint paths of length 2 
    # from u to v? No, that's for squares.
    # In Q3, edge (0,1) is in square (0,1,3,2) and (0,1,5,4).
    # So, adding (u,v) creates a Q3 if there exist two distinct "witness" structures.
    # Actually, it's simpler to just check all 8-vertex subsets if the graph is small,
    # but 16 choose 8 is 12870. That's fast enough for a check function.
    
    def is_q3_free(adj):
        # adj is a 16x16 boolean matrix
        # Check all 8-subsets
        for verts in itertools.combinations(range(n), 8):
            # Check if induced subgraph on verts is isomorphic to Q3
            # Q3 properties: 8 vertices, 12 edges, bipartite, 3-regular.
            # 1. Check degrees in induced subgraph
            degs = [0]*8
            edges_in_sub = 0
            for i, u in enumerate(verts):
                for j, v in enumerate(verts):
                    if i < j and adj[u][v]:
                        degs[i] += 1
                        degs[j] += 1
                        edges_in_sub += 1
            if edges_in_sub != 12:
                continue
            if any(d != 3 for d in degs):
                continue
            # 2. Check bipartiteness (Q3 is bipartite)
            # 3. Check no triangles (Q3 is triangle-free)
            # If 3-regular, 8 vertices, 12 edges, and triangle-free, is it Q3?
            # The only 3-regular triangle-free graph on 8 vertices is Q3.
            # Let's verify triangle-free.
            has_triangle = False
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[verts[i]][verts[j]]:
                        for k in range(j+1, 8):
                            if adj[verts[i]][verts[k]] and adj[verts[j]][verts[k]]:
                                has_triangle = True
                                break
                        if has_triangle:
                            break
                if has_triangle:
                    break
            if not has_triangle:
                return False
        return True

    # Optimize is_q3_free by precomputing?
    # 12870 subsets * (8*8 checks) is ~1M ops. Very fast.
    # We can do this many times.

    # Strategy: Start with K3,13 (39 edges).
    # Try to add edges greedily.
    # Order missing edges by some heuristic? Or random?
    # Let's try a local search:
    # 1. Start with K3,13.
    # 2. While time allows:
    #    a. Pick a random missing edge (u,v).
    #    b. Add it.
    #    c. If Q3-free, keep it.
    #    d. Else, remove it.
    # This is slow because rejection rate is high.
    
    # Better: 
    # 1. Start with empty graph or sparse.
    # 2. Add edges in random order, keeping if valid.
    # 3. Then try to "repair" or "swap".
    
    # Let's try a different start: Random Graph G(16, 0.4) ~ 50 edges.
    # Filter out Q3s?
    
    # Actually, the maximum number of edges in a Q3-free graph on 16 vertices is likely around 48-50.
    # K_{4,12} has 48 edges. Is K_{4,12} Q3-free?
    # Q3 is bipartite with parts of size 4.
    # If we have K_{4,12}, any Q3 subgraph would need 4 vertices from Part A and 4 from Part B.
    # Since Part A has only 4 vertices, the subgraph on Part A would be K4? No, Q3 is not K4.
    # Q3 induced on one part is 4 isolated vertices.
    # So K_{4,12} contains Q3 if we can find 4 vertices in B such that the connections to A form a Q3.
    # In K_{4,12}, every vertex in B is connected to every vertex in A.
    # So the subgraph induced by A and any 4 vertices in B is K_{4,4}.
    # Does K_{4,4} contain Q3? Yes. Q3 is a subgraph of K_{4,4}.
    # So K_{4,12} is NOT Q3-free.
    
    # K_{3,13} is Q3-free because Q3 needs 4 vertices on each side.
    
    # What about K_{4,12} minus some edges?
    # Or a different structure?
    
    # Let's use the Greedy approach with the fast checker.
    
    adj = [[False] * n for _ in range(n)]
    
    # Initialize with K3,13
    for u in range(3):
        for v in range(3, 16):
            adj[u][v] = True
            adj[v][u] = True
            
    current_edges = 39
    
    missing = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                missing.append((i,j))
                
    random.shuffle(missing)
    
    # Greedy addition
    # We want to maximize edges.
    # Try adding edges from missing list.
    # If adding an edge creates a Q3, skip it.
    
    # To improve, we can try to "swap":
    # Remove an edge e1, add two edges e2, e3 if valid.
    
    # Let's stick to simple greedy first, then random restarts if time permits.
    
    for i in range(len(missing)):
        if time.time() > deadline - 5:
            break
        u, v = missing[i]
        if adj[u][v]:
            continue
        adj[u][v] = True
        adj[v][u] = True
        if not is_q3_free(adj):
            adj[u][v] = False
            adj[v][u] = False
        else:
            current_edges += 1
            # Remove from missing list? 
            # It's still in the list, but adj check handles it.
            
    # Local Search Phase: 
    # Try to find a better configuration by swapping.
    # Pick a random existing edge to remove, and try to add two missing edges.
    
    existing = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                existing.append((i,j))
                
    missing = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                missing.append((i,j))
                
    # Time budget for local search
    ls_deadline = deadline - 2
    
    while time.time() < ls_deadline:
        random.shuffle(existing)
        random.shuffle(missing)
        
        found_swap = False
        for e1 in existing:
            if time.time() > ls_deadline:
                break
            u1, v1 = e1
            # Temporarily remove e1
            adj[u1][v1] = False
            adj[v1][u1] = False
            
            # Try to add two edges from missing
            added = []
            for e2 in missing:
                if time.time() > ls_deadline:
                    break
                u2, v2 = e2
                if adj[u2][v2]:
                    continue
                # Check if adding e2 is valid in current state (without e1)
                adj[u2][v2] = True
                adj[v2][u2] = True
                if is_q3_free(adj):
                    added.append(e2)
                    if len(added) == 2:
                        break
                else:
                    adj[u2][v2] = False
                    adj[v2][u2] = False
            
            if len(added) == 2:
                # We added 2 edges, removed 1. Net +1.
                found_swap = True
                # Update existing and missing lists
                existing.append(added[0])
                existing.append(added[1])
                existing.remove(e1)
                # Missing list update is tricky, we can just rebuild or mark
                # For simplicity, we'll just continue the loop and rely on adj checks
            else:
                # Roll back
                for e2 in added:
                    u2, v2 = e2
                    adj[u2][v2] = False
                    adj[v2][u2] = False
            
        if not found_swap:
            # Try a single edge addition if any missing edge fits?
            # We already did greedy, so probably none fit.
            # Break if no progress?
            # Let's just break to save time if no swap found in a full pass
            # But maybe a different random order helps.
            # We'll do a few restarts of the local search.
            pass

    # Final count
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges

# Quick self-test
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))
