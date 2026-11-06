import random
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 148  # Safety margin (evaluator kills at 150s)
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set(all_edges)
    
    # Helper to check if a set of 8 vertices induces a Q3 in the current graph
    # This is the expensive part. We need to be efficient.
    # A Q3 is a 3-regular bipartite graph on 8 vertices.
    # To check if a specific set of 8 vertices S contains a Q3:
    # We can check if the induced subgraph G[S] contains a Q3.
    # Since G[S] is small, we can just check if it's isomorphic to Q3?
    # No, G[S] might have more edges than Q3 (up to 28).
    # A Q3 is a subgraph of G[S] if there exists a mapping of Q3 vertices to S such that all edges are in G.
    # This is equivalent to: Does G[S] contain a subgraph isomorphic to Q3?
    # For 8 vertices, we can precompute the 840 edge sets that form a Q3 in K8.
    # Let's do that.
    
    # Generate all 840 Q3 edge sets on vertices 0..7
    # Q3 vertices can be represented by binary tuples (0,0,0) to (1,1,1)
    # Edges connect vertices differing in exactly 1 bit.
    q3_vertices = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    q3_edges_canonical = []
    for i in range(8):
        for j in range(i+1, 8):
            v1 = q3_vertices[i]
            v2 = q3_vertices[j]
            diff = sum(1 for a, b in zip(v1, v2) if a != b)
            if diff == 1:
                q3_edges_canonical.append((i, j))
                
    # There are 48 automorphisms of Q3.
    # We need to generate all distinct edge sets.
    # We can generate all permutations of 0..7, map the canonical edges, and store unique sets.
    q3_edge_sets = set()
    for perm in range(40320): # 8!
        # Generate permutation
        # This is slow in pure python. 
        # Alternative: Just check if G[S] is isomorphic to Q3?
        # Actually, checking if G[S] *contains* a Q3 is harder.
        # But wait, if G[S] has 12 edges and is 3-regular, it IS Q3 (since it's unique).
        # If G[S] has >12 edges, it might contain a Q3.
        # The condition "contains a Q3" is equivalent to "there exists a subset of 12 edges forming Q3".
        
        # Let's optimize the search strategy instead of brute forcing all subsets.
        # Strategy:
        # 1. Start with a dense graph (e.g. p=0.6).
        # 2. Find a Q3.
        # 3. Remove an edge.
        # 4. Repeat.
        
        # To find a Q3 faster:
        # Iterate over all 8-vertex subsets S (12870 of them).
        # For each S, check if G[S] contains a Q3.
        # How to check fast?
        # If G[S] has < 12 edges, skip.
        # If G[S] has 12 edges, check if it's 3-regular and bipartite.
        # If G[S] has > 12 edges, we need to check if it contains a Q3.
        # We can do this by:
        #   For each vertex v in S, look at its neighbors in S.
        #   If deg_S(v) < 3, v cannot be in a Q3 (as a vertex of degree 3).
        #   Wait, a vertex in a Q3 has degree 3 in the Q3, but can have higher degree in G[S].
        #   So we just need to find a subgraph isomorphic to Q3.
        
        # Let's use the "find one Q3" approach from the current program, but optimize it.
        # The current program's find_q3 is O(N^7) or something?
        # It iterates v, then 3 neighbors a,b,c.
        # Then checks if a,b,c are independent.
        # Then finds x,y,z.
        # This is relatively fast if the graph is not too dense.
        
        pass # Placeholder for logic that was in the comment above
        
    # Let's implement a faster Q3 finder.
    # Precompute all 8-vertex subsets? No, 12870 is manageable.
    # But checking each one is expensive.
    
    # Alternative: Use the current find_q3 but make it faster.
    # The current find_q3 uses set intersections.
    
    def build_adj(edge_set):
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_q3(adj):
        # Returns list of 12 edges if Q3 found, else None
        # Optimization: Check degrees first?
        # A Q3 requires 8 vertices of degree >= 3 in the induced subgraph? No.
        
        # Let's stick to the structure search.
        # v is a vertex.
        # a, b, c are 3 neighbors of v.
        # a, b, c must be pairwise non-adjacent.
        # x is neighbor of a and b (not v, not c).
        # y is neighbor of a and c (not v, not b).
        # z is neighbor of b and c (not v, not a).
        # x, y, z must be pairwise non-adjacent.
        # w is neighbor of x, y, z (not v, a, b, c).
        
        # We can precompute common neighbors?
        # For each pair (u,v), common_neighbors[u][v] = adj[u] & adj[v]
        # This takes O(N^2 * AvgDeg) to build, which is fine.
        
        common = {}
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[(u,v)] = adj[u] & adj[v]
        
        for v in range(n):
            neighbors = list(adj[v])
            if len(neighbors) < 3:
                continue
            for a, b, c in combinations(neighbors, 3):
                # Check a,b,c independent
                if b in adj[a] or c in adj[a] or c in adj[b]:
                    continue
                
                # Find x in N(a) \cap N(b) \ {v, c}
                # N(a) \cap N(b) = common[(a,b)]
                # Remove v, c
                cand_x = common[(a,b)] - {v, c}
                # Remove a, b (they are not in their own neighbor sets usually, but safe)
                cand_x -= {a, b}
                
                if not cand_x:
                    continue
                    
                # Find y in N(a) \cap N(c) \ {v, b}
                cand_y = common[(a,c)] - {v, b}
                cand_y -= {a, c}
                
                if not cand_y:
                    continue
                
                # Find z in N(b) \cap N(c) \ {v, a}
                cand_z = common[(b,c)] - {v, a}
                cand_z -= {b, c}
                
                if not cand_z:
                    continue
                
                # x, y, z must be distinct and independent
                for x in cand_x:
                    for y in cand_y:
                        if x == y: continue
                        if y in adj[x]: continue
                        
                        for z in cand_z:
                            if x == z or y == z: continue
                            if z in adj[x] or z in adj[y]: continue
                            
                            # x, y, z are independent.
                            # Find w in N(x) \cap N(y) \cap N(z) \ {v, a, b, c}
                            # N(x) \cap N(y) = common[(x,y)]
                            # N(x) \cap N(y) \cap N(z) = common[(x,y)] & common[(x,z)]
                            # Or just intersect sets.
                            cand_w = common[(x,y)] & common[(x,z)]
                            # Remove v, a, b, c, x, y, z
                            forbidden = {v, a, b, c, x, y, z}
                            cand_w -= forbidden
                            
                            for w in cand_w:
                                # Check w is connected to a,b,c? No, w is only connected to x,y,z in the Q3 structure?
                                # Let's verify the Q3 structure.
                                # Vertices: v, a,b,c, x,y,z, w
                                # Edges:
                                # v-a, v-b, v-c
                                # a-x, b-x
                                # a-y, c-y
                                # b-z, c-z
                                # x-w, y-w, z-w
                                #
                                # We checked:
                                # v-a, v-b, v-c (by construction)
                                # a-x, b-x (x in N(a) \cap N(b))
                                # a-y, c-y (y in N(a) \cap N(c))
                                # b-z, c-z (z in N(b) \cap N(c))
                                # x-w, y-w, z-w (w in N(x) \cap N(y) \cap N(z))
                                #
                                # We also need to ensure NO other edges exist in the Q3?
                                # The definition is "subgraph isomorphic to Q3".
                                # This means the 12 edges MUST exist.
                                # It does NOT mean the induced subgraph is exactly Q3.
                                # So we don't need to check for absence of other edges (like v-x, etc).
                                # We just need the 12 edges to be present.
                                
                                q3_edges = [
                                    (v, a), (v, b), (v, c),
                                    (a, x), (b, x),
                                    (a, y), (c, y),
                                    (b, z), (c, z),
                                    (x, w), (y, w), (z, w)
                                ]
                                # Normalize edges
                                q3_edges = [ (min(u,v), max(u,v)) for u,v in q3_edges ]
                                return q3_edges
        return None

    best_edges = []
    best_count = 0
    random.seed(42)
    
    # Try multiple restarts with different initial densities
    for restart in range(50):
        if time.time() > deadline - 2:
            break
        
        # Start with p=0.55
        p = 0.55 + 0.05 * (restart % 5) # Vary p slightly
        if p > 0.8: p = 0.8
        
        current_edges = set()
        for e in all_edges:
            if random.random() < p:
                current_edges.add(e)
                
        adj = build_adj(current_edges)
        
        # Phase 1: Remove Q3s
        steps = 0
        while steps < 5000:
            if time.time() > deadline - 2:
                break
            q3_edges = find_q3(adj)
            if q3_edges is None:
                break
            
            # Heuristic: Remove edge with lowest degree sum
            best_edge = None
            min_score = float('inf')
            for u, v in q3_edges:
                score = len(adj[u]) + len(adj[v])
                if score < min_score:
                    min_score = score
                    best_edge = (u, v)
            
            if best_edge in current_edges:
                current_edges.remove(best_edge)
                u, v = best_edge
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                break
            steps += 1
            
        # Phase 2: Hill Climbing (Add edges)
        # Try to add missing edges one by one.
        missing = [e for e in all_edges if e not in current_edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 2:
                break
            u, v = e
            current_edges.add(e)
            adj[u].add(v)
            adj[v].add(u)
            
            q3_edges = find_q3(adj)
            
            if q3_edges is not None:
                # Check if e is in the Q3
                e_norm = (u, v) if u < v else (v, u)
                contains_e = False
                for qe in q3_edges:
                    if qe == e_norm:
                        contains_e = True
                        break
                
                if contains_e:
                    current_edges.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    # Should not happen, but remove for safety
                    current_edges.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
