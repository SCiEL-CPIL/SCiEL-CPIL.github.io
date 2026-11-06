"""Dense Q3-free graph on 16 vertices via random greedy + local search."""
import random
import time
import itertools
import math

# Precompute the 8 vertex subsets that form a cube (complement of a cube)
# Actually, we need subsets of size 8 that induce a Q3.
# Q3 has 8 vertices and 12 edges.

# All 8-vertex subsets of 0..15: C(16,8) = 12870
# For each, check if the induced subgraph is isomorphic to Q3.

# Let's precompute the 12870 subsets once.
SUBSETS_8 = list(itertools.combinations(range(16), 8))

def is_q3(edges, subset):
    """Check if the induced subgraph on subset is isomorphic to Q3."""
    s = set(subset)
    # Build adjacency on subset
    adj = {v: set() for v in subset}
    for u, v in edges:
        if u in s and v in s:
            adj[u].add(v)
            adj[v].add(u)
    
    # Q3 has 12 edges, 8 vertices, each vertex degree 3
    edge_count = sum(len(adj[v]) for v in subset) // 2
    if edge_count != 12:
        return False
    for v in subset:
        if len(adj[v]) != 3:
            return False
    
    # Check bipartite and check structure
    # Q3 is a 3-regular bipartite graph on 8 vertices with girth 6
    # Actually, let's just check if it's isomorphic to Q3 by comparing to a canonical form
    # Since there are only a few 3-regular bipartite graphs on 8 vertices, 
    # we can enumerate the possible Q3 structures.
    
    # Actually, the simplest check: Q3 is the unique 3-regular bipartite graph on 8 vertices
    # with no 4-cycles... wait, Q3 has 4-cycles. 
    # Let's just do a simple isomorphism check by trying to map to a standard Q3.
    
    # Standard Q3: vertices (i,j,k) for i,j,k in {0,1}
    # Map our 8 vertices to 0..7
    verts = list(subset)
    # Try all 8! = 40320 permutations? That's too many.
    # Instead, use a canonical form or invariants.
    
    # Invariant: Q3 has exactly 6 cycles of length 4 (one per face)
    # and 8 cycles of length 6... 
    # Actually, let's just check if it's bipartite and 3-regular, 
    # and that the number of 4-cycles is 6.
    
    # Check bipartite
    color = {}
    for start in subset:
        if start not in color:
            color[start] = 0
            queue = [start]
            bip = True
            while queue:
                u = queue.pop(0)
                for w in adj[u]:
                    if w not in color:
                        color[w] = 1 - color[u]
                        queue.append(w)
                    elif color[w] == color[u]:
                        bip = False
                        break
                if not bip:
                    break
        if not bip:
            return False
    
    # Count 4-cycles
    # A 4-cycle is u-v-w-x-u
    count_4cycles = 0
    for u in subset:
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w == u or w <= v:
                    continue
                for x in adj[w]:
                    if x == v or x <= w:
                        continue
                    if x in adj[u] and x != u:
                        # u-v-w-x-u
                        # Need to avoid double counting
                        if u < v < w < x or u < x < w < v:
                            count_4cycles += 1
    # This counting is tricky. Let's use a different approach.
    # Since we know Q3 is the unique 3-regular bipartite graph on 8 vertices
    # that is not the complete bipartite K_{4,4} (which has 4-regular... wait no).
    # Actually, there are only a few 3-regular bipartite graphs on 8 vertices.
    # Let's just compare the sorted adjacency matrix to a canonical Q3.
    
    # Canonical Q3 adjacency (vertices 0-7, cube bits 000-111):
    # 0: 1,2,4
    # 1: 0,3,5
    # 2: 0,6,7
    # 3: 1,2,7
    # 4: 0,5,6
    # 5: 1,4,7
    # 6: 2,4,5
    # 7: 3,5,6
    canon_adj = [
        [1,2,4],
        [0,3,5],
        [0,6,7],
        [1,2,7],
        [0,5,6],
        [1,4,7],
        [2,4,5],
        [3,5,6]
    ]
    
    # Map our vertices to 0..7
    mapping = {}
    for i, v in enumerate(verts):
        mapping[v] = i
    
    # Build adjacency in 0..7
    my_adj = [set() for _ in range(8)]
    for v in subset:
        for w in adj[v]:
            my_adj[mapping[v]].add(mapping[w])
    
    # Compare to canon_adj
    for i in range(8):
        if set(my_adj[i]) != set(canon_adj[i]):
            return False
    
    return True

def has_q3(edges):
    """Check if the graph contains a Q3 subgraph."""
    for subset in SUBSETS_8:
        if is_q3(edges, subset):
            return True
    return False

def construct():
    deadline = time.time() + 110  # Safety margin
    
    # Start with a known Q3-free graph: K_{3,13} + K_3 on the 3-side
    # Part A = {0,1,2}, Part B = {3..15}
    edges = set()
    # All pairs in A
    for i in range(3):
        for j in range(i+1, 3):
            edges.add((i, j))
    # All pairs between A and B
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
    
    current_edges = list(edges)
    
    # Try to add edges within B
    # B has 13 vertices, C(13,2) = 78 possible edges
    # Add edges one by one, checking for Q3
    b_verts = list(range(3, 16))
    all_b_edges = list(itertools.combinations(b_verts, 2))
    random.shuffle(all_b_edges)
    
    # Greedy addition
    for u, v in all_b_edges:
        if time.time() > deadline - 5:
            break
        if (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))
            if not has_q3(list(edges)):
                pass  # Keep it
            else:
                edges.discard((u, v))
    
    current_edges = list(edges)
    current_count = len(current_edges)
    
    # Local search: try to swap edges to improve
    # This is complex, so let's just do random restarts with different initial orderings
    
    best_edges = current_edges
    best_count = current_count
    
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        # Random restart
        edges = set()
        for i in range(3):
            for j in range(i+1, 3):
                edges.add((i, j))
        for a in range(3):
            for b in range(3, 16):
                edges.add((a, b))
        
        random.shuffle(all_b_edges)
        for u, v in all_b_edges:
            if time.time() > deadline - 5:
                break
            if (u, v) not in edges:
                edges.add((u, v))
                if not has_q3(list(edges)):
                    pass
                else:
                    edges.discard((u, v))
        
        if len(edges) > best_count:
            best_edges = list(edges)
            best_count = len(edges)
    
    return best_edges
