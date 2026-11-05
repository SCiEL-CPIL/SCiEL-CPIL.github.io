import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 16
    
    # Precompute all Q3 subgraphs (sets of 8 vertices that form a Q3)
    # A Q3 is determined by its 8 vertices. We need to find all 8-subsets
    # that could form a Q3, then check edge patterns.
    
    # Actually, let's use a different approach: 
    # For checking if adding edge (u,v) creates a Q3, we need to check
    # if there's a Q3 containing (u,v). 
    # 
    # Key insight: In a Q3, every edge is in exactly 2 4-cycles (faces).
    # So if we add (u,v), we look for two 4-cycles containing (u,v) that
    # share only that edge and whose union forms a Q3.
    
    # Even better: enumerate all possible Q3 vertex sets containing (u,v).
    # A Q3 on 8 vertices: pick u,v as one edge. The Q3 structure is rigid.
    # Given u and v adjacent in Q3, the other 6 vertices are determined by
    # choosing 2 "directions" from the remaining 2 dimensions.
    # 
    # In {0,1}^3, if u=000, v=001, the Q3 is all of {0,1}^3.
    # The 6 other vertices are: 010, 011, 100, 101, 110, 111.
    # These are determined by: for each of the 2 other bit positions,
    # the vertex can have 0 or 1 in that position.
    # So the 6 vertices are: {u with bit i flipped, v with bit i flipped, 
    #   u with bits i,j flipped, v with bits i,j flipped} for the two other dims i,j.
    
    # In our abstract graph, we don't have a natural coordinate system.
    # But we can enumerate: for each pair of "intermediate" vertices.
    
    # Let me just do the C(14,6) check but make it faster with early termination.
    
    def check_q3_with_edge(adj, u, v):
        """Check if adding edge (u,v) creates a Q3. Returns True if Q3 exists."""
        # We need 6 more vertices such that the 8-vertex induced subgraph is Q3.
        # Q3 properties: 8 vertices, 12 edges, all degree 3, no triangles, connected.
        
        # For speed: first check that u has at least 2 other neighbors and v has at least 2.
        u_neighbors = [x for x in range(n) if x != u and x != v and adj[u][x]]
        v_neighbors = [x for x in range(n) if x != u and x != v and adj[v][x]]
        
        if len(u_neighbors) < 2 or len(v_neighbors) < 2:
            return False
        
        # The Q3 containing (u,v): u connects to v, a, b. v connects to u, c, d.
        # where a,b are from u's other Q3-neighbors, c,d from v's.
        # Then a-c, b-d must be edges, and there are 2 more vertices e,f.
        # a-e, b-f, c-e, d-f, e-f... let me re-derive.
        
        # Q3 with u=000, v=001:
        # u neighbors: 001(v), 010, 100 -> a=010, b=100
        # v neighbors: 000(u), 011, 101 -> c=011, d=101
        # Remaining: 110, 111 -> e=110, f=111
        # Edges: (u,v), (u,a), (u,b), (v,c), (v,d), (a,c), (b,d), (a,e), (b,f)? 
        # Wait: 010-110? differ in bit 2, yes. 100-111? differ in bits 1,2, no!
        # 100-110: differ in bit 1, yes. 100-101: differ in bit 2, yes (that's b-d).
        # Let me list all 12 edges again:
        # 000-001, 000-010, 000-100, 001-011, 001-101, 010-011, 010-110,
        # 100-101, 100-110, 011-111, 101-111, 110-111
        
        # So: a=010 connects to: u(000), c(011), e(110)
        # b=100 connects to: u(000), d(101), e(110)
        # c=011 connects to: v(001), a(010), f(111)
        # d=101 connects to: v(001), b(100), f(111)
        # e=110 connects to: a(010), b(100), f(111)
        # f=111 connects to: c(011), d(101), e(110)
        
        # So the structure: given a,b (u's other neighbors) and c,d (v's other neighbors):
        # Required edges: (a,c), (b,d), and there exist e,f such that:
        # (a,e), (b,e), (c,f), (d,f), (e,f)
        
        # So: a and b must share a common neighbor e (other than u),
        # c and d must share a common neighbor f (other than v),
        # and e-f must be an edge.
        
        # For each pair (a,b) from u_neighbors and (c,d) from v_neighbors:
        for a in u_neighbors:
            for b in u_neighbors:
                if a >= b:
                    continue
                for c in v_neighbors:
                    for d in v_neighbors:
                        if c >= d:
                            continue
                        # Check (a,c) and (b,d) exist
                        if not (adj[a][c] and adj[b][d]):
                            continue
                        # Find common neighbors of a,b (excluding u)
                        common_ab = [x for x in range(n) if x != u and x != v and x != a and x != b 
                                     and adj[a][x] and adj[b][x]]
                        # Find common neighbors of c,d (excluding v)
                        common_cd = [x for x in range(n) if x != u and x != v and x != c and x != d 
                                     and adj[c][x] and adj[d][x]]
                        # Check if there exist e in common_ab, f in common_cd with e!=f and adj[e][f]
                        for e in common_ab:
                            for f in common_cd:
                                if e == f:
                                    continue
                                if adj[e][f]:
                                    return True
        return False
    
    # Initialize with K3,13
    adj = [[False] * n for _ in range(n)]
    for u in range(3):
        for v in range(3, 16):
            adj[u][v] = True
            adj[v][u] = True
    
    missing_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                missing_edges.append((i, j))
    
    best_edge_count = sum(sum(row) for row in adj) // 2
    best_adj = [row[:] for row in adj]
    
    # Multiple random restarts
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        adj = [row[:] for row in best_adj]  # Start from best so far
        # Actually, let's restart from K3,13 each time for diversity
        adj = [[False] * n for _ in range(n)]
        for u in range(3):
            for v in range(3, 16):
                adj[u][v] = True
                adj[v][u] = True
        
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    missing.append((i, j))
        
        random.seed(attempt * 7919 + 104729)
        random.shuffle(missing)
        
        for (u, v) in missing:
            if time.time() > deadline:
                break
            if not check_q3_with_edge(adj, u, v):
                adj[u][v] = True
                adj[v][u] = True
        
        edge_count = sum(sum(row) for row in adj) // 2
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = [row[:] for row in adj]
    
    # Also try local improvement from best
    if time.time() < deadline - 5:
        adj = [row[:] for row in best_adj]
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    missing.append((i, j))
        random.shuffle(missing)
        for (u, v) in missing:
            if time.time() > deadline:
                break
            if not check_q3_with_edge(adj, u, v):
                adj[u][v] = True
                adj[v][u] = True
        edge_count = sum(sum(row) for row in adj) // 2
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = [row[:] for row in adj]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
