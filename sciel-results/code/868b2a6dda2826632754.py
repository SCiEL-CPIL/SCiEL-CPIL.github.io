"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

_DEADLINE = 140  # safety margin before 150s kill

def _has_c5(n, adj):
    """Check if graph has a 5-cycle as a subgraph. Returns True if found."""
    # A 5-cycle exists iff there is a cycle of length 5.
    # We look for two vertices at distance 2 that share a common neighbor at distance 4... 
    # Actually, simplest: check all 5-tuples? Too slow for n=35.
    # Better: for each pair (u,v) with dist=2, check if there's a path of length 3 from u to v 
    # that doesn't use the middle vertex, forming a 5-cycle? No, that's a 4-cycle with a chord or something.
    # A 5-cycle is a cycle of exactly 5 vertices. 
    # We can check: for each vertex u, look at its neighbors, then for each pair of neighbors (a,b),
    # check if there's a path of length 2 from a to b through a vertex not u and not a/b? No.
    # Let's use: a 5-cycle exists iff there exist vertices v0,v1,v2,v3,v4 distinct with edges 
    # v0-v1, v1-v2, v2-v3, v3-v4, v4-v0.
    # For n up to 35, we can try a smarter check.
    # If the graph is C5-free, then for any two vertices at distance 2, they cannot have a common 
    # neighbor that is also at distance 2 from the other... 
    # Actually, let's just do a BFS-based check for cycles of length 5.
    # For each edge (u,v), remove it and check if there's a path of length 4 from u to v.
    # That would form a 5-cycle.
    for u in range(n):
        for v in adj[u]:
            if u >= v:
                continue
            # Check if there's a path of length 4 from u to v avoiding the direct edge
            # BFS from u, tracking distance, stop at dist 4
            # But we need to avoid using edge (u,v)
            visited = [[False]*(n) for _ in range(5)]
            visited[0][u] = True
            found = False
            for dist in range(1, 5):
                for node in range(n):
                    if not visited[dist-1][node]:
                        continue
                    for nb in adj[node]:
                        if dist == 4 and nb == v and node != u:
                            found = True
                            break
                        if dist == 4 and nb == v and node == u:
                            # This would be a 2-cycle via u, not a 5-cycle
                            continue
                        if not visited[dist][nb]:
                            visited[dist][nb] = True
                if found:
                    break
            if found:
                return True
    return False

def _has_independent_set_k(n, adj, k):
    """Check if there's an independent set of size k. Returns True if found."""
    # Use Bron-Kerbosch on the complement graph to find cliques of size k
    # Or use a simple backtracking
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
    
    # Find clique of size k in complement graph
    def bronk(R, P, X):
        if len(R) >= k:
            return True
        if not P and not X:
            return False
        # Choose pivot
        if P or X:
            pivot = next(iter(P | X))
            candidates = P - comp_adj[pivot]
        else:
            candidates = P
        for v in list(candidates):
            if bronk(R | {v}, P & comp_adj[v], X & comp_adj[v]):
                return True
            P.remove(v)
            X.add(v)
        return False
    
    return bronk(set(), set(range(n)), set())

def construct():
    start_time = time.time()
    
    # Best known: 8 disjoint K4s gives n=32, alpha=8, no C5
    # Let's start with that and see if we can do better.
    # 8 disjoint K4s:
    def make_disjoint_k4(num_copies):
        n = 4 * num_copies
        edges = []
        for b in range(num_copies):
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append([4*b+i, 4*b+j])
        return {'n': n, 'edges': edges}
    
    # 8 copies gives n=32
    best_result = make_disjoint_k4(8)
    
    # Check if we can add more vertices
    # The limit is 35, so we can try n=33, 34, 35
    # But we need to verify no C5 and no independent set of 9.
    
    # Let's try a different approach: use a base graph and blow up vertices
    # C4 is C5-free. Blow up C4 with parts of size s1,s2,s3,s4
    # Independent set in blow-up of C4: max independent set = max(s1+s3, s2+s4)
    # (since in C4, the independent sets are {0,2} and {1,3})
    # So we need max(s1+s3, s2+s4) <= 8
    # Total n = s1+s2+s3+s4
    # To maximize n subject to s1+s3 <= 8 and s2+s4 <= 8: n <= 16
    # That's worse than 32.
    
    # K4 is C5-free. Blow up K4 with parts of size s1,s2,s3,s4
    # Independent set in blow-up of K4: max(s1,s2,s3,s4)
    # (since in K4, independent sets are single vertices)
    # So we need max(si) <= 8
    # Total n = s1+s2+s3+s4
    # To maximize: set all si = 8, n = 32. Same as 8 disjoint K4s? No, this is one blow-up.
    # Wait, 8 disjoint K4s is different. In 8 disjoint K4s, independent set = 8 (one from each K4).
    # In a blow-up of K4 with parts of size 8, independent set = 8 (one part).
    # Both give n=32.
    
    # Can we mix? Use some K4s and some other structures?
    # What about K5? K5 has C5 as a subgraph? K5 contains all cycles, including C5. So no.
    
    # What about a complete bipartite graph K_{a,b}? 
    # K_{a,b} is C5-free iff min(a,b) <= 1 or... actually K_{a,b} has C4 if a,b >= 2, but C5?
    # A C5 in K_{a,b} would need 5 vertices alternating between the two parts. 
    # But K_{a,b} is bipartite, so all cycles have even length. So K_{a,b} is C5-free for all a,b.
    # Independent set in K_{a,b}: max(a,b)
    # So we need max(a,b) <= 8, n = a+b <= 16. Worse.
    
    # What about a complete graph K_m? K_m is C5-free iff m <= 4 (since K5 contains C5).
    # So K4 is the largest complete C5-free graph.
    
    # The 8 disjoint K4s construction seems strong. n=32.
    # Can we get n=33, 34, or 35?
    # The problem states the best known has 32 vertices.
    # Let's just return the 8 disjoint K4s construction.
    
    # But wait, maybe there's a non-disjoint construction with more vertices?
    # The task says "the best known has 32 vertices and no valid graph has more than 35."
    # So maybe 33, 34, or 35 are possible but unknown? Or maybe 32 is the max?
    # Let's just go with 32 for now.
    
    return best_result
# EVOLVE-BLOCK-END
