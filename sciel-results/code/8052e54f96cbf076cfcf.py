"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a C6-free graph on 40 vertices maximizing edge count.
    Strategy: Start with a strong base (e.g., 8 copies of K5 + some extra edges)
    and then greedily add edges that don't create a C6, using a fast C6 detection
    check. We also try random restarts with different initial structures if time permits.
    """
    n = 40
    deadline = time.time() + 110.0  # 10s safety margin before 120s limit
    
    def has_c6_with_edge(adj, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        A C6 containing edge (u, v) looks like:
        u - v - x - y - z - w - u
        where x != w, y != w, x != y, etc. and all distinct from u, v.
        
        Actually, a C6 is a cycle of length 6. If we add edge (u, v),
        a new C6 must use this edge. So the cycle is u-v-a-b-c-d-u.
        This means we need a path of length 4 from v to u that does not use u or v internally.
        Specifically, we need vertices a, b, c, d such that:
        v-a, a-b, b-c, c-d, d-u are edges, and u, v, a, b, c, d are all distinct.
        
        Efficient check:
        Let N(u) be neighbors of u, N(v) be neighbors of v.
        The path is v-a-b-c-d-u.
        So a is in N(v), d is in N(u).
        And there is a path of length 2 from a to d (a-b-c).
        So we need to check if there exists a in N(v) and d in N(u) such that
        dist(a, d) == 2 in the graph G \ {u, v} (excluding the edge (u,v) which isn't there yet,
        and excluding u and v from the path).
        
        Actually, the path is v-a-b-c-d-u.
        So a is adjacent to v.
        d is adjacent to u.
        b is adjacent to a and c.
        c is adjacent to b and d.
        So b-c is an edge, a-b is an edge, c-d is an edge.
        
        We can iterate over all pairs (a, d) where a in N(v) and d in N(u).
        Then check if there is a path a-b-c-d of length 2.
        This is equivalent to checking if N(a) and N(d) share a common neighbor c,
        such that c is not u, v, a, d, and b is not u, v, a, d, c.
        Wait, the path is a-b-c-d. So b is a common neighbor of a and c? No.
        a-b and b-c and c-d.
        So c is a neighbor of d and b.
        b is a neighbor of a and c.
        
        Let's rephrase: We need a path of length 3 from a to d (a-b-c-d).
        So dist(a, d) = 3 in G \ {u, v}.
        
        Algorithm:
        1. Let A = N(v) \ {u}
        2. Let D = N(u) \ {v}
        3. For each a in A, for each d in D:
           if a == d: continue (distinct vertices required)
           Check if there is a path of length 3 from a to d in G \ {u, v}.
           This means there exist b, c such that a-b, b-c, c-d are edges,
           and b, c not in {u, v, a, d}, b != c.
           
        To speed this up:
        Precompute for each vertex x, the set of vertices at distance 2 from x (excluding u, v).
        Let S2[x] = { z | exists y: x-y and y-z, y,z not in {u,v,x} }
        Then we need to check if d is in S2[a] and the path doesn't use u or v.
        But S2 calculation depends on u, v.
        
        Alternative:
        For a fixed (u, v):
        Let N_u = N(u), N_v = N(v).
        We want to know if there is a path v-a-b-c-d-u.
        This is equivalent to: exists a in N_v, d in N_u such that a and d have a common neighbor at distance 2?
        No, distance 3.
        
        Let's use a different approach:
        Remove u and v from the graph.
        Let G' be the induced subgraph on V \ {u, v}.
        We need to check if there is a path of length 3 between some a in N(v) and some d in N(u).
        Actually, the path is a-b-c-d. So a and d are at distance 3 in G'.
        
        We can precompute the square of the adjacency matrix of G' to find distance 2,
        but distance 3 is harder.
        
        Let's stick to a simpler check:
        For each a in N(v):
          For each b in N(a) \ {u, v, a}:
            For each c in N(b) \ {u, v, a, b}:
              For each d in N(c) \ {u, v, a, b, c}:
                If d in N(u) and d != a:
                  return True
        This is O(deg(v) * deg(a) * deg(b) * deg(c)), which might be too slow.
        
        Optimization:
        Precompute for each vertex x, the set of vertices at distance 2 from x in G \ {u, v}.
        Let S2[x] = { z | dist(x, z) == 2 in G \ {u, v} }
        Then we need to check if there exists a in N(v) and d in N(u) such that d in S2[a] and a != d.
        But S2 depends on u, v.
        
        Let's try a different angle:
        The number of C6s in a graph is related to the trace of A^6.
        But we need a specific check for adding one edge.
        
        Let's use the following:
        When adding edge (u, v), a new C6 is formed iff there is a path of length 4 from u to v in G.
        Path: u-d-c-b-a-v.
        So we need to check if dist(u, v) == 4 in G.
        
        How to check dist(u, v) == 4 efficiently?
        We can do a BFS from u up to depth 4, avoiding v, and see if we reach v.
        But BFS is O(n^2) in worst case, which is 1600 operations.
        With ~1000 edges to test, that's 1.6 million operations. This is fast in Python if optimized.
        
        Let's implement BFS for distance 4.
        """
        # Check if there is a path of length 4 from u to v in the current graph G.
        # Path: u - x - y - z - v, where x, y, z are distinct and not u, v.
        # We can do a BFS from u, up to depth 3, and see if any node at depth 3 is adjacent to v.
        
        # To make it faster, we can precompute neighbors as sets.
        # adj is a list of sets.
        
        # BFS from u
        # Level 0: {u}
        # Level 1: N(u) \ {v}
        # Level 2: N(N(u)) \ {u, v}
        # Level 3: N(N(N(u))) \ {u, v}
        # Then check if any node in Level 3 is in N(v).
        
        # But we must ensure the path is simple (no repeated vertices).
        # BFS naturally finds shortest paths, which are simple.
        # So if dist(u, v) == 4, then there is a simple path of length 4.
        
        # Implementation:
        # visited = {u}
        # current = {u}
        # for step in range(4):
        #   next_level = set()
        #   for node in current:
        #     for neighbor in adj[node]:
        #       if neighbor not in visited and neighbor != v:
        #         next_level.add(neighbor)
        #         visited.add(neighbor)
        #   current = next_level
        #   if step == 3:
        #     # Check if any node in current is adjacent to v
        #     for node in current:
        #       if v in adj[node]:
        #         return True
        #     return False
        # return False
        
        # This is O(n^2) per check.
        # Let's optimize by using bitmasks or just sets.
        
        # Actually, we can do:
        # Let L0 = {u}
        # L1 = N(u) \ {v}
        # L2 = (N(L1)) \ {u, v}
        # L3 = (N(L2)) \ {u, v}
        # Then check if L3 intersects N(v).
        
        # But N(L) = union of N(x) for x in L.
        
        # Let's code this.
        
        # adj is a list of sets of neighbors.
        
        # To avoid modifying adj, we just read from it.
        
        # Note: u and v are not connected yet.
        
        # Level 1
        L1 = set()
        for x in adj[u]:
            if x != v:
                L1.add(x)
        
        # Level 2
        L2 = set()
        for x in L1:
            for y in adj[x]:
                if y != u and y != v and y not in L1: # y not in L1 is not necessary for BFS, but for simple path?
                    # Actually, BFS level 2 should be nodes at distance exactly 2.
                    # So y should not be in L0 or L1.
                    # But we only care if we can reach v in one more step.
                    # So we just need to find nodes at distance 3.
                    # Let's keep it simple: collect all nodes at distance <= 3, then check.
                    L2.add(y)
        
        # Wait, the above L2 includes nodes at distance 1 as well if they are neighbors of L1 nodes.
        # We need nodes at distance exactly 3.
        
        # Let's do proper BFS layers.
        visited = {u}
        current = {u}
        
        for step in range(3):
            next_level = set()
            for node in current:
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        if neighbor == v:
                            # We found v at distance step+1
                            # But we need distance 4.
                            # So if step+1 < 4, continue.
                            # If step+1 == 4, return True.
                            # But here step is 0, 1, 2. So step+1 is 1, 2, 3.
                            # So we don't return yet.
                            pass
                        else:
                            next_level.add(neighbor)
                            visited.add(neighbor)
            current = next_level
            if not current:
                return False
        
        # Now current contains nodes at distance 3 from u.
        # Check if any of them is adjacent to v.
        for node in current:
            if v in adj[node]:
                return True
        
        return False

    def count_edges(edges):
        return len(edges)

    def try_greedy_addition(adj, edges, deadline):
        """
        Greedily add edges that don't create a C6.
        """
        n = 40
        # Generate all possible edges not in the graph
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        # Shuffle for randomness
        random.shuffle(all_edges)
        
        added = 0
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if (u, v) in edges or (v, u) in edges:
                continue
            # Check if adding (u, v) creates a C6
            if not has_c6_with_edge(adj, u, v):
                # Add the edge
                adj[u].add(v)
                adj[v].add(u)
                edges.add((u, v))
                added += 1
        
        return added

    # Initial graph: 8 copies of K5
    adj = [set() for _ in range(n)]
    edges = set()
    
    for k in range(8):
        for a in range(5):
            for b in range(a+1, 5):
                u = 5*k + a
                v = 5*k + b
                adj[u].add(v)
                adj[v].add(u)
                edges.add((u, v))
    
    best_edges = edges.copy()
    best_adj = [s.copy() for s in adj]
    
    # Try greedy addition from this base
    added = try_greedy_addition(adj, edges, deadline)
    
    if len(edges) > len(best_edges):
        best_edges = edges.copy()
        best_adj = [s.copy() for s in adj]
    
    # Try random restarts
    # We can try different base graphs.
    # For example, start with a random graph with some edges, then greedy add.
    
    # But let's first see how good the greedy addition is.
    
    # If time remains, try more sophisticated search.
    
    # Convert best_edges to list of lists
    result = [[u, v] for u, v in best_edges]
    
    return result
# EVOLVE-BLOCK-END
