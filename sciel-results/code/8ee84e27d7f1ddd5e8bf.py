"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    n = 40
    target_edges = 105
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    def has_c6(adj):
        """Check if graph contains a 6-cycle."""
        # A 6-cycle exists if there are two vertices at distance 3 with a common neighbor at distance 1.5
        # More efficiently: count paths of length 3 between pairs of vertices
        # A 6-cycle exists if there's a pair (u,v) with >=2 common neighbors at distance 2
        # Actually, let's use the standard method: 
        # For each pair of vertices (u,v), count the number of length-3 paths between them.
        # If any pair has >=2 distinct length-3 paths that form a cycle, we have a C6.
        # 
        # Better approach: A C6 exists iff there exist vertices u,v such that
        # the number of common neighbors of u and v at distance 2 is >= 2,
        # or more precisely, we can check via (A^3)[u,v] > 1 for some u != v
        # where A^3 counts walks of length 3. But walks include backtracking.
        #
        # Let's use a cleaner approach: 
        # A 6-cycle is u-v-w-x-y-z-u. 
        # This means there exist two disjoint edges (u,v) and (x,y) such that
        # there are paths of length 2 connecting them.
        #
        # Practical check: For each pair of non-adjacent vertices (u,v),
        # count the number of length-3 paths from u to v. If >= 2, and the paths
        # are "distinct" in a way that forms a cycle, we have a C6.
        #
        # Actually, the simplest reliable method for small n:
        # Enumerate all 6-cycles by checking all combinations.
        # But that's O(n^6) = 40^6 which is too slow.
        #
        # Better: Use the fact that a C6 exists iff there is a pair of vertices
        # at distance 3 with at least 2 distinct shortest paths of length 3.
        #
        # Let me use a different approach: 
        # Compute A^2 (number of length-2 paths between all pairs).
        # A 6-cycle u-v-w-x-y-z-u exists iff there are vertices u, x such that
        # there are at least 2 distinct intermediate paths of length 2 from u to x
        # that are "compatible" (i.e., the endpoints form edges).
        #
        # Simplest correct method for n=40:
        # For each ordered pair (u, v) with u != v, count the number of 
        # length-3 walks from u to v that don't immediately backtrack.
        # This is getting complicated. Let's just do a direct search.
        
        # Direct approach: check all 6-tuples (a,b,c,d,e,f) with all distinct
        # and edges (a,b),(b,c),(c,d),(d,e),(e,f),(f,a) exist.
        # To speed up, iterate over edges and use adjacency lists.
        
        adj_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_list[i].append(j)
        
        # For each pair of edges (a,b) and (d,e) with {a,b} disjoint from {d,e},
        # check if there's a path of length 2 from b to d and from e to a,
        # or from b to e and from d to a, etc.
        # 
        # Actually, let's just check: for each 6-cycle, it has 3 pairs of opposite edges.
        # Let's iterate over all pairs of vertices (u, v) and count length-3 paths.
        
        # Compute A^3
        A = adj.astype(float)
        A2 = A @ A
        A3 = A2 @ A
        
        # A3[u,v] = number of walks of length 3 from u to v
        # A walk of length 3 from u to v is u-a-b-v.
        # For this to be part of a 6-cycle, we need another walk u-c-d-v where
        # {a,b} and {c,d} are disjoint and the edges form a cycle.
        #
        # Key insight: A 6-cycle exists iff there exist u != v such that
        # A3[u,v] >= 2 AND the two length-3 walks are "distinct" enough.
        # But A3 counts walks, not paths, so backtracking is included.
        #
        # Let me just do a brute-force check for n=40. It should be fast enough.
        
        # Build adjacency set for O(1) lookup
        adj_set = set()
        for i in range(n):
            for j in adj_list[i]:
                adj_set.add((i, j))
        
        # Check for C6 by iterating over all possible 6-cycles
        # A 6-cycle is determined by its 6 vertices in order.
        # To avoid checking all 40P6 permutations, use the following:
        # Pick two opposite edges (a,b) and (d,e). Then c and f are determined
        # by the requirement that b-c-d and e-f-a are paths of length 2.
        
        # For each pair of disjoint edges (a,b) and (d,e):
        # Find common neighbors of b and d -> candidates for c
        # Find common neighbors of e and a -> candidates for f
        # If both have at least one candidate, and the candidates form valid cycles,
        # we have a C6.
        
        # Actually, even simpler: just check if A3[u][v] > 1 for any u != v,
        # but we need to be careful about what counts.
        
        # Let me just do the direct enumeration. For n=40, the number of 6-cycles
        # to check is manageable if we're smart about it.
        
        # Alternative: use the fact that if the graph is C6-free, then for every
        # pair of vertices (u,v), the number of length-3 paths between them is at most 1.
        # (This is a necessary condition, and for simple graphs it's also sufficient
        # when combined with other conditions, but let's just check it.)
        
        # Actually, this isn't quite right. Let me just check directly.
        
        # For each pair of non-adjacent vertices (u, v):
        # Count the number of paths of length 3 from u to v.
        # If any count >= 2, we have a C6 (or a C4 with a chord, but C4 is allowed).
        # Wait, a path of length 3 u-a-b-v doesn't necessarily form a C6 by itself.
        # We need TWO such paths that together form a cycle.
        
        # OK let me just do the straightforward check:
        # For each 6-tuple of distinct vertices (a,b,c,d,e,f), check if all 6 edges exist.
        # Number of such tuples: 40 * 39 * 38 * 37 * 36 * 35 ≈ 3.5 billion. Too slow.
        
        # Better: iterate over edges (a,b), then for each common neighbor c of b and some d,
        # then for each common neighbor e of d and some f, etc.
        
        # Let me use a different strategy: 
        # A C6 is a cycle of length 6. We can detect it by checking:
        # For each pair of vertices (u, v) at distance exactly 3, if there are 
        # at least 2 distinct shortest paths of length 3, we have a C6.
        
        # Let me compute the number of simple paths of length 3 between all pairs.
        # A simple path of length 3 from u to v is u-a-b-v where all 4 vertices are distinct
        # and u != v, a != b, etc.
        
        # For each u, v with u != v:
        #   count = 0
        #   for a in neighbors(u):
        #     for b in neighbors(v):
        #       if a != b and a != v and b != u and a is adjacent to b:
        #         count += 1
        #   if count >= 2: return True
        
        # This is O(n^2 * d^2) where d is max degree. With d ~ 5, n=40:
        # 40^2 * 25 = 40,000 operations. Very fast!
        
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    continue  # Only check non-adjacent pairs
                count = 0
                for a in adj_list[u]:
                    if a == v:
                        continue
                    for b in adj_list[v]:
                        if b == u:
                            continue
                        if a == b:
                            continue
                        if a == v or b == u:
                            continue
                        if adj[a][b]:
                            count += 1
                            if count >= 2:
                                return True
        return False
    
    # Start with a random graph
    def random_graph(n, m):
        """Generate a random graph with n vertices and m edges."""
        adj = np.zeros((n, n), dtype=int)
        edges = set()
        while len(edges) < m:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j and (min(i,j), max(i,j)) not in edges:
                edges.add((min(i,j), max(i,j)))
                adj[i][j] = 1
                adj[j][i] = 1
        return adj, list(edges)
    
    def add_edge(adj, i, j):
        adj[i][j] = 1
        adj[j][i] = 1
    
    def remove_edge(adj, i, j):
        adj[i][j] = 0
        adj[j][i] = 0
    
    def count_edges(adj):
        return int(np.sum(adj) // 2)
    
    # Initialize with a random graph with target number of edges
    adj, edges_list = random_graph(n, target_edges)
    
    # Try to improve by adding edges and removing conflicting edges
    best_adj = adj.copy()
    best_count = count_edges(best_adj)
    
    # Local search: try to add edges, and if that creates a C6, try to fix it
    # by removing an edge from the C6
    
    for iteration in range(5000):
        current_count = count_edges(adj)
        
        # Try to add a random non-existent edge
        improved = False
        for _ in range(20):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j or adj[i][j]:
                continue
            
            # Try adding edge (i, j)
            add_edge(adj, i, j)
            
            if not has_c6(adj):
                improved = True
                break
            else:
                remove_edge(adj, i, j)
        
        if not improved:
            # Try removing an edge to see if we can add two others
            # Or just try a random restart
            if random.random() < 0.1:
                adj, edges_list = random_graph(n, target_edges)
                continue
        
        if count_edges(adj) > best_count:
            best_adj = adj.copy()
            best_count = count_edges(best_adj)
    
    # Convert best_adj to edge list
    result_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                result_edges.append([i, j])
    
    return result_edges
# EVOLVE-BLOCK-END
