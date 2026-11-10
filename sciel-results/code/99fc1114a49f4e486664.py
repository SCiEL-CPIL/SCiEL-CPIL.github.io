"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Helper: count 4-cycles in a graph given as adjacency matrix
    # A 4-cycle is a set of 4 distinct vertices {a,b,c,d} with edges ab, bc, cd, da (or any cyclic ordering)
    # Number of 4-cycles = (1/8) * sum over all pairs of vertices (u,v) of C(common_neighbors(u,v), 2)
    # Wait, that's not quite right.
    # Let's use the standard formula:
    # Number of C4s = (1/8) * sum_{u,v} (|N(u) ∩ N(v)| choose 2)
    # This works because each C4 is counted 8 times (4 starting vertices * 2 directions)
    
    def count_c4(adj):
        # adj is a numpy boolean matrix n x n
        # common_neighbors = adj @ adj.T (number of common neighbors for each pair)
        # But we need C(common, 2) for each pair, then sum, then divide by 8
        # Actually, let's be careful.
        # Let M = adj @ adj.T. M[i][j] is the number of common neighbors of i and j.
        # For each pair (i,j) with i != j, if M[i][j] >= 2, then there are C(M[i][j], 2) 4-cycles 
        # that have i and j as opposite vertices? No.
        
        # Let's re-derive.
        # A 4-cycle is v1-v2-v3-v4-v1.
        # The pairs of opposite vertices are (v1,v3) and (v2,v4).
        # For a fixed pair of opposite vertices (u,v), the number of 4-cycles where u and v are opposite 
        # is C(common_neighbors(u,v), 2).
        # Each 4-cycle has 2 pairs of opposite vertices.
        # So total 4-cycles = (1/2) * sum_{u < v} C(common_neighbors(u,v), 2)
        
        # Let's verify with a single C4: vertices 0,1,2,3. Edges (0,1),(1,2),(2,3),(3,0).
        # Opposite pairs: (0,2) and (1,3).
        # common_neighbors(0,2) = {1,3} -> size 2 -> C(2,2)=1
        # common_neighbors(1,3) = {0,2} -> size 2 -> C(2,2)=1
        # Sum = 2. Divide by 2 -> 1. Correct.
        
        # So: count = (1/2) * sum_{u < v} C(M[u][v], 2)
        # where M = adj @ adj.T
        
        # Since adj is symmetric and has 0 diagonal, M[i][i] = degree(i).
        # We only care about u < v.
        
        # Compute M = adj @ adj.T
        # M is integer-valued.
        # We need sum_{u < v} M[u][v]*(M[u][v]-1)/2
        # = (1/2) * sum_{u < v} (M[u][v]^2 - M[u][v])
        # = (1/2) * ( (1/2) * (sum_{u,v} M[u][v]^2 - sum_u M[u][u]^2) - (1/2) * (sum_{u,v} M[u][v] - sum_u M[u][u]) )
        # This might be easier to compute with vectorized ops.
        
        # Let S2 = sum_{u,v} M[u][v]^2
        # Let S1 = sum_{u,v} M[u][v]
        # Let D2 = sum_u M[u][u]^2
        # Let D1 = sum_u M[u][u]
        
        # sum_{u < v} M[u][v]^2 = (S2 - D2) / 2
        # sum_{u < v} M[u][v] = (S1 - D1) / 2
        
        # count = (1/2) * [ (S2 - D2)/2 * 1 - (S1 - D1)/2 * 1 ] / 2 ? No.
        # C(x,2) = x(x-1)/2
        # sum_{u<v} C(M[u][v],2) = sum_{u<v} (M[u][v]^2 - M[u][v])/2
        # = (1/2) * [ (S2 - D2)/2 - (S1 - D1)/2 ]
        # = (1/4) * (S2 - D2 - S1 + D1)
        
        # Total C4s = (1/2) * sum_{u<v} C(M[u][v],2) = (1/8) * (S2 - D2 - S1 + D1)
        
        M = adj @ adj.T  # This is integer, but numpy might make it float if adj is float. Let's use int.
        # adj is boolean, so M will be int if we cast.
        
        # Actually, adj @ adj.T with boolean arrays might not work as expected in numpy.
        # Let's convert to int first.
        adj_int = adj.astype(np.int32)
        M = adj_int @ adj_int.T
        
        S2 = np.sum(M * M)
        S1 = np.sum(M)
        D2 = np.sum(np.diag(M) * np.diag(M))
        D1 = np.sum(np.diag(M))
        
        count = (S2 - D2 - S1 + D1) / 8.0
        
        return int(count)
    
    # We need to find a graph with 31 vertices, 91 edges, minimizing C4s.
    # Start with a random graph or a known good structure.
    # Since n=31 is a projective plane order (q=5, n=q^2+q+1), 
    # the polarity graph of the projective plane of order 5 has 31 vertices and 
    # each vertex has degree q+1=6? No, polarity graph of PG(2,q) has q^2+q+1 vertices.
    # Each point is adjacent to q+1 points (the absolute points of the polarity + itself? No, no self-loops).
    # Actually, the standard polarity graph of PG(2,q) has edges between points P and Q if Q is on the polar line of P.
    # The number of edges is (q^2+q+1)*(q+1)/2 ? No.
    # Let's not rely on this specific construction without knowing the exact edge count.
    
    # Let's start with a random graph with 91 edges and use simulated annealing or local search.
    # Or start with a sparse random graph and add edges.
    
    # Better: Start with a random graph with 91 edges, then optimize.
    
    def random_graph_with_m_edges(n, m, seed):
        rng = np.random.RandomState(seed)
        adj = np.zeros((n, n), dtype=bool)
        edges = []
        # Generate all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(all_edges)
        for i in range(m):
            u, v = all_edges[i]
            adj[u, v] = True
            adj[v, u] = True
            edges.append((u, v))
        return adj, edges
    
    # Local search: try to swap an edge with a non-edge to reduce C4s.
    # If we remove edge (u,v) and add edge (x,y), what is the change in C4 count?
    # This is complex to compute exactly for a swap. 
    # Instead, let's do a simpler local search:
    # 1. Try removing an edge (u,v). This decreases C4s by the number of C4s containing (u,v).
    #    But we must keep exactly 91 edges, so we must add another edge (x,y).
    # 2. The net change is C4s_added_by(x,y) - C4s_removed_by(u,v).
    
    # To make this efficient, we can precompute for each edge the number of C4s it participates in.
    # But this changes as the graph changes.
    
    # Alternative: Use a greedy approach to build the graph.
    # Start with 0 edges. Add edges one by one, choosing the edge that adds the fewest new C4s.
    # This is a heuristic and may not be optimal, but it's a good starting point.
    
    def greedy_build(n, m, seed):
        rng = np.random.RandomState(seed)
        adj = np.zeros((n, n), dtype=bool)
        edges = []
        
        # Precompute all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # To speed up, we can maintain a list of "costs" for each potential edge.
        # Cost of adding edge (u,v) is the number of new C4s it creates.
        # A new C4 is created if there is a common neighbor w such that w is connected to some other common neighbor... 
        # Actually, adding edge (u,v) creates a C4 for every pair of common neighbors (w1, w2) of u and v 
        # such that w1 and w2 are connected? No.
        
        # Let's think: adding edge (u,v) creates a C4 if there exists a path u-w-x-v of length 3 
        # where w and x are distinct and not u or v. 
        # Wait, a C4 is u-v-x-y-u. If we add (u,v), we need a path v-x-y-u of length 3.
        # So we need x adjacent to v, y adjacent to u, and x-y is an edge.
        # So the number of new C4s created by adding (u,v) is the number of edges (x,y) 
        # such that x is adjacent to v, y is adjacent to u, and x != y, x != u, y != v.
        
        # This is equivalent to: for each x in N(v), for each y in N(u), if x != y and (x,y) is an edge, 
        # then it forms a C4. But we must ensure x != u and y != v, which is true since no self-loops.
        # Also, we must ensure that the C4 is new, i.e., it wasn't already there. 
        # Since we're adding (u,v), the C4 u-v-x-y-u is new if and only if (x,y) is an edge.
        
        # So cost(u,v) = |{ (x,y) in E : x in N(v), y in N(u), x != y }|
        
        # This can be computed as: 
        # Let A = adj[v] (boolean vector of neighbors of v)
        # Let B = adj[u] (boolean vector of neighbors of u)
        # We want the number of edges (x,y) where A[x] and B[y] are true, and x != y.
        # This is the number of 1s in the submatrix adj[A][:, B] minus the number of diagonal 1s 
        # where A[x] and B[x] are both true (i.e., common neighbors).
        
        # So cost(u,v) = sum_{x in N(v)} sum_{y in N(u), y != x} adj[x,y]
        # = sum_{x in N(v)} (degree_in_N(u)(x) - (1 if x in N(u) else 0))
        
        # This is still O(n^2) per edge evaluation, which is 31^2 = 961. 
        # With ~465 possible edges, this is 465 * 961 ~ 450,000 ops per step.
        # 91 steps * 450,000 = 41 million ops. This might be slow in Python.
        
        # Let's try a different approach: start with a random graph and do local search.
        return adj, edges
    
    # Let's use local search from a random start.
    # We'll maintain the adjacency matrix and the list of edges.
    # At each step, we consider all possible swaps: remove (u,v), add (x,y).
    # To make this feasible, we can restrict to a smaller neighborhood.
    
    # Actually, let's try a simpler heuristic: 
    # Start with a random graph, then repeatedly find the edge that participates in the most C4s 
    # and the non-edge that would add the fewest C4s if added, and swap them.
    
    def edge_c4_count(adj, u, v):
        # Number of C4s containing edge (u,v).
        # A C4 containing (u,v) is of the form u-v-x-y-u.
        # So we need x in N(v), y in N(u), x != y, and (x,y) is an edge.
        # This is the same as the cost of adding (u,v) to a graph without (u,v).
        # But here (u,v) is already in the graph.
        # The number of C4s containing (u,v) is the number of paths v-x-y-u of length 3.
        # This is exactly the cost we defined earlier.
        
        Nv = adj[v]
        Nu = adj[u]
        # We want sum_{x in Nv} sum_{y in Nu, y != x} adj[x,y]
        # Let's compute this efficiently.
        
        # Create a mask for x in Nv and y in Nu, x != y.
        # This is a bit tricky with numpy.
        
        # Alternative: iterate over x in Nv.
        count = 0
        for x in np.where(Nv)[0]:
            if x == u: continue
            # For this x, we need y in Nu, y != x, and adj[x,y]
            # adj[x, Nu] gives a boolean vector of neighbors of x that are in Nu.
            # We need to exclude y = x.
            # Since x is in Nv, and we're looking at adj[x, Nu], 
            # adj[x, x] is 0 (no self-loop), so we don't need to explicitly exclude x.
            # Wait, Nu is a boolean vector. adj[x, Nu] is the number of neighbors of x that are in Nu.
            # But we need to exclude y = x. Since adj[x,x] = 0, it's automatically excluded.
            # However, we must also ensure that y != v? No, y can be v? 
            # If y = v, then the path is v-x-v-u, which is not a simple path.
            # But in a C4, all vertices are distinct. So we must exclude y = v and x = u.
            # We already excluded x = u.
            # For y = v: is v in Nu? v is in Nu if (u,v) is an edge, which it is.
            # So we need to exclude y = v.
            
            # Let's be more careful.
            # We need x in N(v) \ {u}, y in N(u) \ {v}, x != y, and (x,y) in E.
            
            # Let's just iterate over all pairs (x,y) in N(v) x N(u) and check.
            # This is O(n^2) per edge, which is 961.
            # For 91 edges, that's 91 * 961 = 87,000 ops. This is fine.
        
        # Let's do it the simple way for correctness.
        Nv_list = np.where(Nv)[0]
        Nu_list = np.where(Nu)[0]
        
        for x in Nv_list:
            if x == u: continue
            for y in Nu_list:
                if y == v: continue
                if x == y: continue
                if adj[x, y]:
                    count += 1
        
        return count
    
    def nonedge_c4_count(adj, u, v):
        # Number of C4s that would be created if we added edge (u,v).
        # This is the same as edge_c4_count, but for a non-edge.
        # The formula is the same: number of paths v-x-y-u of length 3.
        return edge_c4_count(adj, u, v)
    
    # Now, let's implement the local search.
    
    def local_search(adj, edges, max_iter=10000):
        n = adj.shape[0]
        m = len(edges)
        
        # Precompute all possible edges
        all_edges = set((i, j) for i in range(n) for j in range(i+1, n))
        current_edges = set(edges)
        
        best_adj = adj.copy()
        best_edges = edges[:]
        best_count = count_c4(adj)
        
        for iter in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find the edge with the highest C4 count
            max_c4 = -1
            edge_to_remove = None
            for e in edges:
                c = edge_c4_count(adj, e[0], e[1])
                if c > max_c4:
                    max_c4 = c
                    edge_to_remove = e
            
            if edge_to_remove is None:
                break
            
            # Find the non-edge with the lowest C4 count
            min_c4 = float('inf')
            edge_to_add = None
            # To speed up, we can sample a subset of non-edges
            non_edges = list(all_edges - current_edges)
            random.shuffle(non_edges)
            # Consider at most 100 non-edges
            for e in non_edges[:100]:
                c = nonedge_c4_count(adj, e[0], e[1])
                if c < min_c4:
                    min_c4 = c
                    edge_to_add = e
            
            if edge_to_add is None:
                break
            
            # Perform the swap if it improves the count
            if min_c4 < max_c4:
                u, v = edge_to_remove
                x, y = edge_to_add
                
                adj[u, v] = False
                adj[v, u] = False
                adj[x, y] = True
                adj[y, x] = True
                
                edges.remove(edge_to_remove)
                edges.append(edge_to_add)
                
                new_count = count_c4(adj)
                if new_count < best_count:
                    best_count = new_count
                    best_adj = adj.copy()
                    best_edges = edges[:]
            else:
                # No improvement, maybe try a random swap to escape local minima
                # Or break
                break
        
        return best_adj, best_edges, best_count
    
    # Let's run multiple random starts
    best_overall_count = float('inf')
    best_overall_edges = None
    
    for seed in range(50):
        if time.time() > deadline:
            break
        
        adj, edges = random_graph_with_m_edges(n, m, seed)
        adj, edges, count = local_search(adj, edges, max_iter=5000)
        
        if count < best_overall_count:
            best_overall_count = count
            best_overall_edges = edges[:]
    
    # Convert edges to list of lists
    result = [[u, v] for u, v in best_overall_edges]
    
    # Ensure exactly 91 edges
    assert len(result) == 91
    assert len(set(tuple(sorted(e)) for e in result)) == 91
    
    return result
# EVOLVE-BLOCK-END
