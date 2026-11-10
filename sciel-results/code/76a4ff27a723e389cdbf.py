import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer before 150s limit
    
    q = 5
    n = 31
    
    # Generate 31 points of PG(2,5)
    points = []
    points.append((1, 0, 0))
    points.append((0, 1, 0))
    points.append((0, 0, 1))
    for y in range(1, q):
        points.append((1, y, 0))
    for z in range(1, q):
        points.append((1, 0, z))
    for z in range(1, q):
        points.append((0, 1, z))
    for y in range(1, q):
        for z in range(1, q):
            points.append((1, y, z))
    
    # Construct Polarity Graph (base 90 edges)
    adj = [set() for _ in range(n)]
    base_edges = set()
    
    for i in range(n):
        for j in range(i + 1, n):
            p_i = points[i]
            p_j = points[j]
            if (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                base_edges.add((min(i, j), max(i, j)))
                
    target_edges = 91
    edges_to_add = target_edges - len(base_edges)  # Should be 1
    
    # Find the best edge to add (minimizes new C4s)
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                non_edges.append((i, j))
                
    best_edge = None
    min_cost = float('inf')
    
    for u, v in non_edges:
        A = adj[v]
        B = adj[u]
        
        # Cost = number of edges (x, y) where x in A, y in B, and edge exists
        # This counts C4s that would be formed: u-x-y-v-u
        S = 0
        for x in A:
            S += len(adj[x] & B)
            
        # Subtract edges within common neighbors (these don't form C4 with u,v)
        C = A & B
        E_CC = 0
        for x in C:
            E_CC += len(adj[x] & C)
        E_CC //= 2
        
        cost = S - E_CC
        
        if cost < min_cost:
            min_cost = cost
            best_edge = (u, v)
            
    if best_edge is None:
        best_edge = non_edges[0] if non_edges else (0, 1)
        
    # Now try to improve by swapping edges
    # Start with base + best_edge
    current_edges = set(base_edges)
    current_edges.add(best_edge)
    
    # Build adjacency for current graph
    cur_adj = [set() for _ in range(n)]
    for u, v in current_edges:
        cur_adj[u].add(v)
        cur_adj[v].add(u)
    
    def count_c4s(adj_sets, edge_list):
        """Count exact number of C4 subgraphs"""
        count = 0
        # For each pair of vertices, count common neighbors
        # A C4 is a pair of vertices with >=2 common neighbors
        # Number of C4s through vertices (i,j) with k common neighbors = C(k,2)
        for i in range(n):
            Ni = adj_sets[i]
            for j in range(i + 1, n):
                Nj = adj_sets[j]
                common = Ni & Nj
                k = len(common)
                if k >= 2:
                    count += k * (k - 1) // 2
        return count
    
    best_c4_count = count_c4s(cur_adj, current_edges)
    best_edges = set(current_edges)
    
    # Try edge swaps: remove one base edge, add one non-edge
    # Focus on swaps that might reduce C4 count
    
    # Precompute: for each edge, how many C4s it participates in
    # For edge (u,v), it's in C4s u-x-y-v where x in N(u)\{v}, y in N(v)\{u}, edge x-y exists
    
    def edge_c4_contribution(adj_sets, u, v):
        """Number of C4s that include edge (u,v)"""
        Ni = adj_sets[u] - {v}
        Nj = adj_sets[v] - {u}
        count = 0
        for x in Ni:
            count += len(adj_sets[x] & Nj)
        return count
    
    # Calculate contribution of each edge
    edge_contrib = {}
    for u, v in current_edges:
        edge_contrib[(u, v)] = edge_c4_contribution(cur_adj, u, v)
    
    # Try swaps: for each base edge to remove, try non-edges to add
    # Focus on edges with high contribution and non-edges with low cost
    
    base_edge_list = list(base_edges)
    non_edge_list = list(non_edges)
    
    # Sort base edges by contribution (high first) and non-edges by potential cost
    base_edge_list.sort(key=lambda e: edge_contrib.get(e, 0), reverse=True)
    
    # Limit search to top candidates
    max_swaps_to_try = 5000
    swaps_tried = 0
    improved = True
    
    while improved and swaps_tried < max_swaps_to_try:
        improved = False
        swaps_tried += 1
        
        if time.time() > deadline - 15:
            break
        
        # Try a random base edge to remove and random non-edge to add
        rand_base_idx = random.randint(0, len(base_edge_list) - 1)
        remove_edge = base_edge_list[rand_base_idx]
        
        # Find non-edges not already in graph
        available_non_edges = [ne for ne in non_edge_list if ne not in current_edges]
        if not available_non_edges:
            break
            
        rand_non_idx = random.randint(0, len(available_non_edges) - 1)
        add_edge = available_non_edges[rand_non_idx]
        
        # Try the swap
        new_edges = set(current_edges)
        new_edges.discard(remove_edge)
        new_edges.add(add_edge)
        
        if len(new_edges) != target_edges:
            continue
            
        # Build new adjacency
        new_adj = [set() for _ in range(n)]
        for u, v in new_edges:
            new_adj[u].add(v)
            new_adj[v].add(u)
        
        new_c4 = count_c4s(new_adj, new_edges)
        
        if new_c4 < best_c4_count:
            best_c4_count = new_c4
            best_edges = set(new_edges)
            current_edges = set(new_edges)
            cur_adj = [set() for _ in range(n)]
            for u, v in current_edges:
                cur_adj[u].add(v)
                cur_adj[v].add(u)
            improved = True
            
            # Recompute contributions
            edge_contrib = {}
            for u, v in current_edges:
                edge_contrib[(u, v)] = edge_c4_contribution(cur_adj, u, v)
    
    # Return best edges found
    result = []
    for u, v in best_edges:
        result.append([min(u, v), max(u, v)])
        
    return result
