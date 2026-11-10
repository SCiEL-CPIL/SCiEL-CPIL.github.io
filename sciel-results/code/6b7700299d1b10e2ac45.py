import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    q = 5
    # Build PG(2, q) points: 31 points
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # should be 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0 and i != j
    base_edges = []
    base_adj = [set() for _ in range(n)]
    base_edge_set = set()
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                base_edges.append((i, j))
                base_adj[i].add(j)
                base_adj[j].add(i)
                base_edge_set.add((i, j))
    
    # Count C4s in a graph given adjacency as list of sets
    def count_c4s(adj):
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    total += common * (common - 1) // 2
        return total // 2
    
    # Helper: build adjacency list of sets from edge set
    def build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    # Find best edge to add to base graph
    adj_lists = [sorted(base_adj[i]) for i in range(n)]
    
    # Build fast adjacency lookup as numpy array for speed
    is_adj = np.zeros((n, n), dtype=bool)
    for (u, v) in base_edge_set:
        is_adj[u, v] = True
        is_adj[v, u] = True
    
    # Compute cost of adding each non-edge: number of new 4-cycles
    # New 4-cycles from adding (i,j): pairs (a,b) with a~i, b~j, a~b, a!=b
    # Use matrix multiplication for speed
    # A[i,a] * A[a,b] * A[b,j] summed over a,b with a!=b
    # Actually: for each a in N(i), for each b in N(j), if a~b and a!=b, count it
    # Using numpy: for fixed i,j, sum over a!=b of A[i,a]*A[a,b]*A[b,j]
    # = (A[i,:] @ A @ A[:,j]) - sum over a of A[i,a]*A[a,a]*A[a,j] (diagonal terms, but A[a,a]=0)
    # So it's just A[i,:] @ A @ A[:,j] since diagonal is 0
    # Wait: (A @ A)[i,j] = sum_a A[i,a]*A[a,j] = number of length-2 walks from i to j
    # That's not what we want. We want sum_{a!=b} A[i,a]*A[a,b]*A[b,j]
    # = sum_a sum_b A[i,a]*A[a,b]*A[b,j] - sum_a A[i,a]*A[a,a]*A[a,j]
    # Since A[a,a]=0, the second term is 0.
    # So it's sum_a sum_b A[i,a]*A[a,b]*A[b,j] = (A @ A @ A)[i,j]? No.
    # (A @ A)[i,b] = sum_a A[i,a]*A[a,b]
    # (A @ A @ A)[i,j] = sum_b (A@A)[i,b]*A[b,j] = sum_b sum_a A[i,a]*A[a,b]*A[b,j]
    # Yes! So the cost of adding edge (i,j) is (A^3)[i,j] where A is the adjacency matrix.
    # But wait, this counts all triples (a,b) with a~i, a~b, b~j. Since a!=b is automatic
    # (if a=b then a~a which is false in simple graph), this is correct.
    
    # Compute A^3 using numpy
    A = is_adj.astype(np.int32)
    A2 = A @ A
    A3 = A2 @ A
    
    # For each non-edge (i,j), the cost is A3[i,j]
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not is_adj[i, j]:
                non_edges.append((i, j))
    
    # Sort by cost
    edge_costs = [(int(A3[i, j]), i, j) for (i, j) in non_edges]
    edge_costs.sort()
    
    best_add = None
    best_add_cost = float('inf')
    
    for cost, i, j in edge_costs:
        if time.time() > deadline - 30:
            break
        if cost < best_add_cost:
            best_add_cost = cost
            best_add = (i, j)
    
    # Start with best 91-edge graph
    current_edges = set(base_edge_set)
    if best_add:
        current_edges.add(best_add)
    
    current_adj = build_adj(current_edges)
    current_c4 = count_c4s(current_adj)
    
    # Local search: try edge swaps (remove one, add another)
    best_c4 = current_c4
    best_edges = set(current_edges)
    
    # Precompute all pairs
    all_pairs = set()
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.add((i, j))
    
    def get_non_edges(edge_set):
        return all_pairs - edge_set
    
    non_edges_current = get_non_edges(current_edges)
    
    # Build numpy adjacency for current graph
    def build_numpy_adj(edge_set):
        mat = np.zeros((n, n), dtype=bool)
        for (u, v) in edge_set:
            mat[u, v] = True
            mat[v, u] = True
        return mat
    
    # Local search iterations
    max_iterations = 200
    for it in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        improved = False
        non_edges_list = list(non_edges_current)
        
        # Try each edge removal
        for rem_edge in list(current_edges):
            if time.time() > deadline - 10:
                break
            
            rem_u, rem_v = rem_edge
            
            # For each candidate add edge, compute the delta in C4 count
            # This is expensive, so use heuristic: only consider adds with low "cost"
            # relative to the removal
            
            # Compute current adjacency as numpy
            cur_adj_np = build_numpy_adj(current_edges)
            cur_A3 = (cur_adj_np.astype(np.int32) @ cur_adj_np.astype(np.int32) @ cur_adj_np.astype(np.int32))
            
            for add_edge in non_edges_list:
                if add_edge == rem_edge:
                    continue
                
                add_u, add_v = add_edge
                
                # Heuristic: skip if the add edge has many common neighbors
                add_common = int(cur_adj_np[add_u] & cur_adj_np[add_v]).sum() if hasattr(cur_adj_np[add_u], '__and__') else int(np.sum(cur_adj_np[add_u] & cur_adj_np[add_v]))
                rem_common = int(np.sum(cur_adj_np[rem_u] & cur_adj_np[rem_v]))
                
                if add_common >= rem_common:
                    continue
                
                # Try the swap
                new_edges = set(current_edges)
                new_edges.remove(rem_edge)
                new_edges.add(add_edge)
                
                new_adj = build_adj(new_edges)
                new_c4 = count_c4s(new_adj)
                
                if new_c4 < best_c4:
                    best_c4 = new_c4
                    best_edges = new_edges
                    current_edges = new_edges
                    current_adj = new_adj
                    current_c4 = new_c4
                    non_edges_current = get_non_edges(current_edges)
                    non_edges_list = list(non_edges_current)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Return best edges found
    result = []
    for (u, v) in best_edges:
        result.append([u, v])
    
    return result[:91]
