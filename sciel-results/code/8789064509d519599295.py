import numpy as np
import time
import random

N = 31
E = 91
DEADLINE_SECONDS = 115.0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_SECONDS
    
    # 1. Generate points of PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            points.append((x, y, 1))
    for x in range(5):
        points.append((x, 1, 0))
    points.append((1, 0, 0))
    
    # 2. Construct Polarity Graph (Orthogonality Graph)
    # Edge if dot == 0
    adj = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                
    # Current edge count
    current_edges = np.sum(adj) // 2
    
    # Helper to count C4s
    def count_c4(adj):
        A2 = adj @ adj
        i, j = np.triu_indices(N, k=1)
        common_counts = A2[i, j]
        terms = common_counts * (common_counts - 1) // 2
        total = np.sum(terms)
        return total // 2
        
    current_c4 = count_c4(adj)
    
    # 3. Adjust edge count to E=91
    # If current_edges > 91, remove edges that are part of the most C4s.
    # If current_edges < 91, add edges that create the fewest new C4s.
    
    def get_edges(adj):
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 1:
                    edges.append((i,j))
        return edges
        
    def add_edge(adj, u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        
    def remove_edge(adj, u, v):
        adj[u, v] = 0
        adj[v, u] = 0
        
    # Greedy adjustment
    while current_edges > E:
        # Find edge in most C4s
        best_edge = None
        max_c4_reduction = -1
        # Approximation: remove edge with high degree product or part of many triangles?
        # Simple: remove random edge or edge with highest degree sum
        # Better: iterate all edges, calculate delta C4
        # This is slow. Use heuristic: remove edge with highest degree.
        degrees = adj.sum(axis=1)
        # Find edge (u,v) with max degrees[u]+degrees[v]
        max_sum = -1
        u, v = 0, 1
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 1:
                    s = degrees[i] + degrees[j]
                    if s > max_sum:
                        max_sum = s
                        u, v = i, j
        remove_edge(adj, u, v)
        current_edges -= 1
        current_c4 = count_c4(adj)
        
    while current_edges < E:
        # Find non-edge with min common neighbors
        best_edge = None
        min_common = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                if adj[i,j] == 0:
                    common = np.sum(adj[i] & adj[j])
                    if common < min_common:
                        min_common = common
                        best_edge = (i, j)
                        if min_common == 0:
                            break
            if min_common == 0:
                break
        if best_edge:
            add_edge(adj, best_edge[0], best_edge[1])
            current_edges += 1
            current_c4 = count_c4(adj)
            
    # 4. Local Search to minimize C4s
    # Swap an existing edge with a non-edge if it reduces C4s
    best_c4 = current_c4
    no_improvement_count = 0
    
    while time.time() < deadline:
        improved = False
        # Randomize order to escape local minima
        edges = get_edges(adj)
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            # Try removing (u,v)
            remove_edge(adj, u, v)
            c4_removed = count_c4(adj)
            
            # Try adding a random non-edge
            non_edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if adj[i,j] == 0:
                        non_edges.append((i,j))
                        
            if not non_edges:
                add_edge(adj, u, v)
                break
                
            # Pick a few candidates to add
            random.shuffle(non_edges)
            candidates = non_edges[:10]
            
            best_add = None
            best_c4_after = float('inf')
            
            for a, b in candidates:
                add_edge(adj, a, b)
                c4_after = count_c4(adj)
                remove_edge(adj, a, b)
                if c4_after < best_c4_after:
                    best_c4_after = c4_after
                    best_add = (a, b)
                    
            if best_add and best_c4_after < best_c4:
                # Commit the swap
                add_edge(adj, best_add[0], best_add[1])
                best_c4 = best_c4_after
                improved = True
                no_improvement_count = 0
            else:
                # Restore (u,v)
                add_edge(adj, u, v)
                
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > 50:
                break
                
    return get_edges(adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
    adj = np.zeros((N, N), dtype=int)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    def count_c4_final(adj):
        A2 = adj @ adj
        i, j = np.triu_indices(N, k=1)
        common_counts = A2[i, j]
        terms = common_counts * (common_counts - 1) // 2
        total = np.sum(terms)
        return total // 2
    print(f"C4s: {count_c4_final(adj)}")
