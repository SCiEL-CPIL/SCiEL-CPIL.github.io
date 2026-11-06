import time
import random
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n = 31
    m = 91  # number of edges required
    
    # Build PG(2,5) points as 1D subspaces of GF(5)^3
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                inv = pow(vec[first_nonzero], q-2, q)
                vec = [(x * inv) % q for x in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    def perp(p, qpt):
        return (p[0]*qpt[0] + p[1]*qpt[1] + p[2]*qpt[2]) % q == 0
    
    # Build polarity graph (93 edges)
    adj = [[False]*n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if perp(points[i], points[j]):
                adj[i][j] = adj[j][i] = True
                edges.append((i, j))
    
    polarity_edges = len(edges)
    
    if polarity_edges < m:
        # Fallback
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        return [[i, j] for i, j in all_possible[:m]]
    
    # Greedily remove (polarity_edges - m) edges with highest C4 participation
    current_edges = set(edges)
    current_adj = [row[:] for row in adj]
    
    def count_c4s(adj_mat):
        """Count total number of C4s in the graph."""
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj_mat[i][j]:
                    # Count common neighbors of i and j
                    common = 0
                    for k in range(n):
                        if k != i and k != j and adj_mat[i][k] and adj_mat[j][k]:
                            common += 1
                    count += common
        return count // 2  # Each C4 counted twice (once per pair of opposite vertices)
    
    def c4s_containing_edge(adj_mat, u, v):
        """Count C4s that contain the specific edge (u,v)."""
        count = 0
        for w in range(n):
            if w == u or w == v:
                continue
            if adj_mat[u][w]:
                for x in range(n):
                    if x == u or x == v or x == w:
                        continue
                    if adj_mat[v][x] and adj_mat[w][x]:
                        count += 1
        return count
    
    to_remove = polarity_edges - m
    for _ in range(to_remove):
        best_edge = None
        best_count = -1
        for e in current_edges:
            u, v = e
            cnt = c4s_containing_edge(current_adj, u, v)
            if cnt > best_count:
                best_count = cnt
                best_edge = e
        if best_edge is not None:
            current_edges.remove(best_edge)
            u, v = best_edge
            current_adj[u][v] = current_adj[v][u] = False
    
    # Now we have exactly m edges. Use simulated annealing to improve.
    # Move: remove edge (u,v), add non-edge (w,x) where {w,x} != {u,v}
    
    best_adj = [row[:] for row in current_adj]
    best_edges = set(current_edges)
    best_c4 = count_c4s(best_adj)
    
    # Precompute all non-edges in the polarity graph for potential additions
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    current_adj = [row[:] for row in best_adj]
    current_edges = set(best_edges)
    current_c4 = best_c4
    
    # Simulated annealing
    T_start = 10.0
    T_end = 0.01
    cooling = 0.995
    T = T_start
    step = 0
    
    while time.time() < deadline - 5.0 and T > T_end:
        step += 1
        
        # Pick a random edge to remove
        edge_list = list(current_edges)
        u, v = random.choice(edge_list)
        
        # Pick a random non-edge to add
        # First try non-edges from the polarity graph
        non_edges_in_polarity = []
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in current_edges and (j, i) not in current_edges:
                    # Check if this was in the original polarity graph
                    if perp(points[i], points[j]):
                        non_edges_in_polarity.append((i, j))
        
        if non_edges_in_polarity:
            w, x = random.choice(non_edges_in_polarity)
        else:
            # Pick any random non-edge
            while True:
                w, x = random.choice(all_pairs)
                if (w, x) not in current_edges and (x, w) not in current_edges:
                    break
        
        if (w, x) == (u, v) or (w, x) == (v, u):
            T *= cooling
            continue
        
        # Compute delta C4
        # Remove edge (u,v): C4s lost = c4s_containing_edge(current_adj, u, v)
        c4s_lost = c4s_containing_edge(current_adj, u, v)
        
        # Temporarily remove (u,v)
        current_adj[u][v] = current_adj[v][u] = False
        
        # Add edge (w,x): C4s gained = c4s_containing_edge(current_adj, w, x)
        c4s_gained = c4s_containing_edge(current_adj, w, x)
        
        # Restore (u,v) temporarily for delta calculation
        current_adj[u][v] = current_adj[v][u] = True
        
        delta = c4s_gained - c4s_lost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept move
            current_adj[u][v] = current_adj[v][u] = False
            current_edges.remove((u, v))
            current_adj[w][x] = current_adj[x][w] = True
            current_edges.add((w, x))
            current_c4 += delta
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = [row[:] for row in current_adj]
                best_edges = set(current_edges)
        
        T *= cooling
    
    # Return best found
    result = [[u, v] for u, v in best_edges]
    
    # Verify
    assert len(result) == m, f"Expected {m} edges, got {len(result)}"
    for e in result:
        u, v = e
        assert 0 <= u < n and 0 <= v < n
        assert u != v
    
    return result
# EVOLVE-BLOCK-END
