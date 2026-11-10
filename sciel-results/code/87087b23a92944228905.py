"""Optimize four-cycles in graphs with 31 vertices and 91 edges."""
import time
import random
import numpy as np

def _count_c4(adj):
    """Count four-cycles (subgraphs) using degree-of-pairs method.
    C4 count = sum over unordered pairs {u,v} of C(common_neighbors(u,v), 2) / 1
    But we need to be careful: each C4 is counted once per pair of opposite vertices.
    Actually, for a 4-cycle a-b-c-d-a, the pairs of opposite vertices are {a,c} and {b,d}.
    For pair {a,c}, common neighbors include b and d. C(2,2)=1.
    For pair {b,d}, common neighbors include a and c. C(2,2)=1.
    So each C4 is counted twice (once per pair of opposite vertices).
    Total C4 = (1/2) * sum_{u<v} C(|N(u) ∩ N(v)|, 2)
    """
    n = len(adj)
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            # Count common neighbors
            cu = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    cu += 1
            if cu >= 2:
                total += cu * (cu - 1) // 2
    return total // 2

def _count_c4_fast(adj, n):
    """Faster C4 count using bitsets or numpy."""
    total = 0
    adj_np = np.array(adj, dtype=np.uint8)
    for u in range(n):
        for v in range(u + 1, n):
            cu = np.sum(adj_np[u] & adj_np[v])
            if cu >= 2:
                total += cu * (cu - 1) // 2
    return total // 2

def _build_polarity_graph():
    """Build polarity graph from PG(2,5) - known to have few C4s."""
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    adj = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                adj[i][j] = 1
                adj[j][i] = 1
    
    edge_count = sum(sum(row) for row in adj) // 2
    return adj, edge_count, points

def _remove_random_edges(adj, num_remove, rng):
    """Remove random edges from adj, return new adj and list of removed edges."""
    n = len(adj)
    adj_new = [row[:] for row in adj]
    removed = []
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
    rng.shuffle(all_edges)
    for e in all_edges[:num_remove]:
        i, j = e
        adj_new[i][j] = 0
        adj_new[j][i] = 0
        removed.append((i, j))
    return adj_new, removed

def _add_random_edge(adj, rng, n):
    """Add a random non-existent edge, return new adj and the edge."""
    adj_new = [row[:] for row in adj]
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
    if not non_edges:
        return adj_new, None
    i, j = rng.choice(non_edges)
    adj_new[i][j] = 1
    adj_new[j][i] = 1
    return adj_new, (i, j)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    rng = random.Random(42)
    
    # Build initial polarity graph
    adj, edge_count, points = _build_polarity_graph()
    n = 31
    
    # Adjust edge count to exactly 91
    if edge_count > 91:
        # Remove edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
        rng.shuffle(all_edges)
        for e in all_edges[:edge_count - 91]:
            i, j = e
            adj[i][j] = 0
            adj[j][i] = 0
    elif edge_count < 91:
        # Add edges
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        rng.shuffle(non_edges)
        for e in non_edges[:91 - edge_count]:
            i, j = e
            adj[i][j] = 1
            adj[j][i] = 1
    
    # Verify edge count
    ec = sum(sum(row) for row in adj) // 2
    assert ec == 91, f"Edge count is {ec}, expected 91"
    
    # Get initial C4 count
    current_c4 = _count_c4_fast(adj, n)
    best_adj = [row[:] for row in adj]
    best_c4 = current_c4
    
    # Simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
        
        # Try a random move: remove one edge, add another
        adj_temp = [row[:] for row in adj]
        
        # Find an edge to remove
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj_temp[i][j]]
        if not edges:
            break
        i, j = rng.choice(edges)
        adj_temp[i][j] = 0
        adj_temp[j][i] = 0
        
        # Find a non-edge to add
        non_edges = [(a, b) for a in range(n) for b in range(a+1, n) if not adj_temp[a][b]]
        if not non_edges:
            adj = adj_temp
            continue
        a, b = rng.choice(non_edges)
        adj_temp[a][b] = 1
        adj_temp[b][a] = 1
        
        # Count C4s in temp graph
        temp_c4 = _count_c4_fast(adj_temp, n)
        delta = temp_c4 - current_c4
        
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            adj = adj_temp
            current_c4 = temp_c4
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = [row[:] for row in adj]
        
        T *= cooling
    
    # Use best found
    adj = best_adj
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    assert len(edges) == 91, f"Got {len(edges)} edges, expected 91"
    return edges

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
