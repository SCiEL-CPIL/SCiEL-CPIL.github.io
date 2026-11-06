"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

# Deadline guard: return best result before 145s (evaluator kills at 150s)
_DEADLINE = time.time() + 145.0
_VERTS = 18
_EDGE_COUNT = _VERTS * (_VERTS - 1) // 2  # 153

def _edge_index(u, v):
    """Map (u,v) with u<v to index in 0..152."""
    if u > v:
        u, v = v, u
    return u * 17 - u * (u + 1) // 2 + v - u - 1

def _edge_index_inv(idx):
    """Invert edge_index."""
    # u*(17 - u//2) ... let's just precompute
    pass

# Precompute edge indices
_EDGE_MAP = {}
_EDGE_LIST = []
for u in range(_VERTS):
    for v in range(u + 1, _VERTS):
        idx = len(_EDGE_LIST)
        _EDGE_LIST.append((u, v))
        _EDGE_MAP[(u, v)] = idx
        _EDGE_MAP[(v, u)] = idx

def _idx(u, v):
    return _EDGE_MAP[(u, v)]

def _is_k4_free(adj_matrix):
    """Check if graph (as 18x18 bool matrix) is K4-free."""
    for quad in combinations(range(_VERTS), 4):
        a, b, c, d = quad
        if (adj_matrix[a][b] and adj_matrix[a][c] and adj_matrix[a][d] and
            adj_matrix[b][c] and adj_matrix[b][d] and adj_matrix[c][d]):
            return False
    return True

def _largest_triangle_free(adj_matrix):
    """Find the largest triangle-free induced subgraph size via branch and bound."""
    n = _VERTS
    # Precompute for each pair whether they have a common neighbor
    # A set S is triangle-free if for every pair (u,v) in S, no w in S is adjacent to both u and v
    
    best = 0
    # We'll use a greedy upper bound and branch & bound
    # Represent candidate set as a bitmask or list
    
    # For efficiency, precompute common neighbors
    # cn[u][v] = set of w != u,v such that adj[u][w] and adj[v][w]
    cn = [[None] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            s = set()
            for w in range(n):
                if w != u and w != v and adj_matrix[u][w] and adj_matrix[v][w]:
                    s.add(w)
            cn[u][v] = s
            cn[v][u] = s
    
    # Branch and bound: try to build a triangle-free set
    # State: (current_set, available_vertices)
    # A vertex v can be added to current_set if for all u in current_set, 
    # cn[u][v] ∩ current_set is empty (i.e., no triangle)
    
    # Greedy upper bound for a given available set:
    # Actually, let's use a simpler approach: iterate over all subsets is too expensive.
    # Use backtracking with pruning.
    
    # We'll search for a triangle-free set of size > best.
    # Start with all vertices, try to find a large triangle-free subset.
    
    # Alternative: for each starting vertex, greedily build, then branch.
    # Given 18 vertices, we can do a more systematic search.
    
    # Let's use the following: try all subsets of size s from high to low,
    # but that's C(18,s) which is manageable for s up to ~12.
    
    # Actually, C(18,12) = 18564, C(18,13) = 8568, etc. Very manageable.
    
    # Check from high s down to 1
    for s in range(n, 0, -1):
        found = False
        for subset in combinations(range(n), s):
            # Check if subset induces a triangle-free graph
            tf = True
            for i in range(s):
                for j in range(i + 1, s):
                    u, v = subset[i], subset[j]
                    if cn[u][v] & set(subset):
                        tf = False
                        break
                if not tf:
                    break
            if tf:
                return s
    return 0

def _score_graph(adj_matrix):
    """Compute the score: largest triangle-free induced subgraph size."""
    if not _is_k4_free(adj_matrix):
        return _VERTS + 1  # invalid, penalize
    return _largest_triangle_free(adj_matrix)

def _greedy_construct(random_seed=None):
    """Greedy randomized construction of a K4-free graph."""
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = random
    
    # Start with empty graph, add edges greedily while maintaining K4-free
    # and trying to minimize triangle-free subgraph size.
    # Strategy: add edges in random order, skip if it creates a K4.
    # Then try to add more edges to break up large triangle-free sets.
    
    adj = [[False] * _VERTS for _ in range(_VERTS)]
    edges = []
    
    # Generate random edge order
    all_edges = [(u, v) for u in range(_VERTS) for v in range(u + 1, _VERTS)]
    rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        # Check if adding (u,v) creates a K4
        adj[u][v] = adj[v][u] = True
        # Check all common neighbors of u and v
        creates_k4 = False
        for w in range(_VERTS):
            if w == u or w == v:
                continue
            if adj[u][w] and adj[v][w]:
                # Check if w and any other common neighbor form K4
                # Actually, adding (u,v) creates K4 if there exist w,x 
                # such that w,x are both adjacent to u and v, and w-x edge exists
                # Wait, no. A K4 needs 4 vertices all mutually connected.
                # If we add (u,v), a new K4 is formed if there exist w,x 
                # where w,x are common neighbors of u,v AND w-x is an edge.
                # But also, w could be adjacent to u,v and x adjacent to u,v, 
                # and w-x edge exists.
                pass
        # Simpler check: after adding edge, verify K4-free
        # But that's O(n^4) per edge which is 153 * 3060 ≈ 468k, fine.
        
        if not _is_k4_free(adj):
            adj[u][v] = adj[v][u] = False
        else:
            edges.append([u, v])
    
    return edges, adj

def _local_search(edges, adj, max_time=30.0):
    """Local search: try edge swaps to reduce triangle-free set size."""
    global _DEADLINE
    start_time = time.time()
    best_score = _score_graph(adj)
    best_edges = list(edges)
    best_adj = [row[:] for row in adj]
    
    if best_score <= 8:
        return best_edges, best_adj
    
    # Try single edge additions/removals
    all_edges = [(u, v) for u in range(_VERTS) for v in range(u + 1, _VERTS)]
    
    while time.time() < start_time + max_time and time.time() < _DEADLINE:
        improved = False
        rng = random.Random(int(time.time() * 1000) % 100000)
        rng.shuffle(all_edges)
        
        for (u, v) in all_edges:
            if time.time() >= start_time + max_time or time.time() >= _DEADLINE:
                break
            
            # Try adding edge (u,v) if not present
            if not adj[u][v]:
                adj[u][v] = adj[v][u] = True
                if _is_k4_free(adj):
                    score = _score_graph(adj)
                    if score < best_score:
                        best_score = score
                        best_edges = list(edges)
                        best_edges.append([u, v])
                        best_adj = [row[:] for row in adj]
                        improved = True
                    else:
                        adj[u][v] = adj[v][u] = False
                else:
                    adj[u][v] = adj[v][u] = False
            else:
                # Try removing edge (u,v)
                adj[u][v] = adj[v][u] = False
                score = _score_graph(adj)
                if score < best_score:
                    best_score = score
                    best_edges = [e for e in edges if e[0] != u or e[1] != v and e[0] != v or e[1] != u]
                    # More carefully:
                    best_edges = [e for e in best_edges if not (set(e) == {u, v})]
                    best_adj = [row[:] for row in adj]
                    improved = True
                else:
                    adj[u][v] = adj[v][u] = True
        
        if not improved:
            break
    
    return best_edges, best_adj

def construct():
    """Construct a K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
    global _DEADLINE
    start_time = time.time()
    
    best_edges = None
    best_score = _VERTS + 1
    best_adj = None
    
    # Try multiple random greedy constructions
    num_trials = 50
    for trial in range(num_trials):
        if time.time() >= _DEADLINE - 10:
            break
        
        edges, adj = _greedy_construct(random_seed=trial * 12345 + 67890)
        score = _score_graph(adj)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            best_adj = adj
        
        # Local search on promising candidates
        if score <= best_score and time.time() < _DEADLINE - 20:
            edges2, adj2 = _local_search(edges, adj, max_time=15.0)
            score2 = _score_graph(adj2)
            if score2 < best_score:
                best_score = score2
                best_edges = edges2
                best_adj = adj2
        
        if best_score <= 8:
            break
    
    # If we haven't found a good graph, try a structured approach
    if best_score > 9 and time.time() < _DEADLINE - 5:
        # Try a more balanced construction
        # Split into groups and create a specific pattern
        for seed in range(20):
            if time.time() >= _DEADLINE - 3:
                break
            edges, adj = _greedy_construct(random_seed=seed * 999983 + 1)
            score = _score_graph(adj)
            if score < best_score:
                best_score = score
                best_edges = edges
                best_adj = adj
    
    if best_edges is None:
        # Fallback: complete tripartite
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        best_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) 
                      for a in parts[i] for b in parts[j]]
    
    return best_edges
