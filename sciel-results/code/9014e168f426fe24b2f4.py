import time
import random
import numpy as np

DEADLINE = 145.0  # safety margin before 150s kill

def _count_c4s(adj):
    """Count 4-cycles using common-neighbor matrix."""
    n = adj.shape
    # M[u,v] = number of common neighbors of u and v (for u != v)
    # A 4-cycle u-v-w-x-u is counted by pair (u,w) and pair (v,x)
    # Total C4 = (1/2) * sum over non-edges (u,v) of common_neighbors(u,v)
    # Actually: each C4 has 2 pairs of opposite vertices, both non-adjacent
    # So C4 = sum over non-edges (u,v) of common_neighbors(u,v) / 2
    # Wait: for a C4 u-v-w-x-u, the opposite pairs are (u,w) and (v,x).
    # common_neighbors(u,w) >= 1 (includes v and x), common_neighbors(v,x) >= 1 (includes u and w)
    # So each C4 contributes 2 to the sum (1 for each opposite pair).
    # But common_neighbors can be > 2, meaning multiple C4s share the same opposite pair.
    # So C4 = (1/2) * sum_{non-edge (u,v)} common_neighbors(u,v)
    
    # Compute common neighbors: adj @ adj gives for each (i,j) the number of common neighbors
    common = adj @ adj
    n = adj.shape[0]
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i, j]:
                total += common[i, j]
    return total // 2

def _build_base_graph():
    """Build a C4-sparse base graph on 31 vertices using finite field structure."""
    n = 31
    adj = np.zeros((n, n), dtype=np.int32)
    
    # Use GF(25) structure: represent elements as pairs (a,b) in GF(5)^2
    # Multiplication: (a,b)*(c,d) = (ac - bd, ad + bc) since i^2 = -1 = 4 in GF(5)
    # Actually in GF(25) = GF(5)[x]/(x^2+1), i^2 = -1 = 4
    
    def gf25_mul(a, b, c, d):
        return ((a*c - b*d) % 5, (a*d + b*c) % 5)
    
    def gf25_eq(a, b, c, d):
        return a == c and b == d
    
    # The 26 points of PG(1,25) can be represented as:
    # 25 affine points: (a, b) for a,b in GF(5)
    # 1 point at infinity
    # Plus 5 more vertices (31-26=5)
    
    # Polarity graph of PG(1,25): 
    # Points are 1-dim subspaces of GF(25)^2
    # Two points are adjacent if they are perpendicular under a polarity
    # For the standard polarity: (x,y) perp (x',y') iff x*x' + y*y' = 0 in GF(25)
    
    # Let's enumerate the 26 projective points
    # Each point is a 1-dim subspace, represented by a non-zero vector up to scalar
    # We can represent each point by its "canonical" form
    
    # Actually, let's use a simpler approach:
    # The polarity graph of PG(1,q) for odd q has q+1 vertices and is C4-free
    # For q=25: 26 vertices
    
    # Let me build it more carefully
    # Points of PG(1,25): 26 points
    # Label them 0..25
    # Point i for i in 0..24: represented by (1, alpha^i) where alpha is a primitive element
    # Point 25: point at infinity (1, 0)
    
    # Primitive element in GF(25): 1+i = (1,1)
    # (1,1)^2 = (1-1, 1+1) = (0, 2)
    # Let's just use a direct construction
    
    # Simpler: use the known fact that the polarity graph of PG(1,25) is a (26, 6)-regular graph
    # that is C4-free. Let me construct it explicitly.
    
    # GF(25) elements: 25 of them, indexed 0..24
    # 0 = (0,0)
    # For i=1..24: element i
    
    # Let me use a different approach: construct a C4-free graph on 31 vertices
    # by taking a known C4-free graph and adding vertices/edges.
    
    # The (5,5) grid graph is C4-free? No, it has many C4s.
    
    # Let's use a random greedy approach with C4 avoidance for the base,
    # then the local search will optimize.
    
    # Actually, let's just use a random graph with 91 edges and let the local search do the work.
    # But a good base helps convergence.
    
    # Let me try: take a bipartite graph between two sets of ~15 vertices
    # Bipartite graphs can be C4-free if they're incidence graphs of projective planes etc.
    
    # For now, use a structured base: 
    # Vertices 0-24: GF(25) elements
    # Vertices 25-30: extra vertices
    
    # Connect GF(25) vertices in a C4-free pattern:
    # Use the "sum graph": connect a to b if a+b is a quadratic residue (non-zero)
    # This is a known C4-free construction
    
    # Quadratic residues in GF(25): 12 non-zero squares + 0
    # Let's compute squares
    
    # Build GF(25) table
    gf = []
    gf.append((0, 0))
    for a in range(5):
        for b in range(5):
            if a == 0 and b == 0:
                continue
            gf.append((a, b))
    # gf[0] = 0, gf[1..24] = non-zero elements
    
    # Compute squares
    squares = set()
    for i in range(1, 25):
        a, b = gf[i]
        c, d = gf25_mul(a, b, a, b)
        squares.add((c, d))
    
    # Add 0 to squares for the "sum is square or zero" test
    squares.add((0, 0))
    
    # Build adjacency for vertices 0-24
    for i in range(25):
        for j in range(i+1, 25):
            ai, bi = gf[i]
            aj, bj = gf[j]
            si, sj = gf25_mul(1, 0, 0, 0)  # placeholder
            # sum = gf[i] + gf[j]
            s = ((ai + aj) % 5, (bi + bj) % 5)
            if s in squares:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Now add vertices 25-30 and connect them
    # Try to connect them in a way that minimizes C4s
    # For each new vertex, connect to existing vertices that have low "badness"
    
    for v in range(25, 31):
        # Connect to a few existing vertices
        # Try all possible connections and pick ones that minimize C4
        candidates = list(range(25))
        random.shuffle(candidates)
        count = 0
        for u in candidates:
            if adj[v, u] == 1:
                continue
            # Check if adding edge (v, u) creates C4s
            # A C4 is created if there exists w such that adj[v,w]=1 and adj[u,w]=1
            # and also adj[v,w] and adj[u,w] are both 1 (already checked)
            # Actually we need: exists w with adj[v,w]=1 and adj[u,w]=1 -> that's a C4 v-u-w-? 
            # Wait, C4 needs 4 vertices. Adding edge (v,u) creates C4s if there's a path of length 3 from v to u.
            # i.e., exists w1, w2 with adj[v,w1]=1, adj[w1,w2]=1, adj[w2,u]=1
            # This is expensive to check. Let's just add a few edges and let local search fix it.
            
            # For the base, just add edges somewhat randomly
            if count < 3:  # connect each new vertex to ~3 existing vertices
                adj[v, u] = 1
                adj[u, v] = 1
                count += 1
    
    # Now we need exactly 91 edges
    current_edges = sum(1 for i in range(n) for j in range(i+1, n) if adj[i,j])
    
    # Add or remove edges to get to 91
    # First, let's see how many we have
    if current_edges < 91:
        # Add more edges
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i,j]]
        random.shuffle(pairs)
        for i, j in pairs:
            if current_edges >= 91:
                break
            adj[i, j] = 1
            adj[j, i] = 1
            current_edges += 1
    elif current_edges > 91:
        # Remove edges
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j]]
        random.shuffle(pairs)
        for i, j in pairs:
            if current_edges <= 91:
                break
            adj[i, j] = 0
            adj[j, i] = 0
            current_edges -= 1
    
    return adj

def _local_search(adj, time_limit):
    """Local search to minimize C4 count. Returns best adj found."""
    n = adj.shape[0]
    best_adj = adj.copy()
    best_c4 = _count_c4s(best_adj)
    
    start_time = time.time()
    
    # Precompute common neighbor matrix
    def compute_common(adj):
        return adj @ adj
    
    common = compute_common(best_adj)
    
    # M[u,v] for non-edge (u,v) = common[u,v] = number of common neighbors
    # Total C4 = (1/2) * sum_{non-edge (u,v)} M[u,v]
    
    # For a swap: remove edge (u,v), add edge (a,b)
    # Delta in C4:
    # Removing (u,v): 
    #   - (u,v) becomes a non-edge, contributing M[u,v] to the sum (but was an edge before)
    #   - For each w: common[u,w] changes by -1 if adj[v,w]=1, and common[v,w] changes by -1 if adj[u,w]=1
    #   - This affects M values for non-edges involving u or v
    # Adding (a,b):
    #   - (a,b) becomes an edge, so M[a,b] no longer contributes
    #   - For each w: common[a,w] changes by +1 if adj[b,w]=1, common[b,w] changes by +1 if adj[a,w]=1
    #   - This affects M values for non-edges involving a or b
    
    # This is complex. Let's use a simpler approach: 
    # Just try swaps and recompute C4 from scratch (O(n^3) per swap is too slow)
    # Or use the matrix update approach.
    
    # Let's use a simpler heuristic: try random swaps and accept if better
    # Use numpy for speed
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        # Try a random swap
        # Pick a random edge to remove
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if best_adj[i,j]]
        if not edges:
            break
        u, v = random.choice(edges)
        
        # Pick a random non-edge to add
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not best_adj[i,j]]
        if not non_edges:
            break
        a, b = random.choice(non_edges)
        
        # Make sure (a,b) != (u,v) and the swap is valid (a != u, a != v, b != u, b != v 
        # not strictly required, but let's avoid trivial cases)
        if (a == u and b == v) or (a == v and b == u):
            continue
        if a == u or a == v or b == u or b == v:
            # This is a "move" not a "swap", still valid
            pass
        
        # Try the swap
        trial = best_adj.copy()
        trial[u, v] = 0
        trial[v, u] = 0
        trial[a, b] = 1
        trial[b, a] = 1
        
        # Verify edge count
        edge_count = sum(1 for i in range(n) for j in range(i+1, n) if trial[i,j])
        if edge_count != 91:
            continue
        
        c4 = _count_c4s(trial)
        
        if c4 < best_c4:
            best_adj = trial
            best_c4 = c4
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Restart with a different random seed
            no_improve = 0
            # Perturb the graph
            for _ in range(10):
                edges = [(i, j) for i in range(n) for j in range(i+1, n) if best_adj[i,j]]
                if not edges:
                    break
                u, v = random.choice(edges)
                non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not best_adj[i,j]]
                if not non_edges:
                    break
                a, b = random.choice(non_edges)
                if a == u or a == v or b == u or b == v:
                    continue
                best_adj[u, v] = 0
                best_adj[v, u] = 0
                best_adj[a, b] = 1
                best_adj[b, a] = 1
                best_c4 = _count_c4s(best_adj)
    
    return best_adj, best_c4

def construct():
    """Construct a graph with 31 vertices and 91 edges minimizing 4-cycles."""
    start_time = time.time()
    time_budget = DEADLINE - start_time
    
    # Build base graph
    adj = _build_base_graph()
    
    # Verify 91 edges
    edge_count = sum(1 for i in range(31) for j in range(i+1, 31) if adj[i,j])
    if edge_count != 91:
        # Fix edge count
        pairs = [(i, j) for i in range(31) for j in range(i+1, 31)]
        if edge_count < 91:
            for i, j in pairs:
                if edge_count >= 91:
                    break
                if not adj[i,j]:
                    adj[i,j] = 1
                    adj[j,i] = 1
                    edge_count += 1
        else:
            for i, j in reversed(pairs):
                if edge_count <= 91:
                    break
                if adj[i,j]:
                    adj[i,j] = 0
                    adj[j,i] = 0
                    edge_count -= 1
    
    # Local search
    elapsed = time.time() - start_time
    search_time = max(1.0, time_budget - elapsed - 5.0)  # leave 5s margin
    adj, c4 = _local_search(adj, search_time)
    
    # Convert to list of edges
    edges = []
    for i in range(31):
        for j in range(i+1, 31):
            if adj[i,j]:
                edges.append([i, j])
    
    # Verify exactly 91 edges
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    # Verify distinct
    assert len(set(tuple(sorted(e)) for e in edges)) == 91
    
    return edges
