"""Rewritten program: spectral/Cayley construction on Z5 x Z5 + fast local search."""
import time
import random
import numpy as np

N = 31
M = 91

def count_c4(adj):
    """Count C4s from numpy adjacency matrix."""
    common = adj @ adj
    c4 = 0
    for u in range(N):
        for v in range(u + 1, N):
            c = common[u, v]
            if c >= 2:
                c4 += c * (c - 1) // 2
    return c4

def make_adj(edges):
    adj = np.zeros((N, N), dtype=np.int32)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    return adj

def delta_add_fast(adj, u, v):
    """C4s created by adding edge (u,v) assuming it's not currently present.
    Uses vectorized numpy for speed."""
    # Common neighbors of u and v
    common = adj[u] & adj[v]
    # For each pair of common neighbors a, b, check if a-b is an edge
    # C4s = number of edges among common neighbors
    cn = np.where(common)[0]
    if len(cn) < 2:
        return 0
    # Count edges within cn
    sub = adj[np.ix_(cn, cn)]
    return sub.sum() // 2

def local_search(edges, time_budget):
    """Hill-climbing with 1-swap moves and fast delta computation."""
    adj = make_adj(edges)
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))

    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_edges = [list(e) for e in edges]

    start_time = time.time()
    deadline = start_time + time_budget
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    all_pairs_set = set(all_pairs)

    while time.time() < deadline:
        improved = False
        for i in range(len(edges)):
            if time.time() > deadline:
                return best_edges
            u1, v1 = edges[i]
            e1 = (min(u1, v1), max(u1, v1))

            # Cost of removing edge (u1, v1) = C4s that use this edge
            remove_delta = delta_add_fast(adj, u1, v1)

            # Temporarily remove
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            edge_set.discard(e1)

            best_move = None
            best_delta = 0

            for u2, v2 in all_pairs:
                e2 = (min(u2, v2), max(u2, v2))
                if e2 in edge_set:
                    continue
                add_delta = delta_add_fast(adj, u2, v2)
                delta = add_delta - remove_delta
                if delta < best_delta:
                    best_delta = delta
                    best_move = (u2, v2)

            if best_move is not None:
                u2, v2 = best_move
                e2 = (min(u2, v2), max(u2, v2))
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                edge_set.add(e2)
                edges[i] = [u2, v2]
                current_c4 += best_delta
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges = [list(e) for e in edges]
                improved = True
            else:
                # Restore
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                edge_set.add(e1)

        if not improved:
            break

    return best_edges

def cayley_graph_z5x5():
    """Construct a Cayley graph on Z5 x Z5 with exactly 91 edges.
    Use connection set S in Z5 x Z5 with |S| = 91/2... but 91 is odd.
    
    Actually for a Cayley graph on group G of order 31, we need an involution-free
    subset S with |S| such that |G|/2 * |S| = M... that doesn't work for non-abelian.
    
    Let's just build a structured graph:
    Vertices are (i, j) in Z5 x Z5, labeled 5*i + j.
    We'll use a connection set that gives us close to 91 edges.
    
    A simpler approach: use a polarity graph construction.
    """
    # Polarity graph of projective plane of order 5
    # The projective plane PG(2,5) has q^2+q+1 = 31 points.
    # A polarity graph has vertices = points, edge (P,Q) if P is on the polar line of Q.
    # For the standard polarity, this gives a graph with specific C4 properties.
    
    # Standard construction: label points as (x, y, 1) and (1, y, 0) and (0, 1, 0)
    # Actually let's use the standard polarity graph construction.
    
    # Points of PG(2,5): 31 points. We can label them:
    # (x, y, 1) for x,y in Z5 -> 25 points (indices 0-24)
    # (1, y, 0) for y in Z5 -> 5 points (indices 25-29)
    # (0, 1, 0) -> 1 point (index 30)
    
    def point_to_idx(p):
        x, y, z = p
        if z == 1:
            return x * 5 + y
        elif z == 0 and x == 1:
            return 25 + y
        else:  # (0, 1, 0)
            return 30
    
    def idx_to_point(i):
        if i < 25:
            x, y = divmod(i, 5)
            return (x, y, 1)
        elif i < 30:
            return (1, i - 25, 0)
        else:
            return (0, 1, 0)
    
    # Polarity: (x, y, z) is on polar of (x', y', z') iff
    # x*x' + y*y' + z*z' = 0 (symmetric bilinear form)
    # Wait, that's not quite right for a polarity. Let me use:
    # The standard polarity in PG(2,q) for q odd: 
    # (x,y,z) is on polar of (x',y',z') iff xx' + yy' + zz' = 0
    # But this is a symmetric bilinear form, so it's a polarity.
    
    # Actually for a proper polarity graph we need:
    # P is on polar line of Q, where the polarity is given by a symmetric bilinear form.
    # The adjacency: A(P,Q) = 1 iff B(P,Q) = 0 where B is the bilinear form.
    
    # Let's use B((x,y,z),(x',y',z')) = x*x' + y*y' + z*z' mod 5
    # Then the graph has edges where B=0.
    
    edges = []
    edge_set = set()
    
    for i in range(N):
        pi = idx_to_point(i)
        for j in range(i + 1, N):
            pj = idx_to_point(j)
            # Check if B(pi, pj) = 0 mod 5
            b = (pi[0]*pj[0] + pi[1]*pj[1] + pi[2]*pj[2]) % 5
            if b == 0:
                edges.append([i, j])
                edge_set.add((i, j))
    
    # Count how many edges we have
    # For the standard polarity in PG(2,5), each point is on its own polar line
    # (absolutely points) and the number of edges should be around 90.
    
    if len(edges) == M:
        return edges
    elif len(edges) > M:
        # Remove extra edges
        return edges[:M]
    else:
        # Need to add edges
        return edges

def random_near_regular_init(seed):
    """Random graph with degree sequence close to regular, exactly M edges."""
    random.seed(seed)
    # Target degrees: 91*2/31 ≈ 5.87, so mix of 5 and 6
    # 27 vertices degree 6, 4 vertices degree 5: 27*6+4*5 = 162+20=182 = 91*2 ✓
    target = [6]*27 + [5]*4
    random.shuffle(target)
    
    edges = []
    edge_set = set()
    deg = [0]*N
    
    candidates = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(candidates)
    
    for u, v in candidates:
        if len(edges) >= M:
            break
        if deg[u] < target[u] and deg[v] < target[v]:
            edges.append([u, v])
            edge_set.add((u, v))
            deg[u] += 1
            deg[v] += 1
    
    if len(edges) < M:
        for u, v in candidates:
            if len(edges) >= M:
                break
            if (u, v) not in edge_set:
                edges.append([u, v])
                edge_set.add((u, v))
                deg[u] += 1
                deg[v] += 1
    
    return edges

def construct():
    """Main entry: multiple strategies + local search."""
    start_time = time.time()
    deadline = start_time + 115
    
    best_edges = None
    best_c4 = float('inf')
    
    def try_edges(edges, do_ls=True, ls_time=3):
        nonlocal best_edges, best_c4
        if edges is None or len(edges) != M:
            return
        adj = make_adj(edges)
        c4 = count_c4(adj)
        if c4 < best_c4:
            best_c4 = c4
            best_edges = [list(e) for e in edges]
        
        if do_ls and time.time() < deadline - 8:
            ls_edges = local_search([list(e) for e in edges], ls_time)
            adj_ls = make_adj(ls_edges)
            c4_ls = count_c4(adj_ls)
            if c4_ls < best_c4:
                best_c4 = c4_ls
                best_edges = [list(e) for e in ls_edges]
    
    # Strategy 1: Polarity graph
    pol_edges = cayley_graph_z5x5()
    try_edges(pol_edges)
    
    # Strategy 2: Multiple random restarts
    for seed in range(500):
        if time.time() > deadline - 10:
            break
        edges = random_near_regular_init(seed)
        if len(edges) != M:
            continue
        # Quick C4 count to filter
        adj = make_adj(edges)
        c4 = count_c4(adj)
        if c4 < best_c4:
            best_c4 = c4
            best_edges = [list(e) for e in edges]
        
        if time.time() < deadline - 8:
            # Only do local search if promising
            if c4 < best_c4 + 50:
                ls_edges = local_search([list(e) for e in edges], 2.5)
                adj_ls = make_adj(ls_edges)
                c4_ls = count_c4(adj_ls)
                if c4_ls < best_c4:
                    best_c4 = c4_ls
                    best_edges = [list(e) for e in ls_edges]
    
    if best_edges is None or len(best_edges) != M:
        pairs = [[i, j] for i in range(N) for j in range(i+1, N)]
        best_edges = pairs[:M]
    
    return best_edges
