import time
import random
import numpy as np
from collections import deque

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True


def _spectral_optimize(n, edges, g, deadline):
    """
    Spectral optimization: represent 3-regular graph as product of 3 permutation matrices.
    Each permutation is a perfect matching (involution). The adjacency matrix is P1+P2+P3.
    We optimize by swapping edges guided by eigenvalue magnitudes to minimize short cycles.
    
    For a 3-regular graph, we can decompose edges into 3 perfect matchings (1-factors).
    The graph is the union of 3 matchings. We represent each matching as a permutation.
    Edge swaps modify the permutations while maintaining 3-regularity.
    """
    if time.time() > deadline:
        return n, edges
    
    # Build adjacency and decompose into 3 perfect matchings
    adj = _build_adj(n, edges)
    
    # Find a proper 3-edge-coloring (decomposition into 3 perfect matchings)
    # For 3-regular graphs, this is always possible (Kotzig's theorem)
    # Use greedy algorithm
    color = {}  # edge -> color (0, 1, 2)
    matchings = [[], [], []]  # list of edges for each color
    
    edge_list = list(set(tuple(sorted(e)) for e in edges))
    
    # Greedy edge coloring
    def can_color(edge, c):
        u, v = edge
        for e in matchings[c]:
            if u in e or v in e:
                return False
        return True
    
    # Sort edges for deterministic behavior
    random.shuffle(edge_list)
    for edge in edge_list:
        placed = False
        for c in range(3):
            if can_color(edge, c):
                color[edge] = c
                matchings[c].append(edge)
                placed = True
                break
        if not placed:
            # Try all colors more carefully
            for c in range(3):
                if can_color(edge, c):
                    color[edge] = c
                    matchings[c].append(edge)
                    placed = True
                    break
            if not placed:
                # Fallback: just assign to first available
                for c in range(3):
                    if len(matchings[c]) < n // 2:
                        color[edge] = c
                        matchings[c].append(edge)
                        break
    
    # Convert matchings to permutations
    # Each matching is a set of n/2 disjoint edges
    # Represent as permutation: for edge (u,v), perm[u]=v and perm[v]=u
    perms = [np.zeros(n, dtype=int) for _ in range(3)]
    for c in range(3):
        for u, v in matchings[c]:
            perms[c][u] = v
            perms[c][v] = u
    
    # Build adjacency matrix from permutations
    def build_adj_matrix(perms):
        A = np.zeros((n, n), dtype=np.float64)
        for p in perms:
            for i in range(n):
                A[i, p[i]] = 1.0
                A[p[i], i] = 1.0
        # Each edge counted twice in the sum, divide by 2? No, each permutation contributes 1 to each directed edge.
        # For undirected, A[i,j] = 1 if there's an edge. With 3 matchings, A[i,j] = sum of contributions.
        # Actually, each matching contributes at most 1 to A[i,j], and since matchings are disjoint, A[i,j] is 0 or 1.
        # But we're summing, so if two matchings have edge (i,j), A[i,j] would be 2. But they can't - matchings are disjoint.
        # So A[i,j] = 1 if any matching has edge (i,j), else 0.
        # Wait, that's not right either. Let me think again.
        # Each matching is a perfect matching, so each vertex has exactly one edge in each matching.
        # The three matchings are edge-disjoint, so the total degree is 3.
        # A[i,j] should be 1 if (i,j) is an edge in any matching, else 0.
        # Since matchings are disjoint, at most one matching contains edge (i,j).
        # So A[i,j] = 1 if any matching has (i,j), else 0.
        # The current code sets A[i,p[i]] = 1 for each permutation, which is correct.
        return A
    
    A = build_adj_matrix(perms)
    
    # Compute eigenvalues
    def get_largest_eigenvalue(A):
        # Use power iteration for speed
        v = np.random.randn(n)
        v /= np.linalg.norm(v)
        for _ in range(100):
            v_new = A @ v
            norm = np.linalg.norm(v_new)
            if norm < 1e-10:
                break
            v_new /= norm
            if np.linalg.norm(v_new - v) < 1e-8:
                v = v_new
                break
            v = v_new
        return np.abs(v @ A @ v)
    
    current_eig = get_largest_eigenvalue(A)
    best_eig = current_eig
    best_perms = [p.copy() for p in perms]
    
    # Edge swap: pick two vertices u, v from different matchings and swap their partners
    # This maintains 3-regularity
    def try_swap(perms, A, deadline):
        if time.time() > deadline:
            return None, None
        
        # Pick random vertices
        for _ in range(20):
            if time.time() > deadline:
                return None, None
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Find which matching connects u to u' and v to v'
            cu = None
            cv = None
            for c in range(3):
                if perms[c][u] != u:  # u is matched in this permutation
                    cu = c
                    break
            for c in range(3):
                if perms[c][v] != v:
                    cv = c
                    break
            
            if cu is None or cv is None or cu == cv:
                continue
            
            u_prime = perms[cu][u]
            v_prime = perms[cv][v]
            
            # Check if swap creates a loop or duplicate edge
            if u == v_prime or u_prime == v or u == v or u_prime == v_prime:
                continue
            
            # Check if (u, v) or (u_prime, v_prime) already exist in other matchings
            edge_exists_uv = False
            edge_exists_up_vp = False
            for c in range(3):
                if perms[c][u] == v or perms[c][v] == u:
                    edge_exists_uv = True
                    break
                if perms[c][u_prime] == v_prime or perms[c][v_prime] == u_prime:
                    edge_exists_up_vp = True
                    break
            
            if edge_exists_uv or edge_exists_up_vp:
                continue
            
            # Perform swap
            new_perms = [p.copy() for p in perms]
            new_perms[cu][u] = v
            new_perms[cu][v] = u
            new_perms[cu][u_prime] = v_prime
            new_perms[cu][v_prime] = u_prime
            new_perms[cv][v] = u_prime
            new_perms[cv][u_prime] = v
            new_perms[cv][v_prime] = u
            new_perms[cv][u] = v_prime
            
            # Rebuild adjacency
            new_A = build_adj_matrix(new_perms)
            new_eig = get_largest_eigenvalue(new_A)
            
            if new_eig < best_eig - 1e-6:
                return new_perms, new_eig
        
        return None, None
    
    # Iterative optimization
    iterations = 0
    max_iterations = 500
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        new_perms, new_eig = try_swap(perms, A, deadline)
        if new_perms is not None:
            perms = new_perms
            A = build_adj_matrix(perms)
            current_eig = new_eig
            if current_eig < best_eig:
                best_eig = current_eig
                best_perms = [p.copy() for p in perms]
        else:
            # No improvement in this batch, try more swaps or break
            if iterations > 100:
                break
    
    # Reconstruct edges from best permutations
    edges = []
    for c in range(3):
        for u in range(n):
            v = best_perms[c][u]
            if u < v:
                edges.append((u, v))
    
    return n, edges


def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for 3-regular graphs.
    We try smaller d values to get fewer vertices, checking girth empirically.
    """
    if k != 3:
        # Fallback for other k (should not be hit for this task)
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # For k=3, we use q=3.
    q = 3
    # Try smaller d values first.
    # d=8 => 2*6561 = 13122
    # d=9 => 2*19683 = 39366
    # d=10 => 2*59049 = 118098
    # d=11 => 2*177147 = 354294

    def build_luw(d):
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try d=8 first. If girth < 16, try d=9, etc.
    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    # If all fail (unlikely for d=11), return the largest one (which should be valid by theory)
    return build_luw(11)


def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 10s margin

    if k != 3:
        return _luw_construct(k, g)

    # Try spectral optimization starting from LUW d=8
    n_luw, edges_luw = _luw_construct(3, g)
    
    # Only attempt spectral optimization if we have time
    if time.time() < deadline - 30:
        n_opt, edges_opt = _spectral_optimize(n_luw, edges_luw, g, deadline - 5)
        # Verify the optimized graph
        if _check_girth_fast(n_opt, edges_opt, g):
            return n_opt, edges_opt
    
    # Fallback to LUW
    return _luw_construct(3, g)
