"""Algebraic voltage lift of the Petersen graph over Z_m with closed-walk filtering."""
import time
import sys
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 138

    # Petersen graph (10 vertices, 15 edges, 3-regular, girth 5)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    # Base edges (u < v) for lift construction
    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
    num_base = len(base_edges)  # 15

    # Precompute all closed walks of length < g in the Petersen graph
    # Each closed walk is represented as a list of (edge_index, direction) pairs
    # where direction is +1 if traversing in base edge direction, -1 otherwise
    closed_walks = _enumerate_closed_walks(petersen_edges, petersen_adj, g)
    num_walks = len(closed_walks)
    print(f"Precomputed {num_walks} closed walks of length < {g}", file=sys.stderr)

    # Moore bound: n >= 2 + k * sum_{i=0}^{(g/2-1)} (k-1)^i = 2 + 3*(1+2+4+8+16) = 2+3*31 = 95
    # Wait, for g=16: sum_{i=0}^{7} 2^i = 255, so n >= 2 + 3*255 = 767
    # Actually the Moore bound for (3,16) is 2 + 3*(1+2+4+8+16+32+64+128) = 2+3*255 = 767
    # But n must be multiple of 10 (Petersen lift), so m >= 77
    # Actually let me recompute: for degree k=3, girth g=16 (even),
    # n >= 2 + k * sum_{i=0}^{g/2 - 1} (k-1)^i = 2 + 3 * sum_{i=0}^{7} 2^i = 2 + 3*255 = 767
    # So m >= ceil(767/10) = 77
    m_start = 77

    # For each m, try multiple voltage assignments and check closed walks
    # Voltage assignment: for edge e with index i, voltage = f(i, m)
    # We'll try:
    #   1. Powers of a base: v_i = base^i mod m
    #   2. Linear: v_i = (a*i + b) mod m
    #   3. Quadratic: v_i = (a*i^2 + b*i + c) mod m
    #   4. Random

    # Precompute edge index for each (u,v) pair
    edge_index = {}
    for idx, (u, v) in enumerate(petersen_edges):
        edge_index[(u, v)] = (idx, 1)   # direction +1
        edge_index[(v, u)] = (idx, -1)  # direction -1

    # For a closed walk (list of (edge_idx, dir)), the total voltage is:
    #   sum over steps of dir * voltage[edge_idx] mod m
    # Precompute for each closed walk the coefficient of each edge voltage
    # Actually, we can just compute the sum directly.

    best_n = None
    best_edges = None

    for m in range(m_start, 2000):
        if time.time() > deadline:
            break

        found = False

        # Try several voltage assignment schemes
        # Scheme 1: Powers of small bases
        for base in range(2, min(m, 50) + 1):
            if time.time() > deadline:
                break
            if m % base == 0:
                continue
            # Check if base is a unit mod m
            if _gcd(base, m) != 1:
                continue
            voltages = [pow(base, i, m) for i in range(num_base)]
            if _check_closed_walks(closed_walks, voltages, m):
                # Found valid assignment!
                n, edges = _build_lift(base_edges, m, voltages)
                # Verify with BFS to be absolutely sure
                if _verify_girth_fast(n, edges, g):
                    return n, edges
                # If BFS fails (shouldn't happen), continue
                found = True
                break

        if found:
            continue

        # Scheme 2: Linear voltages v_i = (a*i + b) mod m
        for a in range(1, min(m, 20) + 1):
            if time.time() > deadline:
                break
            for b in range(0, min(m, 10)):
                if _gcd(a, m) != 1:
                    continue
                voltages = [(a * i + b) % m for i in range(num_base)]
                if _check_closed_walks(closed_walks, voltages, m):
                    n, edges = _build_lift(base_edges, m, voltages)
                    if _verify_girth_fast(n, edges, g):
                        return n, edges
                    break

        # Scheme 3: Random voltages (a few tries)
        num_random = 20 if m < 200 else 10
        for _ in range(num_random):
            if time.time() > deadline:
                break
            voltages = [random.randint(1, m - 1) for _ in range(num_base)]
            if _check_closed_walks(closed_walks, voltages, m):
                n, edges = _build_lift(base_edges, m, voltages)
                if _verify_girth_fast(n, edges, g):
                    return n, edges

        # Track best so far (unverified but structurally valid)
        if best_n is None or 10 * m < best_n:
            voltages = [random.randint(1, m - 1) for _ in range(num_base)]
            n, edges = _build_lift(base_edges, m, voltages)
            best_n = n
            best_edges = edges

    # Fallback: use a large m with random voltages
    m = 5000
    voltages = [random.randint(1, m - 1) for _ in range(num_base)]
    n, edges = _build_lift(base_edges, m, voltages)
    return n, edges


def _enumerate_closed_walks(edges, adj, g):
    """
    Enumerate all closed walks of length < g in the base graph.
    Returns list of lists, where each inner list is a sequence of (edge_idx, direction).
    We use DFS from each vertex, tracking the path.
    """
    # Map (u,v) to (edge_idx, direction)
    edge_map = {}
    for idx, (u, v) in enumerate(edges):
        edge_map[(u, v)] = (idx, 1)
        edge_map[(v, u)] = (idx, -1)

    walks = []
    seen = set()

    for start in range(len(adj)):
        # DFS with path tracking
        stack = [(start, start, [])]  # (current, parent, path)
        while stack:
            u, parent, path = stack.pop()
            for v in adj[u]:
                if v == parent and len(path) > 0:
                    continue  # avoid backtracking
                new_path = path + [edge_map[(u, v)]]
                if v == start and len(new_path) >= 3 and len(new_path) < g:
                    # Found a closed walk
                    # Normalize: rotate so the smallest edge index is first
                    # Actually, just add it as-is; we'll deduplicate later
                    key = tuple(new_path)
                    if key not in seen:
                        seen.add(key)
                        walks.append(new_path)
                if len(new_path) < g - 1:
                    stack.append((v, u, new_path))

    return walks


def _check_closed_walks(closed_walks, voltages, m):
    """
    Check that no closed walk has total voltage 0 mod m.
    voltages: list of length num_base
    Returns True if all closed walks have non-zero total voltage.
    """
    for walk in closed_walks:
        total = 0
        for edge_idx, direction in walk:
            total += direction * voltages[edge_idx]
        if total % m == 0:
            return False
    return True


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _build_lift(base_edges, m, voltages):
    """
    Build the lifted graph from the Petersen base.
    base_edges: list of (u, v) for u < v, length 15
    m: lift size
    voltages: list of length 15
    """
    n = 10 * m
    edges = []
    for (u, v), vol in zip(base_edges, voltages):
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + vol) % m
            edges.append((u_id, v_id))
    return n, edges


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using BFS from each vertex with early termination.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    half_g = g // 2
    dist = [-1] * n
    parent = [-1] * n
    seen_mark = [0] * n
    mark = 0

    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])

        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= half_g:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True

# EVOLVE-BLOCK-END
