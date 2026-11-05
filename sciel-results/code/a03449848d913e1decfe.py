import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3, uses a combined approach:
      1. Voltage lift of Petersen graph (algebraic, fast, gives ~10m vertices)
      2. Spectral edge-switching heuristic (targets trace(A^L) minimization)
    Returns (n, edges) where n is number of vertices and edges is a list of (i,j).
    """
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    if k == 3:
        return _construct_k3(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)


def _construct_k3(g, deadline, start_time):
    """
    For k=3: try voltage lift first (fast, deterministic structure),
    then fall back to spectral edge-switching if lift fails or time remains.
    """
    best_n = float('inf')
    best_edges = None

    # Phase 1: Voltage lift of Petersen graph
    lift_result = _construct_petersen_lift(g, deadline, start_time, max_primes=120)
    if lift_result is not None:
        n_lift, edges_lift = lift_result
        best_n = n_lift
        best_edges = edges_lift

    # Phase 2: Spectral edge-switching (if time remains)
    remaining = deadline - time.time()
    if remaining > 8.0:
        # Try to find a smaller graph via edge switching
        # Start from a random 3-regular graph with n vertices
        # n should be even and >= 4
        # Target: n < best_n (or just n if no lift found)
        target_n = best_n if best_n < float('inf') else 100
        # Try a few different sizes below best_n
        spectral_result = _spectral_edge_switching(3, g, target_n, deadline, start_time)
        if spectral_result is not None:
            n_spec, edges_spec = spectral_result
            if n_spec < best_n:
                best_n = n_spec
                best_edges = edges_spec

    if best_edges is None:
        # Final fallback
        return _construct_luw(k=3, g=g, d=11)

    return best_n, best_edges


def _construct_petersen_lift(g, deadline, start_time, max_primes=120):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Returns (n, edges) or None if no valid assignment found within time.
    """
    # Petersen graph edges (10 vertices, 15 edges)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    coeff_vectors = set()
    max_walk_len = g - 1

    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * 15)]

        while stack:
            v, prev_v, l, counts = stack.pop()

            if l >= max_walk_len:
                continue

            for next_v in range(n_base):
                if next_v == prev_v:
                    continue
                if next_v == v:
                    continue

                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue

                new_counts = counts.copy()
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1

                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    coeff_vectors.add(tuple(new_counts))

                stack.append((next_v, v, new_l, new_counts))

    # Normalize constraints
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        for i in range(15):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)

    if not coeff_list:
        voltages = [1] * 15
        m = 2
        n_candidate = 10 * m
        edges = _build_lifted_graph(pet_edges, voltages, m)
        return n_candidate, edges

    # Build coefficient matrix for vectorized checking
    coeff_matrix = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(coeff_matrix)

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = []
    candidate = 2
    while len(primes) < max_primes:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1

    for m in primes:
        if time.time() > deadline:
            break

        n_candidate = 10 * m

        max_attempts = 600 if m < 100 else 300

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            voltages = np.random.randint(1, m, size=15).astype(np.int64)

            for repair_step in range(60):
                if time.time() > deadline:
                    break

                # Vectorized constraint checking
                products = coeff_matrix @ voltages
                residues = np.mod(products, m)

                violated_indices = np.where(residues == 0)[0]

                if len(violated_indices) == 0:
                    voltages_list = voltages.tolist()
                    edges = _build_lifted_graph(pet_edges, voltages_list, m)
                    return n_candidate, edges

                violated_idx = violated_indices[random.randint(0, len(violated_indices) - 1)]
                violated_c = coeff_matrix[violated_idx]

                nonzero_indices = np.where(violated_c != 0)[0]
                if len(nonzero_indices) == 0:
                    break
                j = nonzero_indices[random.randint(0, len(nonzero_indices) - 1)]

                voltages[j] = random.randint(1, m - 1)

    return None


def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    edges = set()
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue

            vol = voltages[e_idx]
            u_edge, v_edge = pet_edges[e_idx]
            if u == u_edge and v == v_edge:
                sign = 1
            else:
                sign = -1

            for layer in range(m):
                u_node = layer * n_base + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * n_base + v

                a, b = min(u_node, v_node), max(u_node, v_node)
                edges.add((a, b))

    return list(edges)


def _spectral_edge_switching(k, g, target_n, deadline, start_time):
    """
    Spectral edge-switching heuristic:
    Start with a random k-regular graph on n vertices.
    Repeatedly perform 2-switches (edge swaps) that reduce trace(A^L) for L < g.
    This targets girth minimization directly.
    """
    if target_n < 4 or target_n % 2 != 0:
        target_n = max(4, target_n - 1 if target_n % 2 else target_n)

    n = target_n
    if n < 4 * k:
        n = 4 * k

    # Build initial random k-regular graph
    try:
        edges = _random_k_regular(n, k)
    except:
        return None

    if edges is None or len(edges) != n * k // 2:
        return None

    # Build adjacency structure
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    # Precompute A^L traces for L = 3, 4, ..., g-1
    # Use numpy for matrix operations
    A = np.zeros((n, n), dtype=np.int64)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1

    def compute_trace_power(A, L):
        """Compute trace of A^L using repeated squaring or direct multiplication."""
        if L == 1:
            return np.trace(A)
        result = A.copy()
        for _ in range(L - 1):
            result = result @ A
            # Keep as int64, mod to prevent overflow for large L
            # For g=16, L up to 15, values can be large but n is moderate
        return np.trace(result)

    # Compute initial trace values
    max_L = g - 1
    traces = {}
    for L in range(3, max_L + 1):
        traces[L] = compute_trace_power(A, L)

    best_traces = dict(traces)
    best_edges = list(edges)

    # Edge switching: find pairs of edges (u,v) and (x,y) where:
    # - u,v and x,y are not adjacent (no shared endpoints)
    # - Removing (u,v) and (x,y), adding (u,x) and (v,y) or (u,y) and (v,x)
    #   keeps the graph k-regular and simple
    # Accept if it reduces the sum of traces (or lexicographically minimizes)

    def trace_score(traces):
        """Weighted sum of traces, weighted by L (shorter cycles matter more)."""
        return sum(L * traces[L] for L in traces)

    current_score = trace_score(traces)
    best_score = current_score

    # Build edge list for fast lookup
    edge_list = list(edge_set)

    # For efficiency, precompute adjacency list as lists
    adj_list = [list(s) for s in adj]

    # Edge switching loop
    max_switches = 50000
    switch_count = 0

    while switch_count < max_switches:
        if time.time() > deadline - 5.0:
            break

        switch_count += 1

        # Pick a random edge
        if not edge_list:
            break
        idx = random.randint(0, len(edge_list) - 1)
        u, v = edge_list[idx]

        # Pick a random non-adjacent vertex x
        # x cannot be u, v, or any neighbor of u or v
        forbidden = set(adj_list[u]) | set(adj_list[v])
        forbidden.add(u)
        forbidden.add(v)

        # Sample x
        x = random.randint(0, n - 1)
        if x in forbidden:
            # Try a few times
            for _ in range(10):
                x = random.randint(0, n - 1)
                if x not in forbidden:
                    break
            else:
                continue

        # Pick a random neighbor y of x (y != x's neighbors that would cause issues)
        # y must be a neighbor of x, y != u, y != v
        # Also, (u,x) and (v,y) or (u,y) and (v,x) must not already be edges
        neighbors_x = adj_list[x]
        if not neighbors_x:
            continue

        y = random.choice(neighbors_x)
        if y == u or y == v:
            continue

        # Check if the swap is valid
        # Option 1: remove (u,v) and (x,y), add (u,x) and (v,y)
        # Need: (u,x) not in edge_set, (v,y) not in edge_set, u!=x, v!=y
        # Also need: after removal, u still has degree k-1, etc. (automatically satisfied)

        if (min(u, x), max(u, x)) in edge_set or (min(v, y), max(v, y)) in edge_set:
            # Try option 2: remove (u,v) and (x,y), add (u,y) and (v,x)
            if (min(u, y), max(u, y)) in edge_set or (min(v, x), max(v, x)) in edge_set:
                continue
            else:
                # Do swap option 2
                new_edges = [(u, y), (v, x)]
                removed = [(u, v), (x, y)]
        else:
            # Do swap option 1
            new_edges = [(u, x), (v, y)]
            removed = [(u, v), (x, y)]

        # Perform the swap
        # Remove old edges
        for a, b in removed:
            adj[a].remove(b)
            adj[b].remove(a)
            adj_list[a].remove(b)
            adj_list[b].remove(a)
            edge_set.discard((min(a, b), max(a, b)))
            edge_list.remove((min(a, b), max(a, b)))

        # Add new edges
        for a, b in new_edges:
            adj[a].add(b)
            adj[b].add(a)
            adj_list[a].append(b)
            adj_list[b].append(a)
            edge_set.add((min(a, b), max(a, b)))
            edge_list.append((min(a, b), max(a, b)))

        # Update A matrix
        for a, b in removed:
            A[a, b] = 0
            A[b, a] = 0
        for a, b in new_edges:
            A[a, b] = 1
            A[b, a] = 1

        # Recompute traces (expensive, but necessary)
        # Optimization: only recompute if the swap might help
        # For now, recompute all
        new_traces = {}
        for L in range(3, max_L + 1):
            new_traces[L] = compute_trace_power(A, L)

        new_score = trace_score(new_traces)

        # Accept if better (or equal with some probability for exploration)
        if new_score < current_score or (new_score == current_score and random.random() < 0.1):
            current_score = new_score
            traces = new_traces
            if new_score < best_score:
                best_score = new_score
                best_traces = dict(traces)
                best_edges = list(edge_set)
        else:
            # Revert
            for a, b in new_edges:
                adj[a].remove(b)
                adj[b].remove(a)
                adj_list[a].remove(b)
                adj_list[b].remove(a)
                edge_set.discard((min(a, b), max(a, b)))
                if (min(a, b), max(a, b)) in edge_list:
                    edge_list.remove((min(a, b), max(a, b)))
                A[a, b] = 0
                A[b, a] = 0
            for a, b in removed:
                adj[a].add(b)
                adj[b].add(a)
                adj_list[a].append(b)
                adj_list[b].append(a)
                edge_set.add((min(a, b), max(a, b)))
                edge_list.append((min(a, b), max(a, b)))
                A[a, b] = 1
                A[b, a] = 1

    # Check if the best graph found has girth >= g
    # Use non-backtracking BFS
    if _has_girth_at_least(n, best_edges, g):
        return n, best_edges

    return None


def _random_k_regular(n, k):
    """
    Generate a random k-regular graph on n vertices.
    Uses the configuration model with backtracking.
    """
    if n * k < 4 or n * k % 2 != 0:
        return None

    # Configuration model
    # Each vertex has k stubs
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)

    random.shuffle(stubs)

    # Pair up stubs
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # Self-loop, restart
        if (min(u, v), max(u, v)) in edges:
            return None  # Multi-edge, restart
        edges.add((min(u, v), max(u, v)))

    if len(edges) != n * k // 2:
        return None

    return list(edges)


def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    max_len = g - 1

    for start in range(n):
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))

        while queue:
            u, prev, l = queue.popleft()

            if u == start and l > 0:
                return False

            if l >= max_len:
                continue

            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))

    return True


def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
