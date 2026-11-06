"""Find largest graph with no regular induced subgraph on 6+ vertices."""
import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # --- Fast validator using bitmask + NumPy ---
    def check_valid_fast(adj_bool, n):
        """
        adj_bool: n x n boolean numpy array (symmetric, zero diagonal).
        Returns True if no induced subgraph on >=6 vertices is regular.
        Strategy: For each subset S (as bitmask), compute degrees of vertices in S
        within the induced subgraph. If all equal, S is regular -> invalid.
        We iterate over all subsets of size 6..n.
        """
        # Precompute for each vertex v, the bitmask of its neighbors
        # and the adjacency as integer bitmasks for fast degree counting.
        # adj_mask[v] = bitmask of neighbors of v
        adj_mask = np.zeros(n, dtype=np.uint64)
        for v in range(n):
            m = 0
            for u in range(n):
                if adj_bool[v, u]:
                    m |= (1 << u)
            adj_mask[v] = m

        # For a subset S (bitmask), degree of v in S = popcount(adj_mask[v] & S)
        # We need: not all degrees equal for |S| >= 6.

        # Enumerate all subsets of size >= 6.
        # For n up to 21, total subsets = 2^21 ~ 2M. We skip sizes < 6.
        # Use itertools.combinations for clarity; for n<=21 this is fine.
        vertices = list(range(n))
        for k in range(6, n + 1):
            for subset in itertools.combinations(vertices, k):
                # Build bitmask
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                # Compute degrees
                degs = []
                for v in subset:
                    d = bin(adj_mask[v] & mask).count('1')
                    degs.append(d)
                if len(set(degs)) == 1:
                    return False
        return True

    def adj_to_edges(adj_bool, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_bool[i, j]:
                    edges.append([i, j])
        return edges

    # --- Deterministic algebraic constructions ---
    def polarity_graph(n, p=None):
        """
        Construct a graph from a polarity of a projective plane or similar.
        For n = q^2 + q + 1 (projective plane order q), use the standard
        polarity graph. For other n, fall back to a Paley-like construction.
        """
        adj = np.zeros((n, n), dtype=bool)
        if p is None:
            # Try to find q such that q^2+q+1 <= n
            q = 1
            while (q + 1) ** 2 + (q + 1) + 1 <= n:
                q += 1
            if q * q + q + 1 == n:
                # Use projective plane of order q (requires q prime power)
                # Simple construction: vertices are points of PG(2,q)
                # Edge if point lies on polar line of other point.
                # For simplicity, use a known small case or fallback.
                pass
        # Fallback: Paley-like graph using quadratic residues mod n (if n prime)
        # For general n, use a deterministic pseudo-random edge set.
        rng = random.Random(12345)
        for i in range(n):
            for j in range(i + 1, n):
                # Deterministic but non-trivial edge rule
                # Use a hash-like function to avoid regularity
                val = (i * 7 + j * 13 + i * j * 3) % 10
                if val < 5:
                    adj[i, j] = adj[j, i] = True
        return adj

    def paley_graph(n):
        """
        Paley graph: vertices 0..n-1, edge (i,j) iff (j-i) mod n is a quadratic residue.
        Only works well when n is a prime ≡ 1 (mod 4).
        """
        adj = np.zeros((n, n), dtype=bool)
        # Check if n is prime
        def is_prime(m):
            if m < 2:
                return False
            for d in range(2, int(m ** 0.5) + 1):
                if m % d == 0:
                    return False
            return True

        if not is_prime(n) or n % 4 != 1:
            # Fallback: use a different deterministic construction
            rng = random.Random(n * 7919)
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.45:
                        adj[i, j] = adj[j, i] = True
            return adj

        # Quadratic residues mod n
        residues = set()
        for x in range(1, n):
            residues.add((x * x) % n)
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d in residues:
                    adj[i, j] = adj[j, i] = True
        return adj

    def incidence_graph(n, q):
        """
        Incidence graph of projective plane of order q: n = q^2 + q + 1.
        Vertices are points and lines. Edge if point lies on line.
        This is bipartite, so induced subgraphs may be regular.
        """
        if n != q * q + q + 1:
            return None
        adj = np.zeros((n, n), dtype=bool)
        # Label points 0..q^2, lines q^2+1..q^2+q
        # Simple construction for small q
        # For q=4, n=21
        if q == 4:
            # Known 21-vertex graph from projective plane of order 4
            # Use a standard labeling
            # Points: (x,y) for x,y in GF(4), plus point at infinity
            # This is complex; use a precomputed edge list or fallback
            pass
        return None

    def perturbed_base(base_adj, n, rng, num_flips):
        """Flip num_flips random edges in base_adj."""
        adj = base_adj.copy()
        for _ in range(num_flips):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i != j:
                adj[i, j] = adj[j, i] = not adj[i, j]
        return adj

    # --- Main search ---
    best_n = 1
    best_edges = []

    # Candidate n values to try, in descending order
    for target_n in [21, 20, 19, 18, 17, 16]:
        if elapsed() > DEADLINE - 20:
            break

        found = False
        candidates = []

        # Generate diverse candidate graphs
        # 1. Paley-like
        candidates.append(paley_graph(target_n))
        # 2. Deterministic hash-based
        candidates.append(polarity_graph(target_n))
        # 3. Random graphs with different p values
        for p in [0.4, 0.45, 0.5, 0.55, 0.6]:
            adj = np.zeros((target_n, target_n), dtype=bool)
            rng = random.Random(target_n * 1000 + int(p * 100))
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if rng.random() < p:
                        adj[i, j] = adj[j, i] = True
            candidates.append(adj)

        # 4. Perturbations of the best random graph
        base_adj = np.zeros((target_n, target_n), dtype=bool)
        rng_base = random.Random(42)
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if rng_base.random() < 0.5:
                    base_adj[i, j] = base_adj[j, i] = True
        for num_flips in [1, 2, 3, 5]:
            rng_pert = random.Random(999 + num_flips)
            candidates.append(perturbed_base(base_adj, target_n, rng_pert, num_flips))

        # Check each candidate
        for idx, adj in enumerate(candidates):
            if elapsed() > DEADLINE - 10:
                break
            if check_valid_fast(adj, target_n):
                best_n = target_n
                best_edges = adj_to_edges(adj, target_n)
                found = True
                break

        if found:
            break

        # If not found, try more random restarts for this n
        if elapsed() < DEADLINE - 30:
            for attempt in range(200):
                if elapsed() > DEADLINE - 10:
                    break
                rng = random.Random(attempt * 7919 + target_n)
                adj = np.zeros((target_n, target_n), dtype=bool)
                p = 0.45 + 0.1 * rng.random()
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if rng.random() < p:
                            adj[i, j] = adj[j, i] = True
                if check_valid_fast(adj, target_n):
                    best_n = target_n
                    best_edges = adj_to_edges(adj, target_n)
                    found = True
                    break
            if found:
                break

    # Fallback: ensure at least n=1
    if best_n < 1:
        best_n = 1
        best_edges = []

    return {'n': best_n, 'edges': best_edges}
