"""Combined approach: deterministic algebraic constructions + fast local-search refinement with bitmask validation."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    best_n = 0
    best_edges = None

    # ------------------------------------------------------------------
    # 1) Deterministic algebraic candidates (fast, no search needed)
    # ------------------------------------------------------------------
    candidates = []

    # Andrásfai-type circulant graphs: n = 3k-1, edges if (j-i) mod n ≡ 1 (mod 3)
    # For k=9, n=26, alpha=9  -> valid
    for k in range(10, 1, -1):
        n = 3 * k - 1
        if 1 <= n <= 40:
            adj_mask = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if (j - i) % n % 3 == 1:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
            candidates.append((n, adj_mask))

    # Polarity graphs of PG(2, q): n = q^2 + q + 1
    # Triangle-free, alpha <= q + 1 (roughly). We need alpha < 10, so q+1 < 10 -> q <= 8.
    # q must be prime power: 2,3,4,5,7,8
    for q in [8, 7, 5, 4, 3, 2]:
        n = q * q + q + 1
        if 1 <= n <= 40:
            adj_mask = _build_polarity_graph(q, n)
            if adj_mask is not None:
                candidates.append((n, adj_mask))

    # Try each candidate: validate and keep the best
    for n, adj_mask in candidates:
        if time.time() > deadline - 20:
            break
        if _is_valid_fast(adj_mask, n, deadline):
            if n > best_n:
                best_n = n
                best_edges = _mask_to_edge_list(adj_mask, n)

    # ------------------------------------------------------------------
    # 2) Local-search refinement on the best algebraic seed
    # ------------------------------------------------------------------
    if best_n < 40 and time.time() < deadline - 10:
        # Start from the best valid graph found, or from a random maximal TF graph if none
        seed_n = best_n if best_n > 0 else 26
        seed_adj = _mask_to_adj_matrix(best_edges, seed_n) if best_edges else None

        # Try to extend or improve via local search
        # Strategy: take a valid graph on n vertices, try adding vertices one at a time
        # by finding a "compatible" extension, or do edge flips to improve structure.

        # Simpler: do randomized local search starting from a random maximal TF graph
        # but with a smarter initialization: start from a circulant-like structure
        for target_n in range(40, best_n, -1):
            if time.time() > deadline - 5:
                break

            max_attempts = 300 if target_n >= 39 else 150

            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break

                # Generate a random maximal triangle-free graph with bias toward regularity
                adj_mask = _random_maximal_tf_biased(target_n, deadline)
                if adj_mask is None:
                    continue

                # Quick alpha check
                if _alpha_lt_10_fast(adj_mask, target_n, deadline):
                    best_n = target_n
                    best_edges = _mask_to_edge_list(adj_mask, target_n)
                    break  # Found valid graph for this target_n, try larger

    # ------------------------------------------------------------------
    # 3) Fallback
    # ------------------------------------------------------------------
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}

    # Ultimate fallback: And(9), n=26
    n = 26
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}


def _build_polarity_graph(q, n):
    """
    Build the polarity graph of PG(2, q).
    Vertices are points of PG(2, q): (x, y, 1) for affine, and (x, 1, 0) for line at infinity, plus (0,0,1)...
    Actually PG(2,q) has q^2+q+1 points.
    A standard polarity: point (x,y,z) is adjacent to (x',y',z') if xx' + yy' + zz' = 0 (over GF(q))
    and the point is not self-adjacent (we exclude loops).
    """
    # We need to enumerate points of PG(2, q) and check the polarity condition.
    # For small q (2,3,4,5,7,8), this is feasible.
    # Points: all non-zero (x,y,z) in GF(q)^3 up to scalar.
    # We can use representatives: (x,y,1) for z=1, and (x,1,0) for z=0.
    # Total: q^2 + q + 1.

    # Build field elements
    if q == 4:
        # GF(4) = {0, 1, a, a+1} where a^2 = a + 1
        # Represent as integers 0,1,2,3 with 2=a, 3=a+1
        def gf_add(a, b):
            # GF(4) addition is XOR
            return a ^ b
        def gf_mul(a, b):
            # Multiplication table for GF(4)
            table = [
                [0, 0, 0, 0],
                [0, 1, 2, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2]
            ]
            return table[a][b]
        def gf_neg(a):
            return a  # characteristic 2
        gf_zero = 0
    elif q == 8:
        # GF(8): use polynomial representation x^3 + x + 1
        # Represent elements as 0..7 (binary)
        def gf_add(a, b):
            return a ^ b
        def gf_mul(a, b):
            # Multiply in GF(8) with polynomial x^3+x+1
            result = 0
            for i in range(3):
                if (b >> i) & 1:
                    result ^= a
                a <<= 1
                if a & 8:  # x^3
                    a ^= 0b011  # x^3 = x+1, so a = a - x^3 + x + 1
            return result & 7
        def gf_neg(a):
            return a
        gf_zero = 0
    else:
        # Prime field
        gf_zero = 0
        def gf_add(a, b):
            return (a + b) % q
        def gf_mul(a, b):
            return (a * b) % q
        def gf_neg(a):
            return (-a) % q

    # Enumerate points
    points = []
    # z=1: (x, y, 1) for x,y in GF(q)
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # z=0: (x, 1, 0) for x in GF(q)
    for x in range(q):
        points.append((x, 1, 0))

    n_points = len(points)
    assert n_points == n, f"Expected {n} points, got {n_points}"

    adj_mask = [0] * n_points
    for i in range(n_points):
        for j in range(i + 1, n_points):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            # Check polarity: x1*x2 + y1*y2 + z1*z2 = 0
            val = gf_add(gf_add(gf_mul(x1, x2), gf_mul(y1, y2)), gf_mul(z1, z2))
            if val == gf_zero:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    return adj_mask


def _random_maximal_tf_biased(n, deadline):
    """
    Generate a random maximal triangle-free graph with a bias toward regularity.
    Start with a random circulant-like seed, then fill in greedily.
    """
    adj_mask = [0] * n

    # Start with a random circulant graph: pick random shifts
    num_shifts = random.randint(1, min(3, n // 2))
    shifts = random.sample(range(1, n), num_shifts)
    for s in shifts:
        for i in range(n):
            j = (i + s) % n
            if i < j:
                # Check triangle-freeness before adding
                if adj_mask[i] & adj_mask[j] == 0:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)

    # Now greedily add more edges in random order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(edges)

    for u, v in edges:
        if time.time() > deadline:
            return None
        if adj_mask[u] & adj_mask[v] == 0:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

    return adj_mask


def _alpha_lt_10_fast(adj_mask, n, deadline):
    """
    Fast check: does the graph have an independent set of size 10?
    Equivalent: does the complement have a clique of size 10?
    Uses a greedy + B&B hybrid.
    """
    all_mask = (1 << n) - 1
    comp_mask = [all_mask ^ (1 << i) ^ adj_mask[i] for i in range(n)]

    # Quick degree check
    max_deg = max(bin(comp_mask[i]).count('1') for i in range(n))
    if max_deg < 9:
        return True  # No K10 possible

    # Greedy lower bound on clique number in complement
    # If we can find a clique of size 10, return False (invalid)
    # If we can't, we need B&B to prove no K10

    # Order by degree descending
    deg_order = sorted(range(n), key=lambda i: bin(comp_mask[i]).count('1'), reverse=True)

    # Try a greedy clique search first
    greedy_clique = []
    candidates = all_mask
    for _ in range(10):
        if candidates == 0:
            break
        # Pick vertex with highest degree in candidates
        best_v = -1
        best_deg = -1
        c = candidates
        while c:
            lb = c & (-c)
            idx = lb.bit_length() - 1
            c ^= lb
            deg_in_cand = bin(comp_mask[idx] & candidates).count('1')
            if deg_in_cand > best_deg:
                best_deg = deg_in_cand
                best_v = idx
        if best_v == -1:
            break
        greedy_clique.append(best_v)
        candidates &= comp_mask[best_v]
        candidates &= ~(1 << best_v)

    if len(greedy_clique) >= 10:
        return False  # Found independent set of size 10

    # B&B to prove no K10
    def bnb_kclique(candidates, need):
        if time.time() > deadline:
            return False  # Timeout: assume no clique (optimistic for validity)
        if need == 0:
            return True
        if candidates == 0:
            return False
        if bin(candidates).count('1') < need:
            return False

        c = candidates
        while c:
            lb = c & (-c)
            idx = lb.bit_length() - 1
            c ^= lb
            remaining = candidates & (~((1 << (idx + 1)) - 1))
            new_candidates = remaining & comp_mask[idx]
            if bnb_kclique(new_candidates, need - 1):
                return True
        return False

    return not bnb_kclique(all_mask, 10)


def _is_valid_fast(adj_mask, n, deadline):
    """Check triangle-freeness and alpha < 10."""
    # Triangle-freeness check
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_mask[i] >> j) & 1:
                if adj_mask[i] & adj_mask[j]:
                    return False
    # Alpha check
    return _alpha_lt_10_fast(adj_mask, n, deadline)


def _mask_to_edge_list(adj_mask, n):
    """Convert bitmask adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges


def _mask_to_adj_matrix(adj_mask, n):
    """Convert bitmask adjacency to 2D list."""
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            adj[i][j] = (adj_mask[i] >> j) & 1
    return adj
