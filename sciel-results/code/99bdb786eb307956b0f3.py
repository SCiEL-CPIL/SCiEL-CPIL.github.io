import math
import time

def construct():
    start = time.time()
    deadline = start + 80  # safety margin under 90s

    N = 256

    # Sieve for primes
    def sieve(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    primes = sieve(N)
    prime_set = set(primes)

    # Precompute prime factorization for all numbers 1..N
    F = [None] * (N + 1)
    F[1] = frozenset()
    for i in range(2, N + 1):
        x = i
        factors = set()
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.add(d)
                while x % d == 0:
                    x //= d
            else:
                d += 1
        if x > 1:
            factors.add(x)
        F[i] = frozenset(factors)

    # Build conflict graph: edge between i and j if gcd(i,j) > 1
    # We want maximum weight independent set where weight = the number itself
    # Vertices: 1..256, but 1 is always included (it's coprime with everything)
    # So we solve for 2..256 and add 1 at the end.

    # Adjacency: for each vertex v (2..256), which other vertices conflict?
    # Use bitmasks for efficiency
    n_vertices = N - 1  # vertices 2..256, indexed 0..254
    # vertex i (0-indexed) corresponds to number i+2

    # Build adjacency lists
    adj = [[] for _ in range(n_vertices)]
    for i in range(n_vertices):
        vi = i + 2
        for j in range(i + 1, n_vertices):
            vj = j + 2
            if math.gcd(vi, vj) > 1:
                adj[i].append(j)
                adj[j].append(i)

    # Convert to bitmask representation for fast operations
    # adj_mask[i] is a bitmask of vertices adjacent to i
    adj_mask = [0] * n_vertices
    for i in range(n_vertices):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)

    # weights[i] = i + 2
    weights = [i + 2 for i in range(n_vertices)]

    # Maximum Weight Independent Set via Branch and Bound
    # Using greedy coloring for upper bound (Tomita's algorithm)

    # Order vertices by descending weight for better pruning
    # Actually, let's use a different ordering strategy
    # Sort by degree (ascending) for better coloring bounds

    # We'll implement the standard branch-and-bound with coloring

    # Color the graph greedily to get an upper bound on the independent set weight
    # For maximum weight independent set, we use a weighted coloring approach

    # Better approach: use the standard MWIS branch and bound
    # with a greedy coloring upper bound on the remaining subgraph

    def max_weight_independent_set(vertices, weights, adj_mask, n):
        """
        vertices: list of vertex indices to consider
        weights: list of weights
        adj_mask: adjacency masks
        n: total number of vertices
        Returns: (best_weight, best_vertex_set_as_bitmask)
        """
        if not vertices:
            return 0, 0

        # For efficiency, work with the set of available vertices as a bitmask
        # and use recursive branching

        best_weight = 0
        best_set = 0

        # We'll use a priority-based branch and bound
        # State: (available_vertices_bitmask, current_weight, current_set_bitmask)

        # Upper bound: sum of weights of all available vertices (crude)
        # Better: greedy coloring bound

        # Greedy coloring upper bound for weighted independent set
        # We color the subgraph induced by available vertices
        # The bound is the sum over colors of the max weight in each color class
        # Actually for MWIS, a common bound is: sort vertices by weight desc,
        # greedily assign colors, bound = sum of max weight per color

        def coloring_bound(available_mask):
            """Compute a coloring-based upper bound for MWIS on the subgraph."""
            # Extract vertices from mask
            verts = []
            m = available_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                verts.append(idx)
                m ^= lsb

            if not verts:
                return 0

            # Greedy coloring: process vertices in order of descending weight
            verts_sorted = sorted(verts, key=lambda v: -weights[v])

            # Assign colors
            color_of = {}
            color_classes = []  # list of (color, max_weight_in_color)

            for v in verts_sorted:
                # Find smallest color not used by neighbors
                used_colors = set()
                for u in verts_sorted:
                    if u == v:
                        break
                    if u in color_of and (adj_mask[v] & (1 << u)):
                        used_colors.add(color_of[u])

                # Actually this is O(n^2) per vertex, too slow
                # Let's use a simpler bound: just sum of top k weights where k is independence number bound
                # Or use a simpler greedy bound

            # Simpler bound: sum of all weights (loose but fast)
            # Or: use the fact that in any independent set, no two adjacent vertices are both included
            # Upper bound = sum of weights / (1 + max_degree/n) ... too complex

            # Let's use a practical bound: sum of weights of available vertices
            # This is loose but the branching will prune
            total = 0
            m = available_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                total += weights[idx]
                m ^= lsb
            return total

        # Better: implement proper branch and bound with recursive search
        # Use a stack-based approach to avoid deep recursion

        # State: (available_mask, current_weight, current_set_mask)
        # We branch on the vertex with highest weight in available

        # For better performance, let's use a different strategy:
        # Since N=255 is manageable, use a recursive B&B with good pruning

        import sys
        sys.setrecursionlimit(10000)

        # Precompute for each vertex, its neighbors mask
        # adj_mask already has this

        # Recursive branch and bound
        def branch_and_bound(available_mask, current_weight, current_set_mask):
            nonlocal best_weight, best_set

            if time.time() > deadline - 5:
                return

            if available_mask == 0:
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_set = current_set_mask
                return

            # Upper bound: current_weight + sum of weights of available vertices
            # This is loose. Let's try a tighter bound.

            # Simple bound: sum of weights of available
            avail_weight = 0
            m = available_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                avail_weight += weights[idx]
                m ^= lsb

            if current_weight + avail_weight <= best_weight:
                return

            # Pick a vertex to branch on: highest weight in available
            # Find highest weight vertex in available_mask
            best_v = -1
            best_w = -1
            m = available_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_v = idx
                m ^= lsb

            if best_v == -1:
                return

            # Branch 1: include best_v
            new_available = available_mask & ~(1 << best_v)
            # Remove neighbors of best_v
            new_available &= ~adj_mask[best_v]
            new_weight = current_weight + weights[best_v]
            new_set = current_set_mask | (1 << best_v)
            branch_and_bound(new_available, new_weight, new_set)

            # Branch 2: exclude best_v
            new_available2 = available_mask & ~(1 << best_v)
            branch_and_bound(new_available2, current_weight, current_set_mask)

        # Initial call
        full_mask = (1 << n_vertices) - 1
        branch_and_bound(full_mask, 0, 0)

        return best_weight, best_set

    # The graph has 255 vertices, which might be too large for simple B&B
    # Let's try a different approach: exploit the structure

    # Key insight: two numbers are coprime iff they share no prime factors
    # So the problem is: select a subset of {2..256} maximizing sum,
    # such that no prime divides two selected numbers
    # Plus we can always include 1

    # This is equivalent to: for each prime p, we can either:
    # - Use p's "slot" for a number divisible by p (and not by other used primes)
    # - Or not use p's slot
    #
    # Actually, it's a set packing problem:
    # Each number c has a set of prime factors P(c)
    # We want to select numbers such that P(c1) ∩ P(c2) = ∅ for all pairs
    # Maximizing sum of selected numbers
    # Plus 1 is always included

    # This is a weighted set packing / maximum weight independent set in a hypergraph

    # Given the structure, let's use the following approach:
    # 1. Start with the "natural" solution: 1 + all prime powers (largest p^k ≤ 256 for each prime p)
    # 2. Try to improve by replacing combinations of prime powers with composites

    # Actually, let's just do a smarter search.

    # Alternative: since the problem is well-structured, let's enumerate possibilities
    # for small prime groups.

    # Let me try a different B&B: branch on primes, not numbers
    # For each prime, decide which number (if any) "uses" it
    # But numbers can use multiple primes...

    # Let's go back to the graph B&B but with better pruning.
    # The key issue is that 255 vertices is a lot.

    # Let's try: reduce the problem.
    # For each prime p, the best number to use that includes p is the largest number ≤ 256
    # whose prime factors are a subset of {p} ∪ (primes we decide to "share")
    # This is getting complex.

    # Practical approach: use the greedy + local search but make it much better
    # Or: use integer linear programming via scipy

    # Let's try scipy MILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np

        n = n_vertices
        # Variables: x_i in {0,1} for each vertex i (number i+2)
        # Maximize: sum(weights[i] * x_i)
        # Subject to: for each edge (i,j): x_i + x_j <= 1
        # And x_i in {0,1}

        # This is a large MILP but scipy might handle it
        # Objective: maximize sum(w_i * x_i) -> minimize -sum(w_i * x_i)
        c = np.array([-weights], dtype=float)

        # Constraints: for each edge, x_i + x_j <= 1
        # Build constraint matrix
        # This could be large: number of edges
        m_constraints = len(adj[0])  # approximate
        # Actually let's count edges
        edge_count = sum(len(a) for a in adj) // 2

        # For 255 vertices, this might have ~2000-5000 edges
        # MILP with 255 binary vars and ~3000 constraints should be feasible

        # Build A_ub, b_ub
        A_list = []
        b_list = []
        for i in range(n_vertices):
            for j in adj[i]:
                if i < j:  # avoid duplicates
                    row = np.zeros(n)
                    row[i] = 1
                    row[j] = 1
                    A_list.append(row)
                    b_list.append(1.0)

        if A_list:
            A_ub = np.array(A_list)
            b_ub = np.array(b_list)
            constraints = LinearConstraint(A_ub, -np.inf, b_ub)
        else:
            constraints = LinearConstraint(np.zeros((0, n)), -np.inf, np.zeros(0))

        integrality = np.ones(n)
        bounds = Bounds(0, 1)

        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if result.success and result.x is not None:
            selected_mask = 0
            selected_weight = 0
            for i in range(n):
                if result.x[i] > 0.5:
                    selected_mask |= (1 << i)
                    selected_weight += weights[i]

            # Build solution: 1 + selected numbers
            solution = [1]
            m = selected_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                solution.append(idx + 2)
                m ^= lsb

            # Verify
            for i in range(len(solution)):
                for j in range(i + 1, len(solution)):
                    if math.gcd(solution[i], solution[j]) != 1:
                        # Fallback
                        solution = [1] + [p for p in primes]
                        break

            return [int(x) for x in solution]
        else:
            # Fallback
            return [int(x) for x in [1] + list(primes)]

    except Exception:
        pass

    # Fallback: greedy + local search (original approach, improved)
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    base_numbers = [1] + [prime_powers[p] for p in primes]

    # Try to improve by replacing prime power combinations with composites
    # For each composite c with 2-4 prime factors, check if c > sum of prime powers of its factors
    # If so, consider using c instead

    composite_moves = []
    for c in range(2, N + 1):
        if c in prime_set:
            continue
        ps = F[c]
        if len(ps) < 2 or len(ps) > 4:
            continue
        base_cost = sum(prime_powers[p] for p in ps)
        gain = c - base_cost
        if gain > 0:
            composite_moves.append((c, ps, gain))

    # Sort by gain descending
    composite_moves.sort(key=lambda x: -x[2])

    # Greedy selection of non-conflicting moves
    selected_moves = []
    used_primes = set()
    for c, ps, gain in composite_moves:
        if ps & used_primes:
            continue
        selected_moves.append((c, ps))
        used_primes |= ps

    # Build solution
    solution = [1]
    for c, ps in selected_moves:
        solution.append(c)
    for p in primes:
        if p not in used_primes:
            solution.append(prime_powers[p])

    # Local search: try swaps
    improved = True
    max_iter = 100
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        if time.time() > deadline - 3:
            break

        current_sum = sum(solution)
        used_primes_sol = set()
        for x in solution:
            if x > 1:
                used_primes_sol |= F[x]

        # Try removing one number and adding a better one
        for i in range(len(solution)):
            if solution[i] == 1:
                continue
            rem_primes = F[solution[i]]
            # Try adding each composite move not currently used
            for c, ps in composite_moves:
                if ps & (used_primes_sol - rem_primes):
                    continue
                if c in solution:
                    continue
                new_sum = current_sum - solution[i] + c
                if new_sum > current_sum:
                    solution[i] = c
                    current_sum = new_sum
                    improved = True
                    break

    # Final verification
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if math.gcd(solution[i], solution[j]) != 1:
                solution = [1] + list(primes)
                break

    # Ensure distinct, in range, Python ints
    solution = list(set(x for x in solution if 1 <= x <= 256))
    solution = [int(x) for x in solution]

    return solution
