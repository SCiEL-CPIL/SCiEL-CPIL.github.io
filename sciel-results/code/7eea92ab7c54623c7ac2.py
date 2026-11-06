import time
import random
import sys
from itertools import combinations

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency matrix as list of sets or bitmasks
    # Using bitmasks for fast operations
    adj = [0] * n
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))

    def get_cliques_4():
        """Find all K4 and Independent K4 (i.e., K4 in complement)"""
        k4s = []
        ik4s = []
        for quad in combinations(range(n), 4):
            is_k4 = True
            is_ik4 = True
            for u, v in combinations(quad, 2):
                if adj[u] & (1 << v):
                    is_ik4 = False
                else:
                    is_k4 = False
            if is_k4:
                k4s.append(quad)
            if is_ik4:
                ik4s.append(quad)
        return k4s, ik4s

    # Initial constraints
    k4s, ik4s = get_cliques_4()

    best_n = n
    best_edges = edges[:]

    def try_add_vertex(current_n, k4s, ik4s):
        """
        Attempts to find an assignment x[0..current_n-1] for a new vertex
        such that no K5 and no IK5 is formed.
        Uses simulated annealing on violation energy.
        Returns assignment x (list of 0/1) or None.
        """
        # Precompute for each variable which constraints it appears in
        # k4_cons: list of (quad, max_sum=3)
        # ik4_cons: list of (quad, min_sum=1)

        # For speed, store as lists of indices
        k4_list = [list(c) for c in k4s]
        ik4_list = [list(c) for c in ik4s]

        n_vars = current_n
        n_k4 = len(k4_list)
        n_ik4 = len(ik4_list)

        # Build variable-to-constraint index for faster updates
        # For each variable, which k4 constraints does it appear in?
        var_k4 = [[] for _ in range(n_vars)]
        for ci, quad in enumerate(k4_list):
            for u in quad:
                var_k4[u].append(ci)

        var_ik4 = [[] for _ in range(n_vars)]
        for ci, quad in enumerate(ik4_list):
            for u in quad:
                var_ik4[u].append(ci)

        # Initialize x randomly, biased toward the "neutral" choice
        # For a random vertex in a Ramsey-free graph, degree is roughly n/2
        x = [1 if random.random() < 0.5 else 0 for _ in range(n_vars)]

        # Compute initial sums
        sums_k4 = [0] * n_k4
        for ci, quad in enumerate(k4_list):
            s = 0
            for u in quad:
                s += x[u]
            sums_k4[ci] = s

        sums_ik4 = [0] * n_ik4
        for ci, quad in enumerate(ik4_list):
            s = 0
            for u in quad:
                s += x[u]
            sums_ik4[ci] = s

        def compute_energy():
            """Total violation energy: for k4, max(0, sum-3); for ik4, max(0, 1-sum)"""
            e = 0
            for s in sums_k4:
                if s > 3:
                    e += (s - 3)
            for s in sums_ik4:
                if s < 1:
                    e += (1 - s)
            return e

        energy = compute_energy()

        # Simulated annealing
        max_iter = 3000
        T = 1.0
        T_min = 0.01
        cooling = 0.995

        for it in range(max_iter):
            if energy == 0:
                return x

            # Pick a random variable to flip
            u = random.randint(0, n_vars - 1)
            old_val = x[u]
            new_val = 1 - old_val

            # Compute delta energy from flipping x[u]
            delta = 0

            # Update k4 constraints containing u
            for ci in var_k4[u]:
                old_s = sums_k4[ci]
                new_s = old_s + (1 if new_val == 1 else -1)
                # Old contribution: max(0, old_s - 3)
                old_c = max(0, old_s - 3)
                new_c = max(0, new_s - 3)
                delta += (new_c - old_c)
                sums_k4[ci] = new_s

            # Update ik4 constraints containing u
            for ci in var_ik4[u]:
                old_s = sums_ik4[ci]
                new_s = old_s + (1 if new_val == 1 else -1)
                # Old contribution: max(0, 1 - old_s)
                old_c = max(0, 1 - old_s)
                new_c = max(0, 1 - new_s)
                delta += (new_c - old_c)
                sums_ik4[ci] = new_s

            new_energy = energy + delta

            # Accept if better or with probability exp(-delta/T)
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                x[u] = new_val
                energy = new_energy
            else:
                # Revert sums
                for ci in var_k4[u]:
                    sums_k4[ci] -= (1 if new_val == 1 else -1)
                for ci in var_ik4[u]:
                    sums_ik4[ci] -= (1 if new_val == 1 else -1)

            T = max(T_min, T * cooling)

        if energy == 0:
            return x
        return None

    # Main Loop: Try to add vertices
    attempts = 0
    consecutive_failures = 0
    while time.time() < deadline:
        attempts += 1
        # Try to add a vertex
        x = try_add_vertex(n, k4s, ik4s)
        if x is not None:
            consecutive_failures = 0
            # Add the vertex
            new_v = n
            adj.append(0)
            for u in range(n):
                if x[u]:
                    adj[new_v] |= (1 << u)
                    adj[u] |= (1 << new_v)
                    edges.append((min(u, new_v), max(u, new_v)))

            n += 1
            if n > best_n:
                best_n = n
                best_edges = edges[:]

            # Update k4s and ik4s
            k4s, ik4s = get_cliques_4()

            # If we reach a high number, we might want to stop or continue
            if n >= 42:
                break
        else:
            consecutive_failures += 1
            # Try a different random seed by resetting random state
            random.seed(attempts * 7919)
            if consecutive_failures > 10:
                # Try perturbing the graph slightly to escape local optimum
                if edges and random.random() < 0.5:
                    # Remove a random edge and try again
                    idx = random.randint(0, len(edges) - 1)
                    u, v = edges[idx]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edges.pop(idx)
                    # Recompute k4s and ik4s
                    k4s, ik4s = get_cliques_4()
                    consecutive_failures = 0
                else:
                    # Add a random non-edge
                    added = False
                    for _ in range(50):
                        u = random.randint(0, n - 1)
                        v = random.randint(0, n - 1)
                        if u < v and not (adj[u] & (1 << v)):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges.append((u, v))
                            added = True
                            break
                    if added:
                        k4s, ik4s = get_cliques_4()
                        consecutive_failures = 0
                    else:
                        break

    return best_n, best_edges

# EVOLVE-BLOCK-END
