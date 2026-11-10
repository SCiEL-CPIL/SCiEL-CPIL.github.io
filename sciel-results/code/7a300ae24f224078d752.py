"""Finds minimum vertex transversals for maximal cliques in three random graphs."""
import time
import random
from itertools import combinations


def construct():
    """
    For each of three fixed random graphs, find a minimum-size set of vertices
    that intersects every maximal clique of size >= 2.
    Returns a list of three vertex lists.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        """Generate a random graph using the same RNG sequence as the seed program."""
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, n):
        """
        Enumerate all maximal cliques using Bron-Kerbosch with pivot.
        Returns a list of frozensets (each a maximal clique).
        """
        cliques = []

        def bk(r, p, x):
            if not p and not x:
                cliques.append(frozenset(r))
                return

            # Choose pivot with max |p ∩ N(u)|
            px = p | x
            if px:
                pivot = max(px, key=lambda u: len(p & adj[u]))
                candidates = p - adj[pivot]
            else:
                candidates = p

            for v in list(candidates):
                bk(r | {v}, p & adj[v], x & adj[v])
                p = p - {v}
                x = x | {v}

        bk(set(), set(range(n)), set())
        return cliques

    def min_hitting_set(cliques, adj, n, time_limit=30.0):
        """
        Find a minimum-size hitting set for the given collection of cliques.
        Uses branch-and-bound with good heuristics.
        """
        # Filter: only cliques of size >= 2
        cliques = [c for c in cliques if len(c) >= 2]
        if not cliques:
            return []

        # Convert to sorted tuples for hashing
        clique_list = [tuple(sorted(c)) for c in cliques]
        
        # Remove duplicate cliques
        clique_list = list(set(clique_list))
        clique_list.sort(key=len)  # Shorter cliques first (more constraining)

        # Precompute for each vertex, which cliques it hits
        vertex_cliques = {v: set() for v in range(n)}
        for i, c in enumerate(clique_list):
            for v in c:
                vertex_cliques[v].add(i)

        # Greedy initial solution
        def greedy_hitting_set():
            remaining = set(range(len(clique_list)))
            chosen = set()
            while remaining:
                # Pick vertex that hits most remaining cliques
                best_v = None
                best_count = -1
                for v in range(n):
                    count = len(vertex_cliques[v] & remaining)
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_count <= 0:
                    break
                chosen.add(best_v)
                remaining -= vertex_cliques[best_v]
            return chosen

        initial_solution = greedy_hitting_set()
        best_size = len(initial_solution)
        best_set = set(initial_solution)

        # Lower bound: for each clique, we need at least 1 vertex from it.
        # Better: use LP relaxation or simple counting.
        # Simple LB: max over cliques of 1 (trivial). Better: if no two cliques 
        # share a vertex, LB = number of cliques. Generally, use a greedy LB.
        
        def lower_bound(remaining_cliques):
            """Simple lower bound: maximum matching of cliques to disjoint vertex sets."""
            if not remaining_cliques:
                return 0
            # Greedy: sort by size, assign
            rc = sorted(remaining_cliques, key=lambda i: len(clique_list[i]))
            used_vertices = set()
            lb = 0
            for i in rc:
                c = clique_list[i]
                if any(v in used_vertices for v in c):
                    continue
                # Find a vertex in c not yet used
                for v in c:
                    if v not in used_vertices:
                        used_vertices.add(v)
                        lb += 1
                        break
            return lb

        # Branch and bound
        # State: (current_set, remaining_cliques_indices)
        # We branch on which vertex to include for the most constrained remaining clique.

        import sys
        sys.setrecursionlimit(10000)

        def bnb(current_set, remaining, depth):
            nonlocal best_size, best_set
            if time.time() > time_limit:
                return

            if not remaining:
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return

            # Lower bound pruning
            lb = lower_bound(remaining)
            if len(current_set) + lb >= best_size:
                return

            # Pick the most constrained clique (smallest)
            min_idx = min(remaining, key=lambda i: len(clique_list[i]))
            clique = clique_list[min_idx]

            # Try each vertex in the clique
            # Order vertices by how many remaining cliques they cover (descending)
            verts = sorted(clique, key=lambda v: -len(vertex_cliques[v] & remaining))

            for v in verts:
                if time.time() > time_limit:
                    return
                new_set = current_set | {v}
                new_remaining = set()
                for i in remaining:
                    if v not in clique_list[i]:
                        new_remaining.add(i)
                bnb(new_set, new_remaining, depth + 1)

        # Start B&B
        all_indices = set(range(len(clique_list)))
        bnb(set(), all_indices, 0)

        return sorted(best_set)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: return empty lists or greedy
            adj = random_graph(n, p, seed)
            cliques = bron_kerbosch(adj, n)
            cliques = [c for c in cliques if len(c) >= 2]
            # Simple greedy fallback
            chosen = set()
            remaining = set(range(len(cliques)))
            while remaining:
                best_v = None
                best_count = -1
                for v in range(n):
                    count = sum(1 for i in remaining if v in cliques[i])
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_count <= 0:
                    break
                chosen.add(best_v)
                remaining = {i for i in remaining if best_v not in cliques[i]}
            results.append(sorted(chosen))
            continue

        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj, n)
        
        # Allocate remaining time
        remaining_time = deadline - time.time()
        per_instance_time = max(5.0, remaining_time / (3 - len(results)))
        
        solution = min_hitting_set(cliques, adj, n, time_limit=per_instance_time)
        results.append(solution)

    return results
