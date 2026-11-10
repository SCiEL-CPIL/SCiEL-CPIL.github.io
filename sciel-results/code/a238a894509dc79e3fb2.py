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
        Uses bitmask-based branch-and-bound with local search polish.
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
        m = len(clique_list)

        # Precompute for each vertex, which cliques it hits (as bitmask)
        vertex_clique_mask = [0] * n
        for i, c in enumerate(clique_list):
            for v in c:
                vertex_clique_mask[v] |= (1 << i)

        # Greedy initial solution
        def greedy_hitting_set():
            remaining_mask = (1 << m) - 1
            chosen = set()
            while remaining_mask:
                # Pick vertex that hits most remaining cliques
                best_v = -1
                best_count = -1
                for v in range(n):
                    count = bin(vertex_clique_mask[v] & remaining_mask).count('1')
                    if count > best_count:
                        best_count = count
                        best_v = v
                if best_count <= 0:
                    break
                chosen.add(best_v)
                remaining_mask &= ~vertex_clique_mask[best_v]
            return chosen

        initial_solution = greedy_hitting_set()
        best_size = len(initial_solution)
        best_set = set(initial_solution)

        # Better lower bound using LP relaxation approximation
        def lower_bound(remaining_mask):
            """Count-based lower bound: for each vertex, count cliques it can cover."""
            if remaining_mask == 0:
                return 0
            # Simple: count how many cliques remain, divide by max coverage
            num_remaining = bin(remaining_mask).count('1')
            if num_remaining == 0:
                return 0
            # Max number of cliques any single vertex can cover from remaining
            max_cover = 0
            for v in range(n):
                cover = bin(vertex_clique_mask[v] & remaining_mask).count('1')
                if cover > max_cover:
                    max_cover = cover
            if max_cover == 0:
                return num_remaining  # Each clique needs its own vertex
            return (num_remaining + max_cover - 1) // max_cover

        # Local search to improve a given hitting set
        def local_search_improve(hitting_set, time_limit_ls):
            """Try to remove vertices or swap to reduce size."""
            current = set(hitting_set)
            current_size = len(current)
            start_ls = time.time()
            
            # Try removing each vertex
            for v in list(current):
                if time.time() - start_ls > time_limit_ls:
                    break
                # Check if removing v still covers all cliques
                new_mask = 0
                for u in current - {v}:
                    new_mask |= vertex_clique_mask[u]
                if new_mask == ((1 << m) - 1):
                    current.remove(v)
                    current_size -= 1
            
            # Try swapping: remove one, add one
            for v_remove in list(current):
                if time.time() - start_ls > time_limit_ls:
                    break
                for v_add in range(n):
                    if v_add in current:
                        continue
                    new_set = (current - {v_remove}) | {v_add}
                    if len(new_set) < current_size:
                        # Check coverage
                        new_mask = 0
                        for u in new_set:
                            new_mask |= vertex_clique_mask[u]
                        if new_mask == ((1 << m) - 1):
                            current = new_set
                            current_size = len(current)
                            break
            return current

        # Apply local search to initial solution
        best_set = local_search_improve(initial_solution, min(5.0, time_limit * 0.1))
        best_size = len(best_set)

        # Bitmask-based branch and bound
        import sys
        sys.setrecursionlimit(10000)
        full_mask = (1 << m) - 1

        def bnb(current_mask, remaining_mask, current_size, depth):
            nonlocal best_size, best_set
            if time.time() > time_limit:
                return

            if remaining_mask == 0:
                if current_size < best_size:
                    # Reconstruct set from current_mask
                    best_set = set(v for v in range(n) if (current_mask >> v) & 1)
                    best_size = current_size
                return

            # Lower bound pruning
            lb = lower_bound(remaining_mask)
            if current_size + lb >= best_size:
                return

            # Pick the most constrained clique (first set bit in remaining)
            # Find a clique in remaining with fewest available vertices
            min_clique_idx = -1
            min_avail = n + 1
            temp = remaining_mask
            while temp:
                i = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                # Count vertices in this clique that could be added
                c = clique_list[i]
                avail = len(c)
                if avail < min_avail:
                    min_avail = avail
                    min_clique_idx = i

            if min_clique_idx < 0:
                return

            clique = clique_list[min_clique_idx]
            
            # Order vertices by how many remaining cliques they cover (descending)
            verts = sorted(clique, key=lambda v: -bin(vertex_clique_mask[v] & remaining_mask).count('1'))

            for v in verts:
                if time.time() > time_limit:
                    return
                new_mask = current_mask | (1 << v)
                new_remaining = remaining_mask & ~vertex_clique_mask[v]
                bnb(new_mask, new_remaining, current_size + 1, depth + 1)

        # Start B&B
        bnb(0, full_mask, 0, 0)

        # Final local search polish on best found
        final_set = local_search_improve(best_set, min(5.0, max(1.0, time_limit * 0.1)))
        return sorted(final_set)

    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
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
        
        # Allocate remaining time more aggressively to earlier instances
        remaining_time = deadline - time.time()
        instances_left = 3 - idx
        # Give more time to smaller/harder instances
        per_instance_time = max(10.0, remaining_time / instances_left * 1.2)
        per_instance_time = min(per_instance_time, remaining_time - 5.0)  # Leave buffer
        
        solution = min_hitting_set(cliques, adj, n, time_limit=per_instance_time)
        results.append(solution)

    return results
