import random
import time
import math
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # safety margin before 120s kill

    def verify(adj_sets):
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj_sets[u] & adj_sets[v]) > 2:
                    return False
        return True

    def to_edge_list(adj_sets):
        return [[u, v] for u in range(n) for v in adj_sets[u] if u < v]

    def from_edge_list(edges):
        adj_sets = [set() for _ in range(n)]
        for u, v in edges:
            adj_sets[u].add(v)
            adj_sets[v].add(u)
        return adj_sets

    def build_common(adj_sets):
        common = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in adj_sets[u]:
                if u < v:
                    for w in adj_sets[u]:
                        if w in adj_sets[v] and w != u and w != v:
                            common[u][v] += 1
        # Actually, simpler to just compute directly
        common = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u][v] = len(adj_sets[u] & adj_sets[v])
        return common

    def can_add(adj, common, u, v):
        if u == v or v in adj[u]:
            return False
        if common[u][v] > 2:
            return False
        for y in adj[v]:
            if common[u][y] > 1:
                return False
        for y in adj[u]:
            if common[v][y] > 1:
                return False
        return True

    def add_edge(adj, common, u, v):
        adj[u].add(v)
        adj[v].add(u)
        for y in adj[v]:
            common[u][y] += 1
            common[y][u] += 1
        for y in adj[u]:
            common[v][y] += 1
            common[y][v] += 1

    def remove_edge(adj, common, u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        for y in adj[v]:
            common[u][y] -= 1
            common[y][u] -= 1
        for y in adj[u]:
            common[v][y] -= 1
            common[y][v] -= 1

    def get_edge_count(adj):
        return sum(len(s) for s in adj) // 2

    def exhaustive_polish(adj, common):
        improved = True
        while improved:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u] and can_add(adj, common, u, v):
                        add_edge(adj, common, u, v)
                        improved = True

    def numpy_greedy_fill(adj, common, seed):
        """
        Vectorized greedy fill using NumPy adjacency matrix.
        Repeatedly finds all simultaneously addable edges and adds them.
        """
        # Build numpy adjacency matrix
        A = np.zeros((n, n), dtype=np.int8)
        for u in range(n):
            for v in adj[u]:
                A[u, v] = 1
                A[v, u] = 1

        # Precompute all missing edges
        missing_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
        if not missing_edges:
            return

        # Iteratively add edges that are simultaneously valid
        # We do this in batches: check all missing edges, add all valid ones at once
        # Then re-check (since adding edges changes validity)
        
        max_iters = 50
        for _ in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            if not missing_edges:
                break

            # For each missing edge (u,v), check if it can be added
            # We need to check against the CURRENT state of adj/common
            # But we want to add multiple at once. 
            # Safe approach: add edges one by one but in a smart order, 
            # or check all, then add all that are valid against the CURRENT state.
            # If we add a set of edges that are pairwise non-interfering, it's safe.
            # Two edges (u,v) and (x,y) interfere if they share a vertex or if adding one
            # makes the other invalid.
            # Simplest safe vectorized approach: 
            # 1. Check all missing edges against current adj/common.
            # 2. Collect valid ones.
            # 3. Add them all.
            # 4. Update missing_edges.
            
            valid_edges = []
            for u, v in missing_edges:
                if can_add(adj, common, u, v):
                    valid_edges.append((u, v))

            if not valid_edges:
                break

            # Add all valid edges
            for u, v in valid_edges:
                add_edge(adj, common, u, v)
                A[u, v] = 1
                A[v, u] = 1

            # Update missing_edges
            added_set = set(valid_edges)
            missing_edges = [(u, v) for u, v in missing_edges if (u, v) not in added_set]
            
            # Also, some edges that were previously invalid might now be... no, adding edges
            # only makes it harder to add more. So we just remove the added ones.
            # But wait, adding an edge (u,v) increases common counts, which might make
            # other edges INVALID. It never makes an invalid edge VALID.
            # So we can just remove the added edges from missing_edges.

    def simulated_annealing(adj, common, rng, time_budget):
        current_count = get_edge_count(adj)
        sa_end_time = start_time + time_budget
        T = 2.0
        cooling = 0.999

        while (time.time() - start_time) < sa_end_time:
            if T < 0.01:
                break

            move_type = rng.random()

            if move_type < 0.5:
                # Try to add a random missing edge
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v or v in adj[u]:
                    continue
                if can_add(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    current_count += 1

            elif move_type < 0.8:
                # Swap: remove one edge, try to add another
                edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                if not edges_list:
                    continue
                u1, v1 = rng.choice(edges_list)
                
                found_swap = False
                for _ in range(30):
                    u2 = rng.randint(0, n - 1)
                    v2 = rng.randint(0, n - 1)
                    if u2 == v2 or v2 in adj[u2]:
                        continue
                    if {u2, v2} == {u1, v1}:
                        continue
                    remove_edge(adj, common, u1, v1)
                    if can_add(adj, common, u2, v2):
                        add_edge(adj, common, u2, v2)
                        found_swap = True
                        break
                    else:
                        add_edge(adj, common, u1, v1)
                
                if not found_swap:
                    # Try to accept a non-improving move to escape local optima
                    # Remove a random edge with low probability
                    if rng.random() < 0.1:
                        u1, v1 = rng.choice(edges_list)
                        remove_edge(adj, common, u1, v1)
                        current_count -= 1

            else:
                # Remove a random edge (exploration)
                edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                if not edges_list:
                    continue
                u, v = rng.choice(edges_list)
                remove_edge(adj, common, u, v)
                current_count -= 1

            T *= cooling

        return current_count

    def structured_init(adj, common):
        """Initialize with a structured pattern."""
        for i in range(n):
            for d in [1, 2, 3]:
                j = (i + d) % n
                if i < j and can_add(adj, common, i, j):
                    add_edge(adj, common, i, j)
        for i in range(n):
            for d in [5, 7]:
                j = (i + d) % n
                if i < j and can_add(adj, common, i, j):
                    add_edge(adj, common, i, j)

    # Multiple restarts
    best_edges = []
    best_count = 0
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333, 444, 555, 666, 777]

    for seed in seeds:
        if time.time() - start_time > time_limit:
            break

        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        common = [[0] * n for _ in range(n)]

        if seed % 2 == 0:
            structured_init(adj, common)
        else:
            # Use numpy_greedy_fill for faster initialization
            numpy_greedy_fill(adj, common, seed)

        # Simulated annealing
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            break
        sa_time = remaining * 0.7
        simulated_annealing(adj, common, rng, sa_time)

        # Exhaustive polish
        exhaustive_polish(adj, common)

        curr_count = get_edge_count(adj)
        if curr_count > best_count:
            best_count = curr_count
            best_edges = to_edge_list(adj)

    # Fallback if no edges found
    if not best_edges:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    # Final verification
    adj_check = from_edge_list(best_edges)
    if not verify(adj_check):
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    return best_edges
