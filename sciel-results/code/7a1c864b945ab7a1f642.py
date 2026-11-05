import math
import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = 256
    # Precompute perfect squares up to 256+256=512
    squares = set()
    for k in range(1, 23):  # 22^2=484, 23^2=529
        sq = k * k
        if sq <= 512:
            squares.add(sq)

    # Build conflict graph: edge between i and j if i+j is a perfect square
    # Include self-loops: i+i is a perfect square (diagonal constraint)
    # We work with 0-indexed internally (0..255), values are 1..256
    # conflict[i][j] = True if (i+1)+(j+1) is a perfect square
    # For independent set, we need NO edges between selected vertices
    # Self-loop: if 2*(i+1) is a perfect square, vertex i cannot be selected at all

    # First, identify vertices that are forbidden (self-loop)
    forbidden = [False] * n
    for i in range(n):
        val = i + 1
        if 2 * val in squares:
            forbidden[i] = True

    # Build adjacency: adj[i] = set of j != i such that (i+1)+(j+1) is square
    adj = [set() for _ in range(n)]
    for i in range(n):
        vi = i + 1
        for j in range(n):
            if i == j:
                continue
            vj = j + 1
            s = vi + vj
            if s in squares:
                adj[i].add(j)

    # Remove forbidden vertices from consideration
    candidates = [i for i in range(n) if not forbidden[i]]
    cand_set = set(candidates)

    # Also remove edges to forbidden vertices (they can't be in set anyway)
    for i in range(n):
        adj[i] = adj[i] & cand_set

    # Convert to sorted lists for faster iteration
    adj_lists = [sorted(adj[i]) for i in range(n)]

    # Greedy initial solution: degree-based ordering
    def greedy_solution():
        remaining = set(candidates)
        selected = set()
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg = len(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            selected.add(best_v)
            remaining.remove(best_v)
            # Remove neighbors of best_v from remaining
            for nb in adj[best_v]:
                if nb in remaining:
                    remaining.remove(nb)
        return selected

    best_set = greedy_solution()
    best_size = len(best_set)

    # Local search: try to add vertices, or swap
    def local_search(s):
        s = set(s)
        improved = True
        while improved:
            improved = False
            # Try adding each non-forbidden vertex not in s
            for v in candidates:
                if v in s:
                    continue
                # Check if v conflicts with any in s
                conflict = False
                for nb in adj[v]:
                    if nb in s:
                        conflict = True
                        break
                if not conflict:
                    s.add(v)
                    improved = True
        return s

    best_set = local_search(best_set)
    best_size = len(best_set)

    # Branch and bound with greedy coloring upper bound
    def greedy_color_upper_bound(vertices):
        """Upper bound via greedy coloring of the induced subgraph on vertices."""
        if not vertices:
            return 0
        vertices = list(vertices)
        # Greedy coloring
        color = {}
        max_color = 0
        # Order by degree descending for tighter bound
        vertices_sorted = sorted(vertices, key=lambda v: -len(adj[v] & set(vertices)))
        for v in vertices_sorted:
            used_colors = set()
            for nb in adj[v]:
                if nb in color:
                    used_colors.add(color[nb])
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            max_color = max(max_color, c + 1)
        return max_color

    # Branch and bound
    def branch_and_bound(available, current_set, current_size):
        nonlocal best_set, best_size
        if time.time() > deadline:
            return

        if not available:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return

        # Upper bound: current_size + greedy coloring of available subgraph
        ub = current_size + greedy_color_upper_bound(available)
        if ub <= best_size:
            return

        # Choose branch vertex: pick one with max degree in available
        avail_list = list(available)
        branch_v = max(avail_list, key=lambda v: len(adj[v] & set(available)))

        # Branch 1: include branch_v
        new_available = available - {branch_v}
        for nb in adj[branch_v]:
            if nb in new_available:
                new_available.discard(nb)
        branch_and_bound(new_available, current_set | {branch_v}, current_size + 1)

        # Branch 2: exclude branch_v
        branch_and_bound(available - {branch_v}, current_set, current_size)

    # Run B&B with initial candidates
    if time.time() < deadline:
        branch_and_bound(set(candidates), set(), 0)

    # Additional local search improvements
    if time.time() < deadline:
        # Try multiple random restarts with different orderings
        for _ in range(20):
            if time.time() > deadline:
                break
            # Random greedy
            remaining = set(candidates)
            selected = set()
            order = list(remaining)
            random.shuffle(order)
            for v in order:
                if v not in remaining:
                    continue
                conflict = False
                for nb in adj[v]:
                    if nb in selected:
                        conflict = True
                        break
                if not conflict:
                    selected.add(v)
                    remaining.discard(v)
                    for nb in adj[v]:
                        remaining.discard(nb)
            selected = local_search(selected)
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = selected

    # Final verification and return
    result = sorted(v + 1 for v in best_set)
    if not result:
        result = [1]  # fallback nonempty

    return result
