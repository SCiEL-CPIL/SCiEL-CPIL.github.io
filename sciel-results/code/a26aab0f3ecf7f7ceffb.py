import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin

    N = 256

    # Build conflict sets
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # conflict[a] = set of b in [1,N] such that a+b is a perfect square
    conflict = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                conflict[a].add(b)

    # Numbers where 2a is a perfect square are forbidden (self-conflict)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]

    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}
    idx_to_num = valid_numbers

    # Build adjacency as bitmasks (conflict graph)
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for b in conflict[a]:
            if b in num_to_idx:
                j = num_to_idx[b]
                if j != i:
                    adj[i] |= (1 << j)

    # Compute degrees
    degrees = [bin(adj[i]).count('1') for i in range(M)]

    def popcount(x):
        return bin(x).count('1')

    def greedy_init(order):
        selected = 0
        for i in order:
            if (adj[i] & selected) == 0:
                selected |= (1 << i)
        return selected

    # Build complement graph adjacency (non-conflict edges)
    # comp_adj[i] = bitmask of vertices that do NOT conflict with i (excluding i itself)
    comp_adj = [0] * M
    all_mask = (1 << M) - 1
    for i in range(M):
        # Non-neighbors in conflict graph = complement
        non_conflict = all_mask & ~(adj[i] | (1 << i))
        comp_adj[i] = non_conflict

    # Maximum clique on complement graph = Maximum independent set on conflict graph
    # Tomita-style branch and bound with greedy coloring
    def max_clique_bnb():
        best_clique = [0] * M
        best_size = [0]

        # Greedy coloring for upper bound
        def color_sort(P):
            """Color the subgraph induced by P, return (coloring, order)"""
            # P is a bitmask
            vertices = [i for i in range(M) if (P >> i) & 1]
            # Greedy coloring: assign smallest possible color
            color_of = {}
            color_class = {}  # color -> list of vertices
            max_color = 0
            # Order by degree in the induced subgraph (descending)
            verts_sorted = sorted(vertices, key=lambda v: popcount(comp_adj[v] & P), reverse=True)
            for v in verts_sorted:
                # Find smallest color not used by neighbors
                used_colors = set()
                for u in range(M):
                    if (comp_adj[v] >> u) & 1 and u in color_of:
                        used_colors.add(color_of[u])
                c = 0
                while c in used_colors:
                    c += 1
                color_of[v] = c
                if c not in color_class:
                    color_class[c] = []
                color_class[c].append(v)
                max_color = max(max_color, c + 1)
            # Return vertices ordered by color (ascending), within same color by index
            order = []
            for c in range(max_color):
                if c in color_class:
                    order.extend(color_class[c])
            colors = [color_of[v] for v in order]
            return order, colors

        def expand(R, P, depth):
            if time.time() > deadline - 1:
                return
            if not P:
                if depth > best_size[0]:
                    best_size[0] = depth
                    for i in range(depth):
                        best_clique[i] = R[i]
                return
            # Get upper bound via coloring
            order, colors = color_sort(P)
            # Process in reverse color order
            n = len(order)
            if n == 0:
                return
            # Upper bound: max color + 1
            if depth + n <= best_size[0]:
                return

            # Branch on vertices in reverse order (highest color first)
            for k in range(n - 1, -1, -1):
                if time.time() > deadline - 1:
                    return
                v = order[k]
                # Upper bound check
                if depth + k + 1 <= best_size[0]:
                    return
                # Check if v is still in P
                if not ((P >> v) & 1):
                    continue
                # Include v in clique
                R[depth] = v
                # New P: intersection of P with neighbors of v
                new_P = P & comp_adj[v]
                # Remove v from P
                P &= ~(1 << v)
                expand(R, new_P, depth + 1)

        # Initial greedy lower bound
        degree_order = sorted(range(M), key=lambda i: degrees[i])
        init_mask = greedy_init(degree_order)
        init_size = popcount(init_mask)
        best_size[0] = init_size
        # Set initial best clique from greedy
        clique_verts = [i for i in range(M) if (init_mask >> i) & 1]
        for i in range(init_size):
            best_clique[i] = clique_verts[i]

        # Start B&B
        R = [0] * M
        full_P = all_mask
        expand(R, full_P, 0)

        return best_size[0], best_clique[:best_size[0]]

    # Local search: try adding a vertex and removing its neighbors
    def local_search(mask, size, steps, t_start, t_end):
        current_mask = mask
        current_size = size
        t = t_start
        if steps > 0:
            cooling_rate = (t_end / t_start) ** (1.0 / steps)
        else:
            cooling_rate = 1.0

        for step in range(steps):
            if time.time() > deadline:
                break

            # Pick a random vertex not in set
            candidates = [i for i in range(M) if not (current_mask >> i) & 1]
            if not candidates:
                break
            v = random.choice(candidates)

            neighbors_in_set = adj[v] & current_mask
            n_neighbors = popcount(neighbors_in_set)

            if n_neighbors == 0:
                # Can add directly
                trial_mask = current_mask | (1 << v)
                trial_size = current_size + 1
                delta_e = -1
            else:
                # Remove all neighbors, add v
                trial_mask = (current_mask & ~neighbors_in_set) | (1 << v)
                trial_size = current_size - n_neighbors + 1
                delta_e = n_neighbors - 1

            # Acceptance criterion
            if delta_e <= 0:
                current_mask = trial_mask
                current_size = trial_size
            else:
                if random.random() < math.exp(-delta_e / t):
                    current_mask = trial_mask
                    current_size = trial_size

            t *= cooling_rate

        return current_mask, current_size

    # Two-vertex swap: remove two vertices, add one or two
    def two_swap_search(mask, size, steps):
        current_mask = mask
        current_size = size

        for step in range(steps):
            if time.time() > deadline:
                break

            # Try adding a vertex v, removing its neighbors
            # If we can add v and remove <= 1 neighbor, it's a net gain or neutral
            candidates = [i for i in range(M) if not (current_mask >> i) & 1]
            if not candidates:
                break
            v = random.choice(candidates)
            neighbors_in_set = adj[v] & current_mask
            n_neighbors = popcount(neighbors_in_set)

            if n_neighbors <= 1:
                trial_mask = (current_mask & ~neighbors_in_set) | (1 << v)
                trial_size = current_size - n_neighbors + 1
                if trial_size > current_size:
                    current_mask = trial_mask
                    current_size = trial_size

        return current_mask, current_size

    random.seed(42)

    # Strategy 1: Greedy + SA
    degree_order = sorted(range(M), key=lambda i: degrees[i])
    best_mask = greedy_init(degree_order)
    best_size = popcount(best_mask)

    mask, size = local_search(best_mask, best_size, 50000, 100.0, 0.01)
    if size > best_size:
        best_mask = mask
        best_size = size

    # Strategy 2: Multiple restarts with random greedy + SA
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        random_order = list(range(M))
        random.shuffle(random_order)
        init_mask = greedy_init(random_order)
        init_size = popcount(init_mask)

        mask, size = local_search(init_mask, init_size, 20000, 50.0, 0.01)
        if size > best_size:
            best_mask = mask
            best_size = size

    # Strategy 3: Branch and bound on complement graph (max clique)
    if time.time() < deadline - 10:
        bnb_size, bnb_clique = max_clique_bnb()
        if bnb_size > best_size:
            # Convert clique to mask
            bnb_mask = 0
            for v in bnb_clique:
                bnb_mask |= (1 << v)
            best_mask = bnb_mask
            best_size = bnb_size

    # Strategy 4: Polish best solution with SA
    if time.time() < deadline - 3:
        mask, size = local_search(best_mask, best_size, 30000, 5.0, 0.001)
        if size > best_size:
            best_mask = mask
            best_size = size

    # Strategy 5: Two-swap polish
    if time.time() < deadline - 2:
        mask, size = two_swap_search(best_mask, best_size, 100000)
        if size > best_size:
            best_mask = mask
            best_size = size

    # Convert back to actual numbers
    result = [idx_to_num[i] for i in range(M) if (best_mask >> i) & 1]

    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]

    return sorted(result)
