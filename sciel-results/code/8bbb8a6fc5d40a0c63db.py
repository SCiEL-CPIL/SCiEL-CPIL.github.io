import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 105.0  # Leave 15s margin for safety (evaluator kills at 150s, but 120s is the task limit)

    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))

    # 1. Build edges and index mapping
    edges = []
    edge_to_idx = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx
                idx += 1

    # 2. Build squares (2-faces)
    # A square is defined by two dimensions i, j and fixed values for other dims.
    # It contains 4 edges.
    squares = []
    # For each pair of dims (i, j)
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over all combinations of the other d-2 bits
            # There are 2^(d-2) such squares
            other_dims = [k for k in range(d) if k != i and k != j]
            for mask in range(1 << len(other_dims)):
                # Construct the 4 vertices of the square
                # Base vertex: bits for other_dims from mask, bits i,j are 0
                v00 = 0
                for k, dim in enumerate(other_dims):
                    if mask & (1 << k):
                        v00 |= (1 << dim)

                v10 = v00 | (1 << i)
                v01 = v00 | (1 << j)
                v11 = v00 | (1 << i) | (1 << j)

                # The 4 edges of this square
                e1 = edge_to_idx[(min(v00, v10), max(v00, v10))]
                e2 = edge_to_idx[(min(v10, v11), max(v10, v11))]
                e3 = edge_to_idx[(min(v11, v01), max(v11, v01))]
                e4 = edge_to_idx[(min(v01, v00), max(v01, v00))]

                squares.append([e1, e2, e3, e4])

    n_squares = len(squares)

    # 3. Precompute for each edge, the squares it belongs to
    edge_to_squares = [[] for _ in range(n_edges)]
    for s_idx, sq in enumerate(squares):
        for e_idx in sq:
            edge_to_squares[e_idx].append(s_idx)

    # 4. Greedy Hitting Set Initialization
    # We want to find a set U of edges to remove such that every square has at least one edge in U.
    # Minimize |U|.

    def greedy_hitting_set(rng):
        """
        Greedy set cover to find a small hitting set U.
        Returns in_U, cover_count, U_size.
        """
        cover_count = [0] * n_squares
        in_U = [False] * n_edges
        U_size = 0
        uncovered = set(range(n_squares))

        # Shuffle edges for randomized greedy
        edge_order = list(range(n_edges))
        rng.shuffle(edge_order)

        while uncovered:
            # Find edge that covers most uncovered squares
            best_e = -1
            max_cover = 0
            # Optimization: iterate all edges, n_edges=448 is small
            for e in edge_order:
                if in_U[e]:
                    continue
                count = 0
                for s_idx in edge_to_squares[e]:
                    if s_idx in uncovered:
                        count += 1
                if count > max_cover:
                    max_cover = count
                    best_e = e

            if best_e == -1 or max_cover == 0:
                break

            # Add best_e to U
            in_U[best_e] = True
            U_size += 1
            for s_idx in edge_to_squares[best_e]:
                if s_idx in uncovered:
                    cover_count[s_idx] += 1
                    if cover_count[s_idx] > 0:
                        uncovered.discard(s_idx)

        return in_U, cover_count, U_size

    def local_search(in_U, cover_count, U_size, time_limit, rng):
        """
        Simulated Annealing / Local Search to minimize |U|.
        Returns best_in_U, best_U_size.
        """
        best_U_size = U_size
        best_in_U = in_U[:]

        T = 1.0
        cooling = 0.9995
        min_T = 0.001

        # Precompute list of edges in U and not in U for faster sampling?
        # Actually, just picking random edge is fine for 448 edges.

        while time.time() < time_limit:
            # Pick random edge
            e = rng.randint(0, n_edges - 1)

            if in_U[e]:
                # Try to remove e
                # Check if any square in edge_to_squares[e] has cover_count == 1
                # If so, removing e makes it 0 (uncovered) -> Invalid
                valid = True
                for s_idx in edge_to_squares[e]:
                    if cover_count[s_idx] == 1:
                        valid = False
                        break

                if valid:
                    # Remove e
                    in_U[e] = False
                    U_size -= 1
                    for s_idx in edge_to_squares[e]:
                        cover_count[s_idx] -= 1

                    if U_size < best_U_size:
                        best_U_size = U_size
                        best_in_U = in_U[:]
                else:
                    # Invalid removal, skip
                    pass
            else:
                # Try to add e
                # This increases U_size, so it's a "worsening" move
                # Accept with probability exp(-1/T)
                if rng.random() < math.exp(-1.0 / T):
                    in_U[e] = True
                    U_size += 1
                    for s_idx in edge_to_squares[e]:
                        cover_count[s_idx] += 1

            T *= cooling
            if T < min_T:
                T = min_T

        return best_in_U, best_U_size

    # 5. Main Loop: Multiple restarts with different random seeds
    # Each restart: Greedy -> Local Search
    best_overall_U_size = n_edges
    best_overall_in_U = None

    # Use a fixed seed for reproducibility within the run, but vary it per restart
    base_seed = 42

    # Estimate how many restarts we can do
    # Greedy is fast (~ms), Local Search takes remaining time
    # Let's do a few quick restarts first, then one long one?
    # Or just loop until deadline.

    restart_count = 0
    while time.time() < deadline:
        rng = random.Random(base_seed + restart_count)
        restart_count += 1

        # Greedy
        in_U, cover_count, U_size = greedy_hitting_set(rng)

        # Local Search
        # Give it a time slice. If we have lots of time left, give more.
        # Simple strategy: give each LS 5 seconds, or remaining time if last.
        ls_time = time.time() + 5.0
        if time.time() > deadline - 10:
            ls_time = deadline

        best_in_U, best_U_size = local_search(in_U, cover_count, U_size, ls_time, rng)

        if best_U_size < best_overall_U_size:
            best_overall_U_size = best_U_size
            best_overall_in_U = best_in_U

        # If we've found a very good solution (close to theoretical lower bound?),
        # we might stop early? No, keep searching.

    # 6. Construct output
    # The kept edges are those NOT in best_overall_in_U
    out = []
    for i in range(n_edges):
        if not best_overall_in_U[i]:
            u, v = edges[i]
            out.append([u, v])

    return out
