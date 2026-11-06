"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    N = 40
    E = 401
    TOTAL_PAIRS = N * (N - 1) // 2  # 780

    def build_adjacency(edges):
        A = np.zeros((N, N), dtype=np.int32)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        return A

    def compute_books(A):
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        return A2

    def get_max_book(A):
        A2 = compute_books(A)
        return int(np.max(A2))

    def spectral_init():
        """Initialize using spectral method: start from balanced bipartite, add edges greedily minimizing max book."""
        # Start with K_{20,20} (400 edges) + 1 extra edge
        left = list(range(20))
        right = list(range(20, 40))
        edges = set()
        for u in left:
            for v in right:
                edges.add((u, v))
        # Add one more edge within left side
        edges.add((0, 1))
        edges = list(edges)

        A = build_adjacency(edges)
        A2 = compute_books(A)
        current_max = int(np.max(A2))

        # Greedy improvement: try to replace edges to reduce max book
        edge_set = set((min(u,v), max(u,v)) for u, v in edges)
        non_edges = set()
        for i in range(N):
            for j in range(i+1, N):
                if (i, j) not in edge_set:
                    non_edges.add((i, j))
        non_edges = list(non_edges)

        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            A2 = compute_books(A)
            max_val = int(np.max(A2))

            # Find edges with max book
            high_edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if A[i,j] == 1 and A2[i,j] == max_val:
                        high_edges.append((i,j))

            if not high_edges:
                break

            for rem_u, rem_v in high_edges:
                if time.time() > deadline - 30:
                    break
                # Try all non-edges, find one that minimizes new max book
                best_add = None
                best_new_max = max_val + 1

                # Sample a subset of non-edges for speed
                sample_size = min(100, len(non_edges))
                candidates = random.sample(non_edges, sample_size)

                for add_u, add_v in candidates:
                    # Quick estimate: book size of new edge = common neighbors
                    common = int(np.sum(A[add_u, :] * A[add_v, :]))
                    # After removing (rem_u, rem_v), common neighbors might change slightly
                    # But common is a good lower bound for the new edge's book
                    if common >= best_new_max:
                        continue

                    # Do the swap and check
                    A[rem_u, rem_v] = 0
                    A[rem_v, rem_u] = 0
                    A[add_u, add_v] = 1
                    A[add_v, add_u] = 1

                    new_A2 = compute_books(A)
                    new_max = int(np.max(new_A2))

                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_add = (add_u, add_v)

                    # Revert
                    A[rem_u, rem_v] = 1
                    A[rem_v, rem_u] = 1
                    A[add_u, add_v] = 0
                    A[add_v, add_u] = 0

                if best_add is not None and best_new_max < max_val:
                    # Apply the best swap
                    A[rem_u, rem_v] = 0
                    A[rem_v, rem_u] = 0
                    A[best_add[0], best_add[1]] = 1
                    A[best_add[1], best_add[0]] = 1

                    edge_set.discard((min(rem_u,rem_v), max(rem_u,rem_v)))
                    edge_set.add(tuple(sorted(best_add)))
                    non_edges.remove(best_add)
                    non_edges.append((min(rem_u,rem_v), max(rem_u,rem_v)))

                    improved = True
                    current_max = best_new_max

        final_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if A[i, j] == 1:
                    final_edges.append((i, j))
        return final_edges, current_max

    def fast_local_search(A, time_limit=60.0):
        """Local search with incremental book updates for speed."""
        A2 = compute_books(A)
        current_max = int(np.max(A2))
        edge_set = set()
        for i in range(N):
            for j in range(i+1, N):
                if A[i,j] == 1:
                    edge_set.add((i,j))
        non_edges = set()
        for i in range(N):
            for j in range(i+1, N):
                if (i,j) not in edge_set:
                    non_edges.add((i,j))
        non_edges = list(non_edges)
        edge_list = list(edge_set)

        no_improve = 0
        max_no_improve = 3000
        end_time = time.time() + time_limit

        while time.time() < end_time and no_improve < max_no_improve:
            A2 = compute_books(A)
            max_val = int(np.max(A2))
            if max_val < current_max:
                current_max = max_val
                no_improve = 0
            else:
                # no_improve increments only when we fail to improve
                no_improve += 1

            # Find max-book edges
            high_edges = [(i,j) for i in range(N) for j in range(i+1,N)
                         if A[i,j]==1 and A2[i,j]==max_val]
            if not high_edges:
                break

            rem_u, rem_v = random.choice(high_edges)

            # Find best replacement: minimize new max book
            best_add = None
            best_new_max = max_val + 1
            best_new_sum = float('inf')

            # Sample non-edges
            sample_size = min(80, len(non_edges))
            candidates = random.sample(non_edges, sample_size)

            for add_u, add_v in candidates:
                common = int(np.sum(A[add_u, :] * A[add_v, :]))
                if common >= max_val:
                    continue

                A[rem_u, rem_v] = 0
                A[rem_v, rem_u] = 0
                A[add_u, add_v] = 1
                A[add_v, add_u] = 1

                new_A2 = compute_books(A)
                new_max = int(np.max(new_A2))
                new_sum = float(np.sum(new_A2**2))

                if new_max < best_new_max or (new_max == best_new_max and new_sum < best_new_sum):
                    best_new_max = new_max
                    best_new_sum = new_sum
                    best_add = (add_u, add_v)

                A[rem_u, rem_v] = 1
                A[rem_v, rem_u] = 1
                A[add_u, add_v] = 0
                A[add_v, add_u] = 0

            if best_add is not None and best_new_max < max_val:
                A[rem_u, rem_v] = 0
                A[rem_v, rem_u] = 0
                A[best_add[0], best_add[1]] = 1
                A[best_add[1], best_add[0]] = 1
                edge_set.discard((min(rem_u,rem_v), max(rem_u,rem_v)))
                edge_set.add(tuple(sorted(best_add)))
                non_edges.remove(best_add)
                non_edges.append((min(rem_u,rem_v), max(rem_u,rem_v)))
                current_max = best_new_max
                no_improve = 0
            else:
                no_improve += 1

        final_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if A[i, j] == 1:
                    final_edges.append((i, j))
        return final_edges, current_max

    # Main: use spectral init + local search
    best_edges = None
    best_max = float('inf')

    # Phase 1: Spectral initialization
    edges, max_val = spectral_init()
    if max_val < best_max:
        best_max = max_val
        best_edges = edges.copy()

    # Phase 2: Local search from spectral result
    A = build_adjacency(edges)
    edges, max_val = fast_local_search(A, time_limit=min(50, deadline - time.time() - 10))
    if max_val < best_max:
        best_max = max_val
        best_edges = edges.copy()

    # Phase 3: Try a few random restarts if time permits
    if time.time() < deadline - 20:
        for r in range(3):
            if time.time() > deadline - 15:
                break
            # Random start
            all_pairs = [(i,j) for i in range(N) for j in range(i+1,N)]
            random.shuffle(all_pairs)
            rand_edges = all_pairs[:E]
            A = build_adjacency(rand_edges)
            rv = get_max_book(A)
            rand_edges, rv = fast_local_search(A, time_limit=min(25, deadline - time.time() - 10))
            if rv < best_max:
                best_max = rv
                best_edges = rand_edges.copy()

    if best_edges is None:
        all_pairs = [(i,j) for i in range(N) for j in range(i+1,N)]
        random.shuffle(all_pairs)
        best_edges = all_pairs[:E]

    result = [[u, v] for u, v in best_edges]
    return result

# EVOLVE-BLOCK-END
