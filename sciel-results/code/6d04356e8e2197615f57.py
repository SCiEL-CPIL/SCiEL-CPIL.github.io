import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    # Norm squared = 4.
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)

    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    # Norm squared = 4.
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Convert supports to binary matrix for fast intersection computation
    support_matrix = np.zeros((N_S, D), dtype=np.uint8)
    for i, sup in enumerate(supports):
        for idx in sup:
            support_matrix[i, idx] = 1

    # Compute intersection sizes via matrix multiply
    inter = (support_matrix @ support_matrix.T).astype(np.int8)

    # Conflict graph: two supports conflict if intersection >= 3
    conflict_mask = (inter >= 3)
    # Build adjacency lists
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        neighbors = np.where(conflict_mask[i])[0]
        adj[i] = set(neighbors.tolist())

    def greedy_mis(adj, order):
        in_set = [False] * N_S
        available = set(range(N_S))
        for v in order:
            if v in available:
                in_set[v] = True
                to_remove = {v} | adj[v]
                available -= to_remove
        return in_set

    def local_search_mis(in_set, adj):
        in_set = list(in_set)
        changed = True
        while changed:
            changed = False
            for v in range(N_S):
                if not in_set[v]:
                    if not any(in_set[u] for u in adj[v]):
                        in_set[v] = True
                        changed = True
        return in_set

    best_size = 0
    best_set = []
    rng = random.Random(42)

    num_restarts = 3000

    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break

        order = list(range(N_S))
        rng.shuffle(order)

        in_set = greedy_mis(adj, order)
        in_set = local_search_mis(in_set, adj)

        size = sum(in_set)
        if size > best_size:
            best_size = size
            best_set = [i for i, b in enumerate(in_set) if b]

    # Generate vectors for best_set
    vectors_D = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors_D.append(v)

    vectors_D = np.array(vectors_D, dtype=np.int64)

    # Greedy fill with sign checks - fix the bug (use s, not support)
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)

    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)

    current_vectors = list(vectors_D)
    current_set = set(map(tuple, current_vectors))
    current_arr = np.vstack(current_vectors) if current_vectors else np.empty((0, D), dtype=np.int64)

    indices_all = list(range(len(all_vectors_D)))
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 5:
            break

        v = all_vectors_D[i]
        if tuple(v) in current_set:
            continue

        if len(current_arr) == 0:
            can_add = True
        else:
            dots = current_arr @ v
            can_add = np.all(dots <= 2)

        if can_add:
            current_vectors.append(v)
            current_set.add(tuple(v))
            current_arr = np.vstack([current_arr, v.reshape(1, -1)])

    final_vectors_D = np.array(current_vectors, dtype=np.int64)

    final_vectors = np.vstack([vectors_A, final_vectors_D])

    return final_vectors
