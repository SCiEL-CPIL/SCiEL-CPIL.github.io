import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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

    # Precompute intersection sizes for all pairs of supports
    # inter[i][j] = len(supports[i] & supports[j])
    inter = np.zeros((N_S, N_S), dtype=np.int8)
    # Use bitmasks for faster intersection
    masks = []
    for s in supports:
        m = 0
        for i in s:
            m |= (1 << i)
        masks.append(m)
    
    for i in range(N_S):
        m_i = masks[i]
        for j in range(i, N_S):
            m_j = masks[j]
            inter_len = bin(m_i & m_j).count('1')
            inter[i, j] = inter_len
            inter[j, i] = inter_len

    # Conflict graph for independent set (intersection >= 3)
    # We want to maximize the number of selected supports such that intersection <= 2.
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(N_S):
            if i != j and inter[i, j] >= 3:
                adj[i].add(j)
                adj[j].add(i)

    def greedy_mis(adj, order):
        in_set = [False] * N_S
        available = set(range(N_S))
        for v in order:
            if v in available:
                in_set[v] = True
                # Remove v and its neighbors
                to_remove = {v} | adj[v]
                available -= to_remove
        return in_set

    def local_search_mis(in_set, adj):
        # Try to add nodes
        changed = True
        while changed:
            changed = False
            for v in range(N_S):
                if not in_set[v]:
                    # Check if any neighbor is in set
                    conflict = False
                    for u in adj[v]:
                        if in_set[u]:
                            conflict = True
                            break
                    if not conflict:
                        in_set[v] = True
                        changed = True
        return in_set

    best_size = 0
    best_set = []
    rng = random.Random(42)

    num_restarts = 5000
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break

        # Random order
        order = list(range(N_S))
        rng.shuffle(order)

        in_set = greedy_mis(adj, order)
        in_set = local_search_mis(in_set, adj)

        size = sum(in_set)
        if size > best_size:
            best_size = size
            best_set = [i for i, b in enumerate(in_set) if b]
            # Optional: try to improve best_set further with local search
            # But local_search_mis is already applied.

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

    # Greedy fill with sign checks for remaining supports?
    # The independent set ensures supports intersect in <= 2.
    # For any two supports in best_set, any pair of vectors is compatible.
    # So we can just take all 16 vectors for each support in best_set.
    # This is already done above.
    
    # Can we add vectors from supports NOT in best_set?
    # If a support S is not in best_set, it intersects some S' in best_set in >= 3.
    # If we add a vector v from S, we must check against all vectors in S'.
    # This is complex. The gain might be small.
    # However, we can try to add vectors from supports that intersect in 3, 
    # if we can find a sign assignment that is compatible with all existing vectors.
    
    # Let's try to add more vectors greedily from all possible Type D vectors
    # that are not already in vectors_D.
    
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

    # Shuffle indices of all_vectors_D
    indices_all = list(range(len(all_vectors_D)))
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 5:
            break

        v = all_vectors_D[i]
        if tuple(v) in current_set:
            continue

        can_add = True
        # Check compatibility with all current vectors
        # dot(v, u) <= 2
        for u in current_vectors:
            if np.dot(v, u) > 2:
                can_add = False
                break

        if can_add:
            current_vectors.append(v)
            current_set.add(tuple(v))

    final_vectors_D = np.array(current_vectors, dtype=np.int64)

    final_vectors = np.vstack([vectors_A, final_vectors_D])

    return final_vectors
