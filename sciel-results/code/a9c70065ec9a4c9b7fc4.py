import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters: simple greedy + reduction
        all_words = list(itertools.product(range(q), repeat=n))
        N = len(all_words)
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = {i}
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    lst = list(w)
                    lst[pos] = val
                    neigh_set.add(word_to_idx[tuple(lst)])
            neighbors.append(list(neigh_set))
        uncovered = np.ones(N, dtype=bool)
        C = []
        t0 = time.time()
        while np.any(uncovered) and time.time() - t0 < 100:
            best_idx = -1; best_cnt = -1
            for i in range(N):
                c = np.sum(uncovered[neighbors[i]])
                if c > best_cnt:
                    best_cnt = c; best_idx = i
            if best_cnt <= 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
        # simple reduction
        counts = np.zeros(N, dtype=int)
        for c in C: counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            newC = []
            for c in C:
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
                    changed = True
                else:
                    newC.append(c)
            C = newC
        return [list(all_words[i]) for i in C]

    # Target: q=3, n=8, radius=1
    N = 3 ** 8  # 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbors as numpy arrays (each codeword covers 17 words)
    neighbors = np.zeros((N, 17), dtype=np.int32)
    for i, w in enumerate(all_words):
        neighbors[i, 0] = i
        idx = 1
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                lst = list(w)
                lst[pos] = val
                neighbors[i, idx] = word_to_idx[tuple(lst)]
                idx += 1
    # idx should be 17 for all i

    t0 = time.time()
    deadline = 135.0  # safety margin before 150s

    def remaining_time():
        return deadline - time.time()

    def compute_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    def reduce_code(code, counts):
        if not code:
            return list(code), counts
        cur = list(code)
        cnt = counts.copy()
        while True:
            removed = False
            nxt = []
            for c in cur:
                if np.min(cnt[neighbors[c]]) >= 2:
                    cnt[neighbors[c]] -= 1
                    removed = True
                else:
                    nxt.append(c)
            cur = nxt
            if not removed:
                break
        return cur, cnt

    # ---- Vectorized greedy initialization ----
    adj = np.zeros((N, N), dtype=np.bool_)
    for i in range(N):
        adj[i, neighbors[i]] = True

    uncovered = np.ones(N, dtype=np.bool_)
    C = []
    # Greedy with vectorized gain; recompute only when needed
    # To reduce overhead, we recompute gains every step (N=6561, adj 6561x6561 bool ~ 43MB)
    # But that's 6561*6561 operations per step which is ~43M; with ~500 steps that's 21.5B - too slow.
    # Instead, use the neighbor-list approach: gain[i] = sum(uncovered[neighbors[i]])
    # We can vectorize this by using the neighbor matrix in a smarter way.
    # Actually, let's precompute a "cover mask" as an integer array and use bincount or similar.
    # Simpler: for each step, loop over candidates but vectorize the sum.
    # 6561 candidates * 17 neighbors = 111k operations per step. 500 steps = 55M. That's fine.
    
    while np.any(uncovered):
        if remaining_time() < 5:
            break
        # Compute gains for all candidates
        gains = np.zeros(N, dtype=np.int32)
        # For each candidate, count how many of its 17 neighbors are uncovered
        # neighbors is (N, 17), uncovered is (N,)
        # We want: gains[i] = sum(uncovered[neighbors[i, :]])
        # This can be done with: np.sum(uncovered[neighbors], axis=1)
        gains = np.sum(uncovered[neighbors], axis=1)
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        if best_gain == 0:
            break
        C.append(int(best_idx))
        uncovered[neighbors[best_idx]] = False

    C, counts = reduce_code(C, compute_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()

    # ---- Local search: removal + swap + 2-opt style ----
    while remaining_time() > 5:
        # Reduction pass
        C, counts = reduce_code(C, counts)
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()

        if len(C) == 0:
            break

        # Try swaps: for each codeword with few unique points, try to replace it
        # with a neighbor that covers all unique points, then reduce.
        improved = False
        indices = list(range(len(C)))
        random.shuffle(indices)

        for idx in indices:
            if remaining_time() < 10:
                break
            c = C[idx]
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]

            if len(unique_points) == 0:
                # Redundant, remove it
                C.remove(c)
                counts[neighbors[c]] -= 1
                improved = True
                break

            if len(unique_points) > 3:
                continue

            # Find candidates (not in C) that cover all unique points
            if len(unique_points) == 1:
                candidates_set = set(neighbors[unique_points[0]].tolist())
            else:
                candidates_set = set(neighbors[unique_points[0]].tolist())
                for u in unique_points[1:]:
                    candidates_set &= set(neighbors[u].tolist())
                    if not candidates_set:
                        break

            candidates_set.difference_update(C)
            if not candidates_set:
                continue

            # Try a random candidate
            c_prime = random.choice(list(candidates_set))
            old_C = C.copy()
            old_counts = counts.copy()

            C.remove(c)
            C.append(c_prime)
            counts[neighbors[c]] -= 1
            counts[neighbors[c_prime]] += 1

            if np.any(counts < 1):
                C = old_C
                counts = old_counts
                continue

            new_C, new_counts = reduce_code(C, counts)
            if len(new_C) < len(old_C):
                C = new_C
                counts = new_counts
                if len(C) < best_size:
                    best_size = len(C)
                    best_C = C.copy()
                    best_counts = counts.copy()
                improved = True
                break
            else:
                # Neutral swap to escape local minima
                C = new_C
                counts = new_counts
                improved = True
                break

        if not improved and remaining_time() > 15:
            # Random kicks: replace a random codeword with a random neighbor, then reduce
            for _ in range(20):
                if remaining_time() < 10:
                    break
                if not C:
                    break
                c_idx = random.randint(0, len(C) - 1)
                c = C[c_idx]
                cand = neighbors[c]
                c_prime = int(cand[random.randint(0, len(cand) - 1)])
                if c_prime in C:
                    continue

                old_C = C.copy()
                old_counts = counts.copy()

                C.remove(c)
                C.append(c_prime)
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1

                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue

                new_C, new_counts = reduce_code(C, counts)
                if len(new_C) < len(old_C):
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = counts.copy()
                    break
                else:
                    C = new_C
                    counts = new_counts

    # Final reduction
    best_C, _ = reduce_code(best_C, compute_counts(best_C))
    return [list(all_words[i]) for i in best_C]
