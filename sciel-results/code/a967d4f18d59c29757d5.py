import numpy as np
import itertools
import time
import random
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate all words and neighbors
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Precompute neighbors for radius 1
    neighbors = []
    neighbor_sets = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbors.append(neigh)
        neighbor_sets.append(set(neigh))

    # 2. Greedy Set Cover with heap-based lazy deletion
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()

    # Build neighbor arrays for fast access
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]

    # Initialize heap with (negative_gain, index) for max-heap behavior
    # We need to compute initial gains
    gains = np.zeros(N, dtype=np.int32)
    for idx in range(N):
        gains[idx] = len(neighbors[idx])  # All uncovered initially

    # Build heap: (-gain, idx)
    heap = [(-gains[i], i) for i in range(N)]
    heapq.heapify(heap)

    while np.any(uncovered):
        if time.time() > deadline - 30:
            break

        # Pop best candidate from heap
        while heap:
            neg_gain, best_idx = heapq.heappop(heap)
            # Check if this is still valid (not already in code)
            if best_idx in code_set:
                continue
            # Recompute gain to verify
            actual_gain = int(np.sum(uncovered[neighbor_arrays[best_idx]]))
            if actual_gain == 0:
                continue
            # Verify it's still the best or close enough
            # For efficiency, we trust the heap ordering approximately
            break
        else:
            break

        if actual_gain == 0:
            break

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbor_arrays[best_idx]] = False

        # Update gains for affected neighbors
        # For each neighbor of best_idx, its gain may have changed
        # We need to push updated entries to heap
        for idx in neighbors[best_idx]:
            if idx in code_set:
                continue
            new_gain = int(np.sum(uncovered[neighbor_arrays[idx]]))
            if new_gain > 0:
                heapq.heappush(heap, (-new_gain, idx))

    # 3. Local Search with enhanced 1-exchange moves
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbor_arrays[idx]] += 1

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False

        # --- Deletion Pass ---
        to_remove = []
        for c in code:
            if np.min(counts[neighbor_arrays[c]]) > 1:
                to_remove.append(c)

        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1

        # --- Enhanced Swap Pass ---
        random.shuffle(code)

        for c in code:
            if time.time() > deadline - 5:
                break

            # Check if c is currently removable
            if np.min(counts[neighbor_arrays[c]]) > 1:
                continue

            # Get critical points for c: points i in neighbors[c] with counts[i] == 1
            crit_points = [i for i in neighbors[c] if counts[i] == 1]

            if not crit_points:
                continue

            # Strategy 1: Find replacement covering all critical points
            found_swap = False
            random.shuffle(crit_points)

            for p in crit_points:
                if found_swap:
                    break

                cands = [x for x in neighbors[p] if x not in code_set]

                best_w = -1
                best_score = -1

                for w in cands:
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break

                    if not valid:
                        continue

                    # Score: maximize redundancy creation
                    score = int(np.sum(counts[neighbor_arrays[w]] == 2))

                    if score > best_score:
                        best_score = score
                        best_w = w

                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1

                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbor_arrays[best_w]] += 1

                    improved = True
                    found_swap = True
                    break

    # Final cleanup: Remove any remaining redundant words
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbor_arrays[idx]] += 1

    changed = True
    while changed:
        changed = False
        for c in list(code):
            if np.min(counts[neighbor_arrays[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
                changed = True

    return [list(words[i]) for i in code]
