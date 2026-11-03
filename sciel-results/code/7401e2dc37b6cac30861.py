import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    limit = 115.0
    deadline = start_time + limit

    def _deadline_ok():
        return time.time() < deadline

    # Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Map block to covered t-subset indices
    block_covers = []
    sub_to_blocks = [[] for _ in range(n_t)]

    for i, block in enumerate(all_blocks):
        covers = []
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            covers.append(idx)
            sub_to_blocks[idx].append(i)
        block_covers.append(np.array(covers, dtype=np.int32))

    if not _deadline_ok():
        return []

    # Greedy initialization
    uncovered_mask = np.ones(n_t, dtype=bool)
    selected = set()

    while np.any(uncovered_mask) and _deadline_ok():
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        target_idx = random.choice(uncovered_indices)
        candidates = sub_to_blocks[target_idx]
        if not candidates:
            break
        best_cand = -1
        max_gain = 0
        for b in candidates:
            gain = np.sum(uncovered_mask[block_covers[b]])
            if gain > max_gain:
                max_gain = gain
                best_cand = b
        if best_cand != -1:
            selected.add(best_cand)
            uncovered_mask[block_covers[best_cand]] = False

    if not selected:
        return []

    # Local search with swap
    counts = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        counts[block_covers[b]] += 1

    def is_redundant(b, counts):
        return np.all(counts[block_covers[b]] > 1)

    def try_remove(b, selected, counts):
        """Try to remove block b if it's redundant."""
        if is_redundant(b, counts):
            selected.remove(b)
            counts[block_covers[b]] -= 1
            return True
        return False

    def try_swap(b, selected, counts):
        """Try to swap block b with another block that allows removing b."""
        covers_b = block_covers[b]
        critical = covers_b[counts[covers_b] == 1]

        if len(critical) == 0:
            # No critical subsets, just remove
            selected.remove(b)
            counts[covers_b] -= 1
            return True

        if len(critical) == 1:
            candidates = sub_to_blocks[critical[0]]
        else:
            # Find the most constrained critical subset
            min_count = float('inf')
            target = -1
            for s in critical:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target = s
            candidates = set(sub_to_blocks[target])
            for s in critical:
                if s == target:
                    continue
                candidates.intersection_update(sub_to_blocks[s])
                if not candidates:
                    return False
            candidates = list(candidates)

        valid_candidates = [c for c in candidates if c not in selected]
        if not valid_candidates:
            return False

        # Pick the candidate that covers the most additional uncovered subsets
        best_new = -1
        max_extra = -1
        for c in valid_candidates:
            extra = np.sum(counts[block_covers[c]] == 0)
            if extra > max_extra:
                max_extra = extra
                best_new = c

        if best_new != -1:
            selected.remove(b)
            counts[covers_b] -= 1
            selected.add(best_new)
            counts[block_covers[best_new]] += 1
            return True
        return False

    best_size = len(selected)
    best_selected = set(selected)
    best_counts = counts.copy()

    # Main local search loop
    while _deadline_ok():
        selected_list = list(selected)
        random.shuffle(selected_list)

        for b in selected_list:
            if not _deadline_ok():
                break
            if b not in selected:
                continue

            # First try simple removal
            if try_remove(b, selected, counts):
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
                    best_counts = counts.copy()
                continue

            # Then try swap
            if try_swap(b, selected, counts):
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = set(selected)
                    best_counts = counts.copy()

    # Final result
    if len(best_selected) < len(selected):
        selected = best_selected

    result = [list(all_blocks[i]) for i in sorted(selected)]
    return result
