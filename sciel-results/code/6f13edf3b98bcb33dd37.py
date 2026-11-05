import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    timeout = 142.0  # evaluator kills at 150s, keep 8s margin

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)

    # Precompute block_subsets as lists of subset indices
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)

    # Precompute blocks_covering_subset
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)

    def time_exceeded():
        return time.time() - start_time > timeout

    def run_greedy(randomize_ties=False, seed=None):
        if time_exceeded():
            return None

        if seed is not None:
            random.seed(seed)

        cover_count = np.zeros(N_subsets, dtype=np.int32)
        potential = np.array([len(s) for s in block_subsets], dtype=np.int32)

        selected_blocks = []
        uncovered_count = N_subsets

        while uncovered_count > 0:
            if time_exceeded():
                return None

            max_pot = np.max(potential)
            if max_pot == 0:
                break

            if randomize_ties:
                candidates = np.where(potential == max_pot)[0]
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_idx = np.argmax(potential)

            selected_blocks.append(best_idx)

            for s in block_subsets[best_idx]:
                old_count = cover_count[s]
                cover_count[s] += 1
                if old_count == 0:
                    uncovered_count -= 1
                    for b in blocks_covering_subset[s]:
                        potential[b] -= 1

        return selected_blocks

    def local_search(selected_blocks, max_time=None):
        """Efficient local search using O(1) removal via index tracking."""
        if time_exceeded():
            return selected_blocks

        if max_time is None:
            max_time = timeout - 2

        # Use a list for selected blocks and a set for O(1) membership
        # Track position for O(1) removal
        sel_list = list(selected_blocks)
        sel_set = set(sel_list)
        pos_in_list = {b: i for i, b in enumerate(sel_list)}

        cover_count = np.zeros(N_subsets, dtype=np.int32)
        for b_idx in sel_list:
            for s in block_subsets[b_idx]:
                cover_count[s] += 1

        def remove_block(b_idx):
            """Remove a block from sel_list in O(1)."""
            pos = pos_in_list[b_idx]
            last = sel_list[-1]
            sel_list[pos] = last
            pos_in_list[last] = pos
            sel_list.pop()
            del pos_in_list[b_idx]
            sel_set.discard(b_idx)
            for s in block_subsets[b_idx]:
                cover_count[s] -= 1

        def add_block(b_idx):
            """Add a block to sel_list."""
            pos_in_list[b_idx] = len(sel_list)
            sel_list.append(b_idx)
            sel_set.add(b_idx)
            for s in block_subsets[b_idx]:
                cover_count[s] += 1

        def do_reduction():
            """Remove all redundant blocks (all their subsets covered >= 2 times)."""
            if time_exceeded() or time.time() - start_time > max_time:
                return
            i = 0
            while i < len(sel_list) and not time_exceeded() and time.time() - start_time <= max_time:
                b_idx = sel_list[i]
                # Check if any subset has cover_count == 1
                is_redundant = True
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    remove_block(b_idx)
                    # Don't increment i; the element that moved into this slot needs checking
                else:
                    i += 1

        def do_swap_search(max_rounds=100):
            """Try to replace blocks with better ones."""
            for round_num in range(max_rounds):
                if time_exceeded() or time.time() - start_time > max_time:
                    break

                improved = False
                # Work on a copy of the list order to avoid mutation issues
                order = list(range(len(sel_list)))
                random.shuffle(order)

                for idx in order:
                    if time_exceeded() or time.time() - start_time > max_time:
                        break
                    if idx >= len(sel_list):
                        continue
                    b_idx = sel_list[idx]
                    if b_idx not in sel_set:
                        continue

                    # Find subsets that would become uncovered if we remove b_idx
                    uncovered_subs = []
                    for s in block_subsets[b_idx]:
                        if cover_count[s] == 1:
                            uncovered_subs.append(s)

                    if not uncovered_subs:
                        # Block is redundant
                        remove_block(b_idx)
                        improved = True
                        break

                    # Find candidate blocks that cover ALL uncovered_subs
                    # Use the smallest list for intersection
                    if len(uncovered_subs) <= 6:
                        # Start with the smallest candidate set
                        min_list = min(uncovered_subs, key=lambda s: len(blocks_covering_subset[s]))
                        candidates = set(blocks_covering_subset[min_list])
                        for s in uncovered_subs:
                            if s == min_list:
                                continue
                            candidates &= set(blocks_covering_subset[s])
                        candidates -= sel_set

                        if candidates:
                            # Score candidates by how many NEW subsets they cover
                            best_j = None
                            best_score = -1
                            candidates_list = list(candidates)
                            random.shuffle(candidates_list)
                            # Limit candidates examined for speed
                            for j_idx in candidates_list[:50]:
                                new_subs = 0
                                for s in block_subsets[j_idx]:
                                    if cover_count[s] == 0:
                                        new_subs += 1
                                if new_subs > best_score:
                                    best_score = new_subs
                                    best_j = j_idx

                            if best_j is not None and best_score > 0:
                                # Perform swap: remove b_idx, add best_j
                                remove_block(b_idx)
                                add_block(best_j)
                                improved = True
                                break

                if not improved:
                    break

                do_reduction()

        do_reduction()
        do_swap_search()
        do_reduction()  # Final cleanup pass

        return sel_list

    best_blocks = None
    best_len = float('inf')

    # Deterministic greedy
    res = run_greedy(randomize_ties=False)
    if res:
        res = local_search(res)
        if res:
            best_blocks = res
            best_len = len(res)

    # Randomized restarts with different seeds
    t_elapsed = time.time() - start_time
    if t_elapsed < timeout - 5:
        for i in range(60):
            if time.time() - start_time > timeout - 3:
                break
            res = run_greedy(randomize_ties=True, seed=42 + i)
            if res:
                res = local_search(res)
                if res and len(res) < best_len:
                    best_len = len(res)
                    best_blocks = res

    if best_blocks is None:
        return []

    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    return final_blocks
