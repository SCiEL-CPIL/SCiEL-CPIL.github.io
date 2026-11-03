import itertools
import time
import numpy as np
import random
import sys
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # block_to_tsets: list of lists of tset indices
    block_to_tsets = [[] for _ in range(num_blocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]

    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    # Convert block_to_tsets to numpy arrays for faster vectorized ops where possible
    # But lists are faster for iteration in Python. We'll keep lists.

    def get_solution(indices):
        return [list(all_blocks[i]) for i in indices]

    def run_local_search(initial_indices, max_time_per_run=10.0, rng=None):
        """
        Takes a list of block indices, applies local search improvements.
        Returns the best list of indices found.
        """
        if rng is None:
            rng = random.Random()

        current = list(initial_indices)
        if not current:
            return []

        # Calculate initial cover counts
        cover_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in current:
            for ts_idx in block_to_tsets[b_idx]:
                cover_counts[ts_idx] += 1

        best_size = len(current)
        best_solution = list(current)

        improved = True
        local_start = time.time()

        while improved and (time.time() - local_start) < max_time_per_run:
            improved = False

            # Phase 1: Remove redundant blocks (multiple passes until stable)
            while True:
                found_redundant = False
                for i in range(len(current) - 1, -1, -1):
                    if (time.time() - local_start) > max_time_per_run:
                        break
                    b_idx = current[i]
                    is_redundant = True
                    for ts_idx in block_to_tsets[b_idx]:
                        if cover_counts[ts_idx] == 1:
                            is_redundant = False
                            break

                    if is_redundant:
                        for ts_idx in block_to_tsets[b_idx]:
                            cover_counts[ts_idx] -= 1
                        current.pop(i)
                        found_redundant = True
                        improved = True
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)

                if not found_redundant or (time.time() - local_start) > max_time_per_run:
                    break

            if not current:
                break

            # Phase 2: 1-for-1 Swaps
            # We try to find a block to swap out and a block to swap in
            # such that the total number of blocks decreases or stays same but
            # coverage improves (though we only care about size here).
            # Actually, a simple 1-for-1 swap doesn't change size.
            # To reduce size, we need to find a block that is "almost" redundant,
            # or use a 2-for-1 swap, or just rely on the redundancy pass after swaps.
            # The current strategy: Swap b_out with b_in if b_in covers all unique tsets of b_out.
            # This keeps size constant but might allow further removals in Phase 1.

            # Strategy: Shuffle and try swaps
            # To avoid getting stuck, we can also try random perturbations (ILS)
            # Let's add a perturbation step occasionally.

            if rng.random() < 0.1:
                # Perturbation: Remove a random block and repair
                # This is risky, might break coverage.
                # Better: Just shuffle the order of trying swaps.
                pass

            random.shuffle(current)

            for i in range(len(current)):
                if (time.time() - local_start) > max_time_per_run:
                    break

                b_out = current[i]

                # Find tsets uniquely covered by b_out
                unique_tsets = []
                for ts_idx in block_to_tsets[b_out]:
                    if cover_counts[ts_idx] == 1:
                        unique_tsets.append(ts_idx)

                if not unique_tsets:
                    # This block is redundant, but Phase 1 should have caught it.
                    # Maybe it became redundant after a previous swap in this loop?
                    # If so, we can just remove it.
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    current.pop(i)
                    improved = True
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
                    break  # Restart the inner loop as indices changed

                # Find candidate blocks B_in that cover all unique_tsets
                candidates = set(tset_to_blocks[unique_tsets[0]])
                candidates.discard(b_out)

                for ts_idx in unique_tsets[1:]:
                    if not candidates:
                        break
                    candidates.intersection_update(tset_to_blocks[ts_idx])

                if not candidates:
                    continue

                best_candidate = -1
                best_score = -1
                b_out_tsets_set = set(block_to_tsets[b_out])

                candidate_list = list(candidates)
                # Limit candidates to speed up
                if len(candidate_list) > 100:
                    random.shuffle(candidate_list)
                    candidate_list = candidate_list[:100]

                for b_in in candidate_list:
                    # Score: How many new tsets does b_in cover?
                    # We want to maximize the "gain" from swapping in b_in.
                    # Gain = number of tsets covered by b_in but not by b_out (and currently covered by others).
                    # Actually, if we swap, the tsets covered by b_out but not b_in must be covered by others.
                    # This is guaranteed by the "unique_tsets" check? No.
                    # unique_tsets are covered ONLY by b_out.
                    # If b_in covers unique_tsets, then after swap, they are still covered.
                    # What about tsets covered by b_out but NOT in unique_tsets?
                    # They are covered by at least one other block. So they remain covered.
                    # So any b_in covering unique_tsets is a valid swap.
                    # To improve future removals, we want b_in to be "less overlapping" with the rest,
                    # or to cover more tsets that are currently covered only once (by other blocks).

                    # Simple score: number of tsets in b_in that are currently covered exactly once.
                    # If we swap in b_in, those tsets will be covered twice.
                    # This might not help removal.
                    # Better score: number of tsets in b_in that are NOT in b_out.
                    # This maximizes the "new" coverage, potentially making other blocks redundant.

                    score = 0
                    for ts_idx in block_to_tsets[b_in]:
                        if ts_idx not in b_out_tsets_set:
                            score += 1

                    if score > best_score:
                        best_score = score
                        best_candidate = b_in

                if best_candidate != -1:
                    for ts_idx in block_to_tsets[b_out]:
                        cover_counts[ts_idx] -= 1
                    for ts_idx in block_to_tsets[best_candidate]:
                        cover_counts[ts_idx] += 1

                    current[i] = best_candidate
                    improved = True

            # If no improvement in this pass, we might be stuck.
            # We can try a random restart or perturbation here.
            # For now, we just exit the while loop if no improvement was found in the whole pass.
            # But `improved` is set to True if ANY swap or removal happened.
            # So if no swap/removal happened, `improved` remains False and we exit.

        return best_solution

    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = np.ones(num_tsets, dtype=np.bool_)
        uncovered_count = num_tsets
        selected = []

        # Weights: 1 / frequency of tset
        tset_counts = np.zeros(num_tsets, dtype=np.int32)
        for b_idx in range(num_blocks):
            for ts_idx in block_to_tsets[b_idx]:
                tset_counts[ts_idx] += 1
        safe_counts = np.maximum(tset_counts, 1).astype(np.float64)
        tset_weights = 1.0 / safe_counts

        # Initial gains
        gains = np.zeros(num_blocks, dtype=np.float64)
        for b_idx in range(num_blocks):
            if block_to_tsets[b_idx]:
                gains[b_idx] = np.sum(tset_weights[block_to_tsets[b_idx]])

        selected_mask = np.zeros(num_blocks, dtype=np.bool_)

        while uncovered_count > 0 and time.time() < deadline - 15:
            # Find best block
            best_idx = -1
            best_gain = -1.0

            for i in range(num_blocks):
                if not selected_mask[i] and gains[i] > best_gain:
                    best_gain = gains[i]
                    best_idx = i

            if best_idx == -1 or best_gain <= 0:
                break

            selected.append(best_idx)
            selected_mask[best_idx] = True

            # Update
            for ts_idx in block_to_tsets[best_idx]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = False
                    uncovered_count -= 1
                    w = tset_weights[ts_idx]
                    for b in tset_to_blocks[ts_idx]:
                        if not selected_mask[b]:
                            gains[b] -= w

        return selected

    best_global_solution = None
    best_global_size = float('inf')

    # Strategy: Run multiple greedy+local search instances with different seeds
    # and time limits.

    # Run 1: Standard greedy with extended local search
    t_start = time.time()
    init1 = greedy_cover(42)
    if init1:
        # Use more time for local search since it's the primary run
        ls_time = min(45.0, deadline - t_start - 10)
        sol1 = run_local_search(init1, max_time_per_run=ls_time)
        if sol1 and len(sol1) < best_global_size:
            best_global_size = len(sol1)
            best_global_solution = sol1

    # Run 2: Greedy with different seed if time permits
    t_start = time.time()
    if time.time() < deadline - 35:
        init2 = greedy_cover(123)
        if init2:
            ls_time = min(25.0, deadline - t_start - 10)
            sol2 = run_local_search(init2, max_time_per_run=ls_time)
            if sol2 and len(sol2) < best_global_size:
                best_global_size = len(sol2)
                best_global_solution = sol2

    # Run 3: Another greedy with different seed if time permits
    t_start = time.time()
    if time.time() < deadline - 20:
        init3 = greedy_cover(777)
        if init3:
            ls_time = min(15.0, deadline - t_start - 5)
            sol3 = run_local_search(init3, max_time_per_run=ls_time)
            if sol3 and len(sol3) < best_global_size:
                best_global_size = len(sol3)
                best_global_solution = sol3

    # Final polish: Re-run local search on the best solution with remaining time
    if best_global_solution and time.time() < deadline - 5:
        remaining_time = deadline - time.time() - 2
        if remaining_time > 5:
            improved_sol = run_local_search(best_global_solution, max_time_per_run=remaining_time)
            if improved_sol and len(improved_sol) < best_global_size:
                best_global_size = len(improved_sol)
                best_global_solution = improved_sol

    if not best_global_solution:
        # Fallback: Return a valid but potentially large solution
        if init1:
            return get_solution(init1)
        return []

    return get_solution(best_global_solution)

if __name__ == "__main__":
    # Test with the specific parameters
    v, k, t = 19, 8, 5
    res = construct(v, k, t)
    print(f"Number of blocks: {len(res)}")
    # Verify
    covered = set()
    for block in res:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    total_tsets = len(list(itertools.combinations(range(v), t)))
    print(f"Covered: {len(covered)}/{total_tsets}")
