import itertools
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Generate all t-subsets and index them
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}

    # Generate all k-blocks and compute which t-subsets each covers
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)

    # For each k-block, compute the set of t-subset indices it covers
    block_t_subsets = []
    for block in k_blocks:
        covered = set()
        for combo in itertools.combinations(block, t):
            covered.add(t_index[combo])
        block_t_subsets.append(covered)

    # Precompute as sorted lists for faster operations
    block_t_lists = [sorted(s) for s in block_t_subsets]

    # Build reverse mapping: for each t-subset, which blocks cover it
    t_to_blocks = [[] for _ in range(num_t)]
    for i, covered in enumerate(block_t_subsets):
        for idx in covered:
            t_to_blocks[idx].append(i)

    # Compute frequency of each t-subset (how many blocks cover it)
    t_freq = np.zeros(num_t, dtype=np.int32)
    for i in range(num_k):
        for idx in block_t_subsets[i]:
            t_freq[idx] += 1

    def greedy_complete(selected_set, cover_count):
        """Complete coverage using greedy if any t-subset is uncovered."""
        uncovered = [i for i in range(num_t) if cover_count[i] == 0]
        if not uncovered:
            return list(selected_set)
        uncovered_set = set(uncovered)
        result = list(selected_set)
        while uncovered_set and elapsed() < deadline - 5:
            best_idx = -1
            best_gain = -1
            for i in range(num_k):
                if i in selected_set:
                    continue
                gain = 0
                for idx in block_t_subsets[i]:
                    if idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            result.append(best_idx)
            selected_set.add(best_idx)
            for idx in block_t_subsets[best_idx]:
                if idx in uncovered_set:
                    uncovered_set.discard(idx)
                    cover_count[idx] += 1
                else:
                    cover_count[idx] += 1
        return result

    def remove_redundant(selected, cover_count):
        """Iteratively remove redundant blocks."""
        changed = True
        while changed and elapsed() < deadline - 3:
            changed = False
            new_selected = []
            for i in selected:
                needed = False
                for idx in block_t_subsets[i]:
                    if cover_count[idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(i)
                else:
                    changed = True
                    for idx in block_t_subsets[i]:
                        cover_count[idx] -= 1
            if changed:
                selected = new_selected
        return selected

    def local_search(selected, cover_count, time_limit):
        """1-for-1 swap local search."""
        selected_set = set(selected)
        for _ in range(3):
            if elapsed() > deadline - time_limit:
                break
            improved = False
            for i in list(selected):
                if elapsed() > deadline - time_limit:
                    break
                unique_to_i = [idx for idx in block_t_subsets[i] if cover_count[idx] == 1]
                if not unique_to_i:
                    continue
                best_swap = -1
                best_new_cover = 0
                unique_set = set(unique_to_i)
                for j in range(num_k):
                    if j in selected_set:
                        continue
                    # Check if j covers all unique_to_i
                    covers_all = True
                    for idx in unique_to_i:
                        if idx not in block_t_subsets[j]:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    # Count new coverage
                    new_cover = 0
                    for idx in block_t_subsets[j]:
                        if cover_count[idx] == 0:
                            new_cover += 1
                    if new_cover > best_new_cover:
                        best_new_cover = new_cover
                        best_swap = j
                if best_swap >= 0 and best_new_cover > 0:
                    selected.remove(i)
                    selected.append(best_swap)
                    selected_set.discard(i)
                    selected_set.add(best_swap)
                    for idx in block_t_subsets[i]:
                        cover_count[idx] -= 1
                    for idx in block_t_subsets[best_swap]:
                        cover_count[idx] += 1
                    improved = True
            if not improved:
                break
        return selected

    # === PARAMETRIC LAGRANGIAN CAPACITY SCALING ===
    # The Lagrangian relaxation: minimize lambda * |S| subject to cover constraints.
    # For a given lambda, a block i is "profitable" if its marginal coverage
    # (weighted by 1/freq) exceeds lambda. We use binary search on lambda.
    #
    # Structural change: instead of fixed greedy, we tune the block-cost penalty
    # to find the threshold where coverage just saturates, giving a tighter initial
    # solution that local search can then refine.

    # Compute per-t-subset weight (inverse frequency - rare subsets matter more)
    safe_freq = np.maximum(t_freq, 1).astype(np.float64)
    t_weights = 1.0 / safe_freq

    # Precompute block "value" = sum of weights of t-subsets it covers
    block_value = np.zeros(num_k, dtype=np.float64)
    for i in range(num_k):
        if block_t_subsets[i]:
            block_value[i] = np.sum(t_weights[list(block_t_subsets[i])])

    # Parametric search: find lambda such that the greedy with threshold lambda
    # produces a valid cover. Lower lambda = more blocks, higher = fewer.
    # We binary search on lambda to find the tightest valid cover.

    def greedy_with_penalty(lambda_val):
        """Greedy that only picks blocks with gain > lambda_val (penalized)."""
        selected = []
        selected_mask = np.zeros(num_k, dtype=np.bool_)
        uncovered = np.ones(num_t, dtype=np.bool_)
        uncovered_count = num_t
        # Current gain for each block (sum of weights of uncovered t-subsets it covers)
        current_gain = block_value.copy()

        while uncovered_count > 0 and elapsed() < deadline - 20:
            # Find best block with gain > lambda_val
            best_idx = -1
            best_gain = lambda_val  # Must exceed this
            for i in range(num_k):
                if selected_mask[i]:
                    continue
                if current_gain[i] > best_gain:
                    best_gain = current_gain[i]
                    best_idx = i

            if best_idx == -1:
                # No block exceeds threshold; pick the best available (fallback)
                best_idx = -1
                best_gain = -1.0
                for i in range(num_k):
                    if selected_mask[i]:
                        continue
                    if current_gain[i] > best_gain:
                        best_gain = current_gain[i]
                        best_idx = i
                if best_idx == -1 or best_gain <= 0:
                    break

            selected.append(best_idx)
            selected_mask[best_idx] = True

            for idx in block_t_subsets[best_idx]:
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    w = t_weights[idx]
                    for b in t_to_blocks[idx]:
                        if not selected_mask[b]:
                            current_gain[b] -= w

        return selected

    # Binary search on lambda
    # lambda=0 gives standard weighted greedy (many blocks)
    # lambda=inf gives empty (invalid)
    # We want the smallest valid cover

    # First, get a baseline with lambda=0 (standard weighted greedy)
    base_selected = greedy_with_penalty(0.0)
    if not base_selected:
        # Fallback to simple greedy
        base_selected = list(range(num_k))  # worst case
    base_count = len(base_selected)

    # Build cover count for baseline
    cover_count = np.zeros(num_t, dtype=np.int32)
    base_set = set(base_selected)
    for i in base_selected:
        for idx in block_t_subsets[i]:
            cover_count[idx] += 1

    # Remove redundant from baseline
    base_selected = remove_redundant(base_selected, cover_count)
    base_set = set(base_selected)
    base_count = len(base_selected)

    # Now try increasing lambda to see if we get a smaller valid cover
    # The idea: higher lambda prunes blocks with low marginal gain,
    # forcing the algorithm to be more selective.

    best_solution = base_selected[:]
    best_cover = cover_count.copy()
    best_len = base_count

    # Try a range of lambda values
    # Block values range from ~0 to ~max_block_value
    max_val = np.max(block_value) if num_k > 0 else 1.0
    min_val = np.min(block_value) if num_k > 0 else 0.0

    # Sample lambda values logarithmically
    if max_val > 0:
        lambda_candidates = np.geomspace(max_val * 0.01, max_val * 0.9, num=12)
    else:
        lambda_candidates = []

    for lam in lambda_candidates:
        if elapsed() > deadline - 40:
            break
        sel = greedy_with_penalty(float(lam))
        if not sel:
            continue

        # Check validity
        cc = np.zeros(num_t, dtype=np.int32)
        sel_set = set(sel)
        for i in sel:
            for idx in block_t_subsets[i]:
                cc[idx] += 1

        if np.all(cc > 0):
            # Valid - remove redundant
            sel = remove_redundant(sel, cc)
            sel_set = set(sel)
            if len(sel) < best_len:
                best_len = len(sel)
                best_solution = sel[:]
                best_cover = cc.copy()
        else:
            # Incomplete - complete it
            sel = greedy_complete(sel_set, cc)
            sel_set = set(sel)
            # Rebuild cover count
            cc = np.zeros(num_t, dtype=np.int32)
            for i in sel:
                for idx in block_t_subsets[i]:
                    cc[idx] += 1
            if np.all(cc > 0):
                sel = remove_redundant(sel, cc)
                if len(sel) < best_len:
                    best_len = len(sel)
                    best_solution = sel[:]
                    best_cover = cc.copy()

    # Also try: start from baseline, then do targeted removals guided by
    # the Lagrangian multipliers (blocks with lowest marginal contribution)
    if elapsed() < deadline - 20:
        # Rebuild cover count for best_solution
        cc = np.zeros(num_t, dtype=np.int32)
        for i in best_solution:
            for idx in block_t_subsets[i]:
                cc[idx] += 1

        # Compute marginal contribution of each selected block
        # = number of t-subsets it uniquely covers
        for i in best_solution[:]:
            if elapsed() > deadline - 10:
                break
            unique_count = 0
            for idx in block_t_subsets[i]:
                if cc[idx] == 1:
                    unique_count += 1
            if unique_count == 0:
                # Try removing
                for idx in block_t_subsets[i]:
                    cc[idx] -= 1
                best_solution.remove(i)

        # Local search on the best solution
        if elapsed() < deadline - 5:
            cc = np.zeros(num_t, dtype=np.int32)
            for i in best_solution:
                for idx in block_t_subsets[i]:
                    cc[idx] += 1
            best_solution = local_search(best_solution, cc, 5)

    # Final safety: ensure completeness
    if elapsed() < deadline - 2:
        cc_final = np.zeros(num_t, dtype=np.int32)
        sel_set = set(best_solution)
        for i in best_solution:
            for idx in block_t_subsets[i]:
                cc_final[idx] += 1
        if not np.all(cc_final > 0):
            best_solution = greedy_complete(sel_set, cc_final)

    result = [list(k_blocks[i]) for i in best_solution]
    return result
