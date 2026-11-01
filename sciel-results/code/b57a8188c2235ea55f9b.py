"""
Optimized covering design C(v,k,t) using bitset-based greedy set cover
with local improvement and deadline guard.

Strategy:
- Represent each t-subset by an index, and each k-subset by a bitset mask
  of which t-subsets it covers.
- Use a greedy set-cover loop that picks the k-subset with maximum gain
  (number of newly covered t-subsets), computed via bit operations.
- After the greedy phase, perform a local-improvement pass: try to remove
  redundant blocks and, if time permits, try to swap blocks for better ones.
- Enforce a strict deadline to stay within the 120 s evaluation window.
"""

import itertools
import time


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120 s limit

    def time_ok():
        return time.time() < deadline

    # ------------------------------------------------------------------
    # 1. Generate all t-subsets and assign them indices
    # ------------------------------------------------------------------
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # ------------------------------------------------------------------
    # 2. Generate all k-subsets and precompute their coverage masks
    # ------------------------------------------------------------------
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # ksub_masks[j] is a Python int whose bit i is 1 iff k-subset j
    # contains t-subset i.
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    # ------------------------------------------------------------------
    # 3. Greedy set-cover phase
    # ------------------------------------------------------------------
    uncovered_mask = (1 << n_tsubs) - 1
    remaining = n_tsubs
    selected_indices = []  # indices into all_ksubs
    used = [False] * n_ksubs

    while remaining > 0 and time_ok():
        best_j = -1
        best_gain = 0

        # Scan all candidate k-subsets for maximum gain
        for j in range(n_ksubs):
            if used[j]:
                continue
            common = uncovered_mask & ksub_masks[j]
            gain = common.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1 or best_gain == 0:
            # Should not happen if masks are correct and remaining > 0,
            # but guard against infinite loops.
            break

        selected_indices.append(best_j)
        used[best_j] = True
        uncovered_mask &= ~ksub_masks[best_j]
        remaining -= best_gain

    # ------------------------------------------------------------------
    # 4. Local improvement: remove redundant blocks
    # ------------------------------------------------------------------
    # A block is redundant if every t-subset it covers is also covered by
    # at least one other selected block. We check this by maintaining a
    # count of how many selected blocks cover each t-subset.
    if time_ok() and len(selected_indices) > 1:
        # Build coverage counts
        cover_count = [0] * n_tsubs
        for j in selected_indices:
            mask = ksub_masks[j]
            # Iterate over set bits in mask
            m = mask
            while m:
                low = m & (-m)
                idx = low.bit_length() - 1
                cover_count[idx] += 1
                m ^= low

        # Try to remove blocks in reverse order (later blocks may be less essential)
        for pos in range(len(selected_indices) - 1, -1, -1):
            if not time_ok():
                break
            j = selected_indices[pos]
            mask = ksub_masks[j]
            # Check if any t-subset in this block has cover_count == 1
            # If so, the block is essential; otherwise it's redundant.
            essential = False
            m = mask
            while m:
                low = m & (-m)
                idx = low.bit_length() - 1
                if cover_count[idx] == 1:
                    essential = True
                    break
                m ^= low

            if not essential:
                # Remove this block
                selected_indices.pop(pos)
                # Decrement cover counts
                m = mask
                while m:
                    low = m & (-m)
                    idx = low.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= low

    # ------------------------------------------------------------------
    # 5. If there are still uncovered t-subsets (shouldn't happen if
    #    greedy completed), add blocks to cover them.
    # ------------------------------------------------------------------
    if remaining > 0 and time_ok():
        # Rebuild uncovered mask from scratch to be safe
        covered_mask = 0
        for j in selected_indices:
            covered_mask |= ksub_masks[j]
        uncovered_mask = ((1 << n_tsubs) - 1) & ~covered_mask
        remaining = uncovered_mask.bit_count()

        while remaining > 0 and time_ok():
            best_j = -1
            best_gain = 0
            for j in range(n_ksubs):
                common = uncovered_mask & ksub_masks[j]
                gain = common.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j == -1 or best_gain == 0:
                break
            selected_indices.append(best_j)
            uncovered_mask &= ~ksub_masks[best_j]
            remaining -= best_gain

    # ------------------------------------------------------------------
    # 6. Build the final list of blocks (as lists of ints)
    # ------------------------------------------------------------------
    blocks = [list(all_ksubs[j]) for j in selected_indices]

    # Ensure distinctness (should already be distinct since we track used)
    # but just in case, deduplicate.
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return unique_blocks
