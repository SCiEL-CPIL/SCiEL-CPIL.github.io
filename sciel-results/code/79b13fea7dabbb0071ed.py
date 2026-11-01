"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if n == 1:
        return "1"

    # Step 1: Build the classic superpermutation for n-1
    def build_superperm_recursive(n_val):
        if n_val == 1:
            return [1]
        word = [1]
        for m in range(2, n_val + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)

            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    base_word = build_superperm_recursive(n - 1)

    # Step 2: Extract all unique permutations of 1..n-1 from base_word
    perms, seen = [], set()
    for i in range(len(base_word) - (n - 2)):
        window = tuple(base_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)

    m = len(perms)

    # Step 3: Create blocks for adding symbol n
    blocks = [list(p) + [n] + list(p) for p in perms]
    block_len = 2 * (n - 1) + 1

    # Step 4: Precompute overlap matrix
    overlap_mat = [[0] * m for _ in range(m)]
    block_strs = ["".join(map(str, b)) for b in blocks]
    for i in range(m):
        bi_str = block_strs[i]
        for j in range(m):
            bj_str = block_strs[j]
            max_ov = block_len - 1
            ov = 0
            for o in range(max_ov, -1, -1):
                if bi_str[-o:] == bj_str[:o]:
                    ov = o
                    break
            overlap_mat[i][j] = ov

    # Overlap with end of base_word
    base_str = "".join(map(str, base_word))
    first_ov = [0] * m
    for i in range(m):
        bi_str = block_strs[i]
        max_ov = min(len(base_str), block_len - 1)
        for o in range(max_ov, -1, -1):
            if base_str[-o:] == bi_str[:o]:
                first_ov[i] = o
                break

    # Step 5: Local Search to maximize total overlap (2-opt)
    order = list(range(m))

    def calc_total_overlap(ord):
        total = first_ov[ord[0]]
        for k in range(1, len(ord)):
            total += overlap_mat[ord[k - 1]][ord[k]]
        return total

    current_overlap = calc_total_overlap(order)

    improved = True
    while improved and (time.time() < deadline - 10):
        improved = False
        for i in range(m - 1):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, m):
                idx_a = i - 1
                idx_b = i
                idx_c = j
                idx_d = j + 1

                old_ov = 0
                new_ov = 0

                if idx_a >= 0:
                    old_ov += overlap_mat[order[idx_a]][order[idx_b]]
                    new_ov += overlap_mat[order[idx_a]][order[idx_c]]
                else:
                    old_ov += first_ov[order[idx_b]]
                    new_ov += first_ov[order[idx_c]]

                if i < j:
                    for k in range(i, j):
                        old_ov += overlap_mat[order[k]][order[k + 1]]
                        new_ov += overlap_mat[order[k + 1]][order[k]]

                if idx_d < m:
                    old_ov += overlap_mat[order[idx_c]][order[idx_d]]
                    new_ov += overlap_mat[order[idx_b]][order[idx_d]]

                if new_ov > old_ov:
                    order[i:j + 1] = reversed(order[i:j + 1])
                    current_overlap = new_ov + (current_overlap - old_ov)
                    improved = True

    # Step 6: Construct the final word
    final_word = list(base_word)
    first_block_idx = order[0]
    first_block = blocks[first_block_idx]
    ov = first_ov[first_block_idx]
    final_word += first_block[ov:]

    for k in range(1, m):
        prev_idx = order[k - 1]
        curr_idx = order[k]
        curr_block = blocks[curr_idx]
        ov = overlap_mat[prev_idx][curr_idx]
        final_word += curr_block[ov:]

    # Step 7: Efficient greedy deletion with incremental validity tracking
    # Key insight: instead of re-checking all 5040 permutations each time,
    # maintain a count of covered permutations and only update affected windows.

    # Precompute all target permutations
    all_perms = set(itertools.permutations(range(1, n + 1)))
    n_perms = len(all_perms)  # 5040 for n=7

    # Build initial coverage: for each position i, the window starting at i
    # Map each permutation to a count of how many times it appears
    perm_count = {}
    for i in range(len(final_word) - n + 1):
        w = tuple(final_word[i:i + n])
        if w in all_perms:
            perm_count[w] = perm_count.get(w, 0) + 1

    covered_count = sum(1 for w in all_perms if perm_count.get(w, 0) > 0)

    def is_valid_fast():
        return covered_count == n_perms

    if not is_valid_fast():
        return "".join(map(str, final_word))

    # Greedy deletion: try removing each position, check if validity is preserved
    # Process positions from end to start so indices remain valid
    changed = True
    # Use multiple passes with different random orderings for better coverage
    while changed and (time.time() < deadline - 3):
        changed = False
        # Try positions in reverse order (safe for index stability)
        for i in range(len(final_word) - 1, -1, -1):
            if time.time() > deadline - 3:
                break

            # Determine which windows are affected by removing position i
            # Windows that include position i: start positions from max(0, i-n+1) to min(i, len(final_word)-n)
            w_start_min = max(0, i - n + 1)
            w_start_max = min(i, len(final_word) - n)

            # Check if any window that becomes invalid was the only occurrence
            can_remove = True
            for ws in range(w_start_min, w_start_max + 1):
                w = tuple(final_word[ws:ws + n])
                if w in all_perms:
                    if perm_count.get(w, 0) == 1:
                        can_remove = False
                        break

            if can_remove:
                # Perform removal
                removed = final_word[i]
                # Update perm_count for affected windows
                for ws in range(w_start_min, w_start_max + 1):
                    w = tuple(final_word[ws:ws + n])
                    if w in all_perms:
                        perm_count[w] -= 1
                        if perm_count[w] == 0:
                            covered_count -= 1

                # After removal, new windows may form (windows that shift)
                # Actually, when we remove position i, windows that were at positions
                # > i shift left by 1. We need to check if new valid permutations appear.
                # The windows that change are those that cross position i.
                # After removal, the window at position ws (ws < i) now covers
                # old positions ws..ws+n-1 but with position i removed.
                # This is complex. Let's just rebuild the affected region.

                # Simpler: just rebuild perm_count for the affected region
                # Affected region: positions from max(0, i-n+1) to min(i, len(final_word)-n-1) after removal
                # Actually, let's just do a targeted rebuild around position i
                final_word = final_word[:i] + final_word[i + 1:]

                # Rebuild perm_count for windows in the affected zone
                # Windows affected: those that include the gap at position i
                # After removal, windows starting from max(0, i-n+1) to i (new indexing)
                # But we need to be careful. Let's rebuild all windows in a window of size n around i.
                rebuild_lo = max(0, i - n + 1)
                rebuild_hi = min(i, len(final_word) - n)

                # Subtract old counts for windows in this range
                # We already subtracted above, but the indices shifted.
                # This is getting complicated. Let's use a different approach:
                # Just do a full rebuild of perm_count if the word is short enough,
                # or use a smarter incremental approach.

                # For n=7 and word length ~6000, a full rebuild is 6000*7 = 42000 ops,
                # which is fast. Let's just rebuild when needed.
                perm_count_new = {}
                for idx in range(len(final_word) - n + 1):
                    w = tuple(final_word[idx:idx + n])
                    if w in all_perms:
                        perm_count_new[w] = perm_count_new.get(w, 0) + 1
                perm_count = perm_count_new
                covered_count = sum(1 for w in all_perms if perm_count.get(w, 0) > 0)

                if not is_valid_fast():
                    # Undo the removal
                    final_word = final_word[:i] + [removed] + final_word[i:]
                    perm_count = {}
                    for idx in range(len(final_word) - n + 1):
                        w = tuple(final_word[idx:idx + n])
                        if w in all_perms:
                            perm_count[w] = perm_count.get(w, 0) + 1
                    covered_count = sum(1 for w in all_perms if perm_count.get(w, 0) > 0)
                else:
                    changed = True

    return "".join(map(str, final_word))

# EVOLVE-BLOCK-END
