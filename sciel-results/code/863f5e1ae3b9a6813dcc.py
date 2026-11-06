import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    points = list(range(12))
    all_5 = list(combinations(points, 5))
    n5 = len(all_5)
    s5_idx = {s: i for i, s in enumerate(all_5)}

    all_6 = list(combinations(points, 6))
    n6 = len(all_6)

    # Precompute for each 5-subset which 6-subsets cover it
    covering = [[] for _ in range(n5)]
    for bi, b in enumerate(all_6):
        for s in combinations(b, 5):
            covering[s5_idx[s]].append(bi)

    # Symmetry breaking: fix first block to {0,1,2,3,4,5}
    first_block = (0, 1, 2, 3, 4, 5)
    first_idx = all_6.index(first_block)

    # Mark the 5-subsets in the first block as covered
    used = [False] * n5
    for s in combinations(first_block, 5):
        used[s5_idx[s]] = True

    selected = [first_block]

    def solve(remaining, depth):
        if time.time() > deadline:
            return False
        if remaining == 0:
            return True
        if len(selected) >= 132:
            return False  # can't exceed 132

        # MRV: find the uncovered 5-subset with fewest available blocks
        best_s = -1
        best_cands = None
        best_count = float('inf')

        # Iterate over uncovered 5-subsets
        for si in range(n5):
            if used[si]:
                continue
            cands = [bi for bi in covering[si]
                     if not any(used[s5_idx[s]] for s in combinations(all_6[bi], 5)
                               if s5_idx[s] != si)]
            # More efficient: check all 6 5-subsets of the block are unused
            cands = []
            for bi in covering[si]:
                block = all_6[bi]
                ok = True
                for s in combinations(block, 5):
                    if used[s5_idx[s]]:
                        ok = False
                        break
                if ok:
                    cands.append(bi)

            cnt = len(cands)
            if cnt == 0:
                return False
            if cnt < best_count:
                best_count = cnt
                best_s = si
                best_cands = cands
                if cnt == 1:
                    break

        for bi in best_cands:
            block = all_6[bi]
            # Mark
            for s in combinations(block, 5):
                used[s5_idx[s]] = True
            selected.append(block)

            # Update remaining count
            old_remaining = remaining
            if solve(old_remaining - 6, depth + 1):
                return True

            # Unmark
            for s in combinations(block, 5):
                used[s5_idx[s]] = False
            selected.pop()

        return False

    # Count uncovered 5-subsets initially
    remaining = sum(1 for u in used if not u)

    solve(remaining, 0)

    if len(selected) == 132:
        return selected

    # Fallback: return what we have (shouldn't happen for a valid solver)
    return selected
