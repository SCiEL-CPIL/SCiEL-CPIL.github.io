import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit to be safe

    points = range(12)

    # Precompute all 5-subsets as bitmasks
    all_5_subsets = []
    for s in combinations(points, 5):
        mask = 0
        for p in s:
            mask |= (1 << p)
        all_5_subsets.append(mask)

    # Generate all 924 6-subsets
    all_6_subsets = []
    for b in combinations(points, 6):
        mask = 0
        for p in b:
            mask |= (1 << p)
        all_6_subsets.append(mask)

    # Precompute: for each 6-subset mask, the list of its 5-subset masks
    block_5s = {}
    for b_mask in all_6_subsets:
        pts = [i for i in range(12) if (b_mask >> i) & 1]
        masks = []
        for i in range(6):
            sub_mask = b_mask & ~(1 << pts[i])
            masks.append(sub_mask)
        block_5s[b_mask] = masks

    # Precompute: for each 5-subset mask, the list of 6-subset masks that contain it
    blocks_by_5 = {mask: [] for mask in all_5_subsets}
    for b_mask in all_6_subsets:
        for s in block_5s[b_mask]:
            blocks_by_5[s].append(b_mask)

    solution = []
    covered = set()

    def backtrack(depth):
        if time.time() > deadline:
            return False

        if depth == 132:
            return True

        # Find the uncovered 5-subset with the fewest available covering blocks (MRV)
        best_5_mask = None
        best_candidates = None
        min_candidates = float('inf')

        for mask in all_5_subsets:
            if mask in covered:
                continue

            candidates = []
            for b_mask in blocks_by_5[mask]:
                if any(s in covered for s in block_5s[b_mask]):
                    continue
                candidates.append(b_mask)

            if len(candidates) < min_candidates:
                min_candidates = len(candidates)
                best_5_mask = mask
                best_candidates = candidates

                if min_candidates == 0:
                    return False
                if min_candidates == 1:
                    break

        if best_candidates is None or len(best_candidates) == 0:
            return False

        for b_mask in best_candidates:
            solution.append(b_mask)
            new_covers = block_5s[b_mask]
            for s in new_covers:
                covered.add(s)

            if backtrack(depth + 1):
                return True

            solution.pop()
            for s in new_covers:
                covered.remove(s)

        return False

    if backtrack(0):
        result = []
        for mask in solution:
            result.append([i for i in range(12) if (mask >> i) & 1])
        return result
    else:
        return solution
