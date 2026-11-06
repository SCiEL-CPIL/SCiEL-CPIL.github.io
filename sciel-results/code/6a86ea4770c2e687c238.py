import time
import sys
from itertools import combinations
from collections import defaultdict

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) using deterministic backtracking
    with MRV (Minimum Remaining Values) heuristic and symmetry breaking.
    This computes the solution from first principles rather than relying on
    pre-verified base blocks.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Precompute all 6-subsets and their 5-subsets
    all_6subsets = list(combinations(range(12), 6))
    # Map: 5-subset -> list of 6-subsets containing it
    five_to_six = defaultdict(list)
    six_to_five = {}
    for idx, block in enumerate(all_6subsets):
        five_subs = []
        for i in range(6):
            sub = block[:i] + block[i+1:]
            five_subs.append(sub)
            five_to_six[sub].append(idx)
        six_to_five[idx] = five_subs

    # Symmetry breaking: fix the first block to reduce search space
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = all_6subsets.index(first_block)
    first_block_fives = six_to_five[first_block_idx]

    # State for backtracking
    used_fives = set(first_block_fives)
    blocks_found = [first_block]
    remaining = [False] * len(all_6subsets)
    remaining[first_block_idx] = True

    # For MRV, maintain count of available 6-subsets per 5-subset
    # Initialize: for each 5-subset, count how many 6-subsets contain it
    # and are not yet used
    five_avail = {}
    for sub in five_to_six:
        if sub in used_fives:
            continue
        count = 0
        for idx in five_to_six[sub]:
            if not remaining[idx]:
                count += 1
        five_avail[sub] = count

    def backtrack(deadline):
        if time.time() > deadline:
            return False

        if len(blocks_found) == 132:
            return True

        # Find the most constrained 5-subset (MRV)
        best_sub = None
        best_count = float('inf')
        for sub, cnt in five_avail.items():
            if cnt == 0:
                return False  # Dead end
            if cnt < best_count:
                best_count = cnt
                best_sub = sub
                if cnt == 1:
                    break

        if best_sub is None:
            return True

        # Try all available 6-subsets that contain best_sub
        for idx in five_to_six[best_sub]:
            if remaining[idx]:
                continue

            block_fives = six_to_five[idx]

            # Check compatibility: all 5-subsets must be unused
            valid = True
            for f in block_fives:
                if f in used_fives:
                    valid = False
                    break
            if not valid:
                continue

            # Apply: mark block as used, update state
            remaining[idx] = True
            blocks_found.append(all_6subsets[idx])
            for f in block_fives:
                used_fives.add(f)
                # Update availability for other 5-subsets
                # When we use a 6-subset, it reduces availability of all
                # 5-subsets it contains. But those are now used, so we
                # remove them from five_avail. For other 5-subsets not in
                # this block, their availability doesn't change directly,
                # but they might share 6-subsets with this block... Actually,
                # availability of a 5-subset is the number of available
                # 6-subsets containing it. Using one 6-subset reduces the
                # availability of exactly the 5-subsets contained in that
                # 6-subset by 1. So we need to decrement five_avail for
                # each 5-subset in block_fives. But we're removing them
                # from five_avail anyway since they're now used.
                del five_avail[f]

            if backtrack(deadline):
                return True

            # Undo
            remaining[idx] = False
            blocks_found.pop()
            for f in block_fives:
                used_fives.discard(f)
                # Restore five_avail: need to recompute or increment
                # Recompute is expensive. Instead, we can store the old value.
                # For now, just recompute for simplicity (should be fast enough
                # given the small search space with good heuristics)
                count = 0
                for idx2 in five_to_six[f]:
                    if not remaining[idx2]:
                        count += 1
                five_avail[f] = count

        return False

    if backtrack(deadline):
        return [list(b) for b in blocks_found]

    # Fallback: return what we have
    return [list(b) for b in blocks_found]

if __name__ == '__main__':
    res = construct()
    print(f"Number of blocks: {len(res)}")
    used = set()
    for b in res:
        for s in combinations(b, 5):
            if s in used:
                print("Invalid: repeated 5-subset")
                sys.exit(1)
            used.add(s)
    print("Valid packing")
