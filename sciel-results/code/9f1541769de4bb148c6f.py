"""Rewritten program: maximize 4-uniform family on [16] with no 3-sunflowers."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 82  # Safety margin before 90s kill

    # Pre-compute all 4-element subsets of [1,16] as bitmasks
    all_rows = []
    all_masks = []
    for row in combinations(range(1, 17), 4):
        mask = 0
        for i in row:
            mask |= (1 << (i - 1))
        all_masks.append(mask)
        all_rows.append(row)

    n_total = len(all_masks)  # 1820

    # Precompute pairwise intersections as masks
    # inter[i][j] = all_masks[i] & all_masks[j]
    # This is 1820x1820 = ~3.3M entries; each is an int. Memory ~ 3.3M * 8 bytes = ~26MB. OK.
    inter = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_masks[i]
        for j in range(i + 1, n_total):
            v = mi & all_masks[j]
            inter[i][j] = v
            inter[j][i] = v

    def can_add_mask(mask, chosen_indices):
        """Check if mask (as index) can be added to family given by chosen_indices."""
        n = len(chosen_indices)
        for a in range(n):
            ia = chosen_indices[a]
            for b in range(a + 1, n):
                ib = chosen_indices[b]
                if inter[ia][ib] == inter[ia][chosen_indices[-1]] == inter[ib][chosen_indices[-1]]:
                    # Wait, this is wrong. We need to check the new mask against pairs.
                    pass
        # Correct check: for all pairs (ia, ib) in chosen, check sunflower with new mask.
        # (ia & ib) == (ia & new) == (ib & new)
        new_idx = chosen_indices[-1]  # placeholder, not used this way
        # Let's rewrite properly.
        return True

    # Proper can_add: given a candidate index 'cand' and a list of chosen indices 'chosen'
    def can_add(cand, chosen):
        mc = all_masks[cand]
        n = len(chosen)
        for a in range(n):
            ia = chosen[a]
            for b in range(a + 1, n):
                ib = chosen[b]
                if inter[ia][ib] == (all_masks[ia] & mc) == (all_masks[ib] & mc):
                    return False
        return True

    # Greedy pass
    def greedy_pass(order):
        chosen = []
        for idx in order:
            if can_add(idx, chosen):
                chosen.append(idx)
        return chosen

    # Local search with random swaps
    def local_search(chosen, max_iters=2000):
        chosen_set = set(chosen)
        unchosen = [i for i in range(n_total) if i not in chosen_set]
        chosen = chosen[:]

        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if not unchosen:
                break

            out_pos = random.randint(0, len(chosen) - 1)
            in_pos = random.randint(0, len(unchosen) - 1)
            out_idx = chosen[out_pos]
            in_idx = unchosen[in_pos]

            # Check if in_idx can replace out_idx
            temp = chosen[:out_pos] + chosen[out_pos + 1:]
            if can_add(in_idx, temp):
                chosen[out_pos] = in_idx
                chosen_set.discard(out_idx)
                chosen_set.add(in_idx)
                unchosen[in_pos] = out_idx

        return chosen

    # Systematic local search: for each chosen element, try all unchosen elements
    # This is O(K * N * K^2) which is too slow for K~250, N~1820.
    # Instead, use a smarter approach: for each pair in chosen, find which unchosen
    # sets are blocked by that pair, and prefer to remove a set that blocks many.

    # Alternative: iterative improvement with scoring.
    # Score of a set = number of unchosen sets it blocks.
    # Remove the set with highest score, try to add a random unchosen set.

    def compute_block_scores(chosen):
        """For each chosen index, count how many unchosen sets it blocks."""
        chosen_set = set(chosen)
        scores = {i: 0 for i in chosen}
        unchosen = [i for i in range(n_total) if i not in chosen_set]

        for u in unchosen:
            mu = all_masks[u]
            blocked_by = set()
            for a in range(len(chosen)):
                ia = chosen[a]
                for b in range(a + 1, len(chosen)):
                    ib = chosen[b]
                    if inter[ia][ib] == (all_masks[ia] & mu) == (all_masks[ib] & mu):
                        blocked_by.add(ia)
                        blocked_by.add(ib)
            for idx in blocked_by:
                scores[idx] += 1
        return scores

    def local_search_scored(chosen, max_rounds=50):
        chosen = chosen[:]
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break

            scores = compute_block_scores(chosen)
            if not scores:
                break

            # Find the chosen index with highest block score
            max_score = max(scores.values())
            if max_score == 0:
                break

            candidates = [i for i, s in scores.items() if s == max_score]
            out_idx = random.choice(candidates)
            out_pos = chosen.index(out_idx)

            chosen_set = set(chosen)
            unchosen = [i for i in range(n_total) if i not in chosen_set]

            # Try to find a replacement
            found = False
            for in_idx in unchosen:
                temp = chosen[:out_pos] + chosen[out_pos + 1:]
                if can_add(in_idx, temp):
                    chosen[out_pos] = in_idx
                    found = True
                    break

            if not found:
                # Try random replacements
                for _ in range(50):
                    in_idx = random.choice(unchosen)
                    temp = chosen[:out_pos] + chosen[out_pos + 1:]
                    if can_add(in_idx, temp):
                        chosen[out_pos] = in_idx
                        found = True
                        break

        return chosen

    best_family = []
    best_size = 0

    indices = list(range(n_total))

    # Multiple orderings
    orders = []
    orders.append(indices[:])
    orders.append(indices[::-1])
    orders.append(sorted(indices, key=lambda i: all_masks[i]))
    orders.append(sorted(indices, key=lambda i: sum(all_rows[i])))
    orders.append(sorted(indices, key=lambda i: -sum(all_rows[i])))

    # Random orderings
    for _ in range(5):
        o = indices[:]
        random.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline - 15:
            break

        family = greedy_pass(order)
        size = len(family)

        if time.time() < deadline - 10:
            family = local_search(family, max_iters=1000)
            size = len(family)

        if time.time() < deadline - 5:
            family = local_search_scored(family, max_rounds=20)
            size = len(family)

        if size > best_size:
            best_size = size
            best_family = family[:]

        if best_size >= 250:
            break

    return [list(all_rows[i]) for i in best_family]
# EVOLVE-BLOCK-END
