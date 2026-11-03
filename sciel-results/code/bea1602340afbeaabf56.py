import itertools
import random
import time
from collections import defaultdict


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(frozenset(c) for c in itertools.combinations(range(v), t))
    blocks = []
    block_set = set()

    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(tuple(uncovered))
        block = list(seed)
        rng.shuffle(block)

        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)

            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            blocks.append(b_tuple)
            block_set.add(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))

    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    block_set = set(blocks)

    def compute_counts(blk_list):
        counts = defaultdict(int)
        for b in blk_list:
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                counts[f] += 1
        return counts

    counts = compute_counts(blocks)
    best_blocks = list(blocks)
    best_count = len(blocks)

    total_t = len(list(itertools.combinations(range(v), t)))
    uncovered = total_t - sum(1 for c in counts.values() if c > 0)

    def apply_block_change(blk, sign):
        """
        sign = 1 to add, -1 to remove.
        Returns change in uncovered count (delta_uncovered).
        """
        delta_uncovered = 0
        for fs in itertools.combinations(blk, t):
            f = frozenset(fs)
            old_c = counts.get(f, 0)
            new_c = old_c + sign
            if new_c == 0:
                if f in counts:
                    if old_c == 1:
                        delta_uncovered += 1
                    del counts[f]
            elif new_c > 0:
                if old_c == 0:
                    delta_uncovered -= 1
                counts[f] = new_c
        return delta_uncovered

    no_improve_iter = 0
    max_no_improve = 200

    while time.time() - start_time < time_limit:
        # 1. Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                delta_u = apply_block_change(b, -1)
                uncovered += delta_u
                if uncovered == 0:
                    blocks.pop(i)
                    block_set.discard(b)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed_any = True
                    no_improve_iter = 0
                    break
                else:
                    apply_block_change(b, 1)
                    uncovered -= delta_u

        if removed_any:
            continue

        # 2. Try swaps
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]

            critical = []
            for fs in itertools.combinations(old_block, t):
                f = frozenset(fs)
                if counts.get(f, 0) == 1:
                    critical.append(f)

            if not critical:
                continue

            points_needed = set()
            for fs in critical:
                points_needed.update(fs)

            if len(points_needed) <= k:
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)

                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t - 1):
                        fs = frozenset(list(rest) + [x])
                        c = counts.get(fs, 0)
                        if c == 0:
                            gain += 10
                        elif c == 1:
                            gain += 1
                    new_block_set.add(x)

                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    if new_block != old_block and new_block not in block_set:
                        delta_u_remove = apply_block_change(old_block, -1)
                        delta_u_add = apply_block_change(new_block, 1)

                        new_uncovered = uncovered + delta_u_remove + delta_u_add

                        if new_uncovered == 0:
                            blocks[i] = new_block
                            block_set.discard(old_block)
                            block_set.add(new_block)
                            uncovered = new_uncovered
                            no_improve_iter = 0
                        else:
                            apply_block_change(new_block, -1)
                            apply_block_change(old_block, 1)
                            uncovered = uncovered

        # 3. Perturbation if stuck
        if no_improve_iter > max_no_improve:
            if len(blocks) > 1:
                old_blocks = list(blocks)
                old_block_set = set(block_set)
                old_counts = dict(counts)
                old_uncovered = uncovered

                i = rng.randrange(len(blocks))
                b = blocks[i]

                delta_u = apply_block_change(b, -1)
                uncovered += delta_u
                blocks.pop(i)
                block_set.discard(b)

                unc_list = []
                for c in itertools.combinations(range(v), t):
                    f = frozenset(c)
                    if counts.get(f, 0) == 0:
                        unc_list.append(f)

                while unc_list:
                    seed = rng.choice(unc_list)
                    new_b = list(seed)
                    rng.shuffle(new_b)

                    while len(new_b) < k:
                        best_x = None
                        best_gain = -1
                        cands = [x for x in range(v) if x not in new_b]
                        rng.shuffle(cands)
                        for x in cands:
                            gain = 0
                            for rest in itertools.combinations(new_b, t - 1):
                                fs = frozenset(list(rest) + [x])
                                if counts.get(fs, 0) == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        new_b.append(best_x)

                    new_b.sort()
                    new_b_tuple = tuple(new_b)

                    if new_b_tuple not in block_set:
                        blocks.append(new_b_tuple)
                        block_set.add(new_b_tuple)
                        delta_u_add = apply_block_change(new_b_tuple, 1)
                        uncovered += delta_u_add

                        unc_list = []
                        for c in itertools.combinations(range(v), t):
                            f = frozenset(c)
                            if counts.get(f, 0) == 0:
                                unc_list.append(f)

                if uncovered == 0:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    no_improve_iter = 0
                else:
                    blocks = old_blocks
                    block_set = old_block_set
                    counts = old_counts
                    uncovered = old_uncovered
                    no_improve_iter += 1
            else:
                no_improve_iter += 1
        else:
            no_improve_iter += 1

    return [list(b) for b in best_blocks]
