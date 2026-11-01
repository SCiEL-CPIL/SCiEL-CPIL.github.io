"""Covering design C(24,7,3) via simulated annealing."""
# EVOLVE-BLOCK-START
import itertools
import random
import math
import time
from collections import defaultdict


def _all_t_subsets(v, t):
    return list(itertools.combinations(range(v), t))


def _block_covers(block, t):
    return set(itertools.combinations(block, t))


def _greedy_construct(v, k, t, rng):
    """Greedy set-cover style construction."""
    all_subs = _all_t_subsets(v, t)
    uncovered = set(all_subs)
    blocks = []
    while uncovered:
        first = min(uncovered)
        block = list(first)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                best_x = next(x for x in range(v) if x not in block)
            block.append(best_x)
        block.sort()
        blocks.append(tuple(block))
        covered = _block_covers(block, t)
        uncovered -= covered
    return blocks


def _sa_optimize(blocks, v, k, t, rng, time_budget=100):
    """Simulated annealing to reduce block count."""
    start = time.time()
    current_blocks = list(blocks)
    best_blocks = list(current_blocks)
    best_size = len(current_blocks)

    all_subs_set = set(_all_t_subsets(v, t))
    total_subs = len(all_subs_set)

    # Coverage count
    cover_count = defaultdict(int)
    for b in current_blocks:
        for sub in _block_covers(b, t):
            cover_count[sub] += 1

    def uncovered_total():
        return sum(1 for s in all_subs_set if cover_count[s] == 0)

    current_uncovered = uncovered_total()

    T = 5.0
    T_min = 0.05
    cooling = 0.9997

    while T > T_min:
        if time.time() - start > time_budget:
            break

        move_type = rng.random()
        if move_type < 0.5:
            # Try removal
            if len(current_blocks) <= 1:
                T *= cooling
                continue
            idx = rng.randrange(len(current_blocks))
            b = current_blocks[idx]
            b_subs = _block_covers(b, t)
            new_uncovered = 0
            for sub in b_subs:
                if cover_count[sub] == 1:
                    new_uncovered += 1
            delta_e = new_uncovered
            if delta_e == 0:
                for sub in b_subs:
                    cover_count[sub] -= 1
                current_blocks.pop(idx)
                current_uncovered = uncovered_total()
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
            else:
                if rng.random() < math.exp(-delta_e / T):
                    for sub in b_subs:
                        cover_count[sub] -= 1
                    current_blocks.pop(idx)
                    current_uncovered = uncovered_total()
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
        elif move_type < 0.85:
            # Try swap
            if len(current_blocks) <= 1:
                T *= cooling
                continue
            idx = rng.randrange(len(current_blocks))
            old_b = current_blocks[idx]
            new_b = tuple(sorted(rng.sample(range(v), k)))
            if new_b == old_b:
                T *= cooling
                continue
            old_subs = _block_covers(old_b, t)
            new_subs = _block_covers(new_b, t)
            delta_e = 0
            for sub in old_subs:
                if cover_count[sub] == 1 and sub not in new_subs:
                    delta_e += 1
            for sub in new_subs:
                if cover_count[sub] == 0 and sub not in old_subs:
                    delta_e -= 1
            if delta_e <= 0:
                for sub in old_subs:
                    cover_count[sub] -= 1
                for sub in new_subs:
                    cover_count[sub] += 1
                current_blocks[idx] = new_b
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
            else:
                if rng.random() < math.exp(-delta_e / T):
                    for sub in old_subs:
                        cover_count[sub] -= 1
                    for sub in new_subs:
                        cover_count[sub] += 1
                    current_blocks[idx] = new_b
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
        else:
            # Try addition (only if we have uncovered)
            if current_uncovered > 0 and len(current_blocks) < 150:
                new_b = tuple(sorted(rng.sample(range(v), k)))
                new_subs = _block_covers(new_b, t)
                gain = sum(1 for sub in new_subs if cover_count[sub] == 0)
                if gain > 0:
                    for sub in new_subs:
                        cover_count[sub] += 1
                    current_blocks.append(new_b)
                    current_uncovered = uncovered_total()
                elif rng.random() < 0.05:
                    for sub in new_subs:
                        cover_count[sub] += 1
                    current_blocks.append(new_b)
                    current_uncovered = uncovered_total()

        T *= cooling

    # Ensure validity; if not, fall back to greedy
    if uncovered_total() > 0:
        # Rebuild from best_blocks if valid, else use greedy
        bc = defaultdict(int)
        for b in best_blocks:
            for sub in _block_covers(b, t):
                bc[sub] += 1
        if all(bc[s] > 0 for s in all_subs_set):
            return best_blocks
        # Fallback: extend best_blocks with greedy
        uncovered = set(s for s in all_subs_set if bc[s] == 0)
        blocks_list = list(best_blocks)
        while uncovered:
            first = min(uncovered)
            block = list(first)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    best_x = next(x for x in range(v) if x not in block)
                block.append(best_x)
            block.sort()
            blocks_list.append(tuple(block))
            covered = _block_covers(block, t)
            uncovered -= covered
        return blocks_list

    return best_blocks


def construct(v, k, t):
    rng = random.Random(42)
    blocks = _greedy_construct(v, k, t, rng)
    optimized = _sa_optimize(blocks, v, k, t, rng, time_budget=100)
    return [list(b) for b in optimized]
# EVOLVE-BLOCK-END
