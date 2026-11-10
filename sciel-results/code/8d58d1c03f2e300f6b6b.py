import time
import random
import numpy as np
import itertools
from collections import Counter

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 135.0

    # ---------- Precomputation ----------
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_to_idx = {ts: i for i, ts in enumerate(all_t)}

    all_k = list(itertools.combinations(range(v), k))
    n_k = len(all_k)
    k_to_idx = {b: i for i, b in enumerate(all_k)}

    block_covers = []
    for b in all_k:
        covers = []
        for ts in itertools.combinations(b, t):
            covers.append(t_to_idx[ts])
        block_covers.append(np.array(covers, dtype=np.int32))

    # ---------- Vectorized Greedy ----------
    def greedy_construct():
        coverage = np.zeros(n_t, dtype=np.int32)
        selected = []
        selected_set = set()

        while True:
            if time.time() > DEADLINE - 15:
                break
            uncovered = coverage == 0
            if not uncovered.any():
                break

            best_gain = -1
            best_b = -1
            if n_k > 50000:
                candidates = random.sample(range(n_k), 50000)
            else:
                candidates = range(n_k)

            for b_idx in candidates:
                if b_idx in selected_set:
                    continue
                covers = block_covers[b_idx]
                gain = uncovered[covers].sum()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx

            if best_b == -1 or best_gain <= 0:
                break

            selected.append(best_b)
            selected_set.add(best_b)
            coverage[block_covers[best_b]] += 1

        return [list(all_k[b]) for b in selected], coverage

    # ---------- Redundancy Removal ----------
    def remove_redundant(blocks, coverage):
        if not blocks:
            return blocks, coverage
        coverage = np.zeros(n_t, dtype=np.int32)
        block_covers_arr = [block_covers[k_to_idx[tuple(b)]] for b in blocks]
        for bc in block_covers_arr:
            coverage[bc] += 1

        changed = True
        while changed:
            if time.time() > DEADLINE - 20:
                break
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                bc = block_covers_arr[i]
                if (coverage[bc] <= 1).any():
                    continue
                coverage[bc] -= 1
                blocks.pop(i)
                block_covers_arr.pop(i)
                changed = True
        return blocks, coverage

    # ---------- Simulated Annealing with proper state tracking ----------
    def simulated_annealing(blocks, coverage, max_iters=3000):
        best_blocks = blocks[:]
        best_count = len(blocks)

        T_start = 3.0
        T_end = 0.1
        cooling = (T_end / T_start) ** (1.0 / max(1, max_iters - 1))
        T = T_start

        for it in range(max_iters):
            if time.time() > DEADLINE - 25:
                break

            # Current state
            cur_blocks = best_blocks[:]
            cur_coverage = np.zeros(n_t, dtype=np.int32)
            cur_bc = []
            cur_set = set()
            for b in cur_blocks:
                bidx = k_to_idx[tuple(b)]
                bc = block_covers[bidx]
                cur_coverage[bc] += 1
                cur_bc.append(bc)
                cur_set.add(bidx)

            move_type = random.random()
            delta = 0
            valid_move = False

            if move_type < 0.35:
                # Remove a redundant block
                redundant = [i for i, bc in enumerate(cur_bc) if not (cur_coverage[bc] <= 1).any()]
                if redundant:
                    i = random.choice(redundant)
                    bc = cur_bc[i]
                    cur_coverage[bc] -= 1
                    removed_block = cur_blocks[i]
                    cur_blocks.pop(i)
                    cur_bc.pop(i)
                    cur_set.discard(k_to_idx[tuple(removed_block)])
                    delta = 1
                    valid_move = True
            elif move_type < 0.65:
                # Add a random block
                b_idx = random.randrange(n_k)
                if b_idx not in cur_set:
                    bc = block_covers[b_idx]
                    cur_coverage[bc] += 1
                    cur_blocks.append(list(all_k[b_idx]))
                    cur_bc.append(bc)
                    cur_set.add(b_idx)
                    delta = -1
                    valid_move = True
            else:
                # Swap
                if cur_blocks:
                    i = random.randrange(len(cur_blocks))
                    j = random.randrange(n_k)
                    if j not in cur_set:
                        old_idx = k_to_idx[tuple(cur_blocks[i])]
                        old_bc = cur_bc[i]
                        new_bc = block_covers[j]
                        cur_coverage[old_bc] -= 1
                        cur_coverage[new_bc] += 1
                        cur_blocks[i] = list(all_k[j])
                        cur_bc[i] = new_bc
                        cur_set.discard(old_idx)
                        cur_set.add(j)
                        delta = 0
                        valid_move = True

            if not valid_move:
                continue

            # Accept/reject
            new_count = len(cur_blocks)
            if delta >= 0:
                # Improvement or neutral
                if new_count < best_count:
                    best_count = new_count
                    best_blocks = cur_blocks[:]
            else:
                # Worse
                if random.random() < np.exp(delta / T):
                    if new_count < best_count:
                        best_count = new_count
                        best_blocks = cur_blocks[:]
                # else: reject, keep best_blocks

            T *= cooling

        return best_blocks

    # ---------- Main ----------
    blocks, coverage = greedy_construct()
    blocks, coverage = remove_redundant(blocks, coverage)

    if time.time() < DEADLINE - 35:
        blocks = simulated_annealing(blocks, coverage, max_iters=2000)

    blocks, coverage = remove_redundant(blocks, coverage)

    return blocks
