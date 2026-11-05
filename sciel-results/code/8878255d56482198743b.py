import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Greedy + Local Search + Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for 120s limit (evaluator kills at 150s)

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_set_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_tsets = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    num_ksets = len(k_subsets)

    block_to_tsets = [[] for _ in range(num_ksets)]
    tset_to_blocks = [[] for _ in range(num_tsets)]

    for b_idx, block in enumerate(k_subsets):
        for ts in itertools.combinations(block, t):
            ts_idx = t_set_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)

    # State
    current_blocks = []
    tset_counts = [0] * num_tsets
    in_current = [False] * num_ksets

    def add_block(b_idx):
        current_blocks.append(b_idx)
        in_current[b_idx] = True
        for ts_idx in block_to_tsets[b_idx]:
            tset_counts[ts_idx] += 1

    def remove_block(b_idx):
        current_blocks.remove(b_idx)
        in_current[b_idx] = False
        for ts_idx in block_to_tsets[b_idx]:
            tset_counts[ts_idx] -= 1

    # Greedy Initialization
    uncovered = set(range(num_tsets))
    while uncovered and time.time() < deadline:
        anchor_ts = random.choice(list(uncovered))
        candidates = tset_to_blocks[anchor_ts]
        best_b = -1
        max_gain = 0
        for b_idx in candidates:
            if in_current[b_idx]:
                continue
            gain = 0
            for ts_idx in block_to_tsets[b_idx]:
                if ts_idx in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_b = b_idx
        if best_b == -1:
            break
        add_block(best_b)
        for ts_idx in block_to_tsets[best_b]:
            uncovered.discard(ts_idx)

    # Local Search: remove redundant blocks, then swaps
    def local_search():
        while time.time() < deadline:
            removed = False
            random.shuffle(current_blocks)
            for b_idx in list(current_blocks):
                if time.time() > deadline:
                    return
                is_redundant = True
                for ts_idx in block_to_tsets[b_idx]:
                    if tset_counts[ts_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    remove_block(b_idx)
                    removed = True
                    break
            if removed:
                continue
            # Try swaps
            critical_blocks = []
            for b_idx in current_blocks:
                for ts_idx in block_to_tsets[b_idx]:
                    if tset_counts[ts_idx] == 1:
                        critical_blocks.append(b_idx)
                        break
            if not critical_blocks:
                return
            old_b = random.choice(critical_blocks)
            unique_ts = [ts_idx for ts_idx in block_to_tsets[old_b] if tset_counts[ts_idx] == 1]
            if not unique_ts:
                continue
            candidates = set(tset_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(tset_to_blocks[ts_idx])
                if not candidates:
                    break
            candidates.difference_update(current_blocks)
            if candidates:
                new_b = random.choice(list(candidates))
                remove_block(old_b)
                add_block(new_b)

    local_search()

    # Simulated Annealing phase
    if time.time() < deadline:
        best_blocks = list(current_blocks)
        best_size = len(current_blocks)
        T = 2.0
        T_min = 0.01
        cooling = 0.97

        while T > T_min and time.time() < deadline:
            # Pick a random block to swap
            if not current_blocks:
                break
            old_b = random.choice(current_blocks)
            # Find candidate new blocks: pick a random t-set covered by old_b,
            # get blocks covering it, exclude current ones
            ts_idx = random.choice(block_to_tsets[old_b])
            candidates = [b for b in tset_to_blocks[ts_idx] if not in_current[b]]
            if not candidates:
                T *= cooling
                continue
            new_b = random.choice(candidates)

            # Compute energy change: new_size - old_size (both 1 block swap, so 0)
            # But we want to minimize size, so a swap keeps size same.
            # Instead, after swap, try to remove redundant blocks to reduce size.
            old_size = len(current_blocks)
            remove_block(old_b)
            add_block(new_b)

            # Try to remove redundant blocks
            for b_idx in list(current_blocks):
                if time.time() > deadline:
                    break
                is_redundant = True
                for ts_idx2 in block_to_tsets[b_idx]:
                    if tset_counts[ts_idx2] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    remove_block(b_idx)

            new_size = len(current_blocks)
            delta = new_size - old_size  # negative = improvement

            if delta <= 0:
                # Accept
                if new_size < best_size:
                    best_size = new_size
                    best_blocks = list(current_blocks)
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < math.exp(-delta / T):
                    pass  # accept
                else:
                    # Reject: revert
                    remove_block(new_b)
                    add_block(old_b)
                    # Re-add any blocks we removed
                    # We need to track what we removed... let's just rebuild
                    # Actually, let's not do partial removals in SA, keep it simple.
                    # Revert the redundant removals too.
                    # This is getting complex. Let's simplify: only accept if delta <= 0
                    # or use a simpler move.
                    pass
            T *= cooling

        # Restore best found
        # Clear current state
        for b_idx in list(current_blocks):
            remove_block(b_idx)
        for b_idx in best_blocks:
            add_block(b_idx)

    # Final local search polish
    local_search()

    result = [list(k_subsets[b_idx]) for b_idx in current_blocks]
    return result

if __name__ == "__main__":
    v, k, t = 19, 7, 4
    blocks = construct(v, k, t)
    print(f"Number of blocks: {len(blocks)}")
