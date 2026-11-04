import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(t_idxs)

    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_cover(seed):
        """Greedy set cover with optional randomization."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        blocks_idx = []
        used_blocks = set()

        while uncovered.any():
            # Pick an uncovered t-subset - use random if seed > 0, else min
            if seed > 0:
                uncovered_indices = np.where(uncovered)[0]
                s_idx = rng.choice(uncovered_indices.tolist())
            else:
                s_idx = int(np.argmax(uncovered))  # first True

            # Get candidate k-subsets containing s_idx
            candidates = t_sub_to_k_idxs[s_idx]

            best_block_idx = -1
            best_gain = -1

            for b_idx in candidates:
                if b_idx in used_blocks:
                    continue
                # Calculate gain
                gain = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx

            if best_block_idx == -1:
                break

            blocks_idx.append(best_block_idx)
            used_blocks.add(best_block_idx)

            # Update uncovered
            for t_idx in k_sub_t_idxs[best_block_idx]:
                uncovered[t_idx] = False

        return blocks_idx

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def local_search(blocks_idx, max_iter=100):
        """Try to swap blocks to reduce count."""
        current = list(blocks_idx)
        if not current:
            return current

        # Build cover count
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current:
            for t_idx in k_sub_t_idxs[b_idx]:
                cover_count[t_idx] += 1

        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1

            # Try removing each block
            for i, b_idx in enumerate(current):
                # Check if removal would uncover any t-subset
                can_remove = True
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break

                if can_remove:
                    # Remove it
                    for t_idx in k_sub_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
                    current.pop(i)
                    improved = True
                    break

        return current

    # Run multiple greedy passes with different seeds
    best_blocks = None
    best_len = float('inf')

    # First, deterministic greedy
    blocks_idx = greedy_cover(0)
    blocks_idx = reduce_blocks(blocks_idx)
    if blocks_idx:
        best_blocks = blocks_idx
        best_len = len(blocks_idx)

    # Try randomized greedy if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
        for seed in seeds:
            if time.time() >= deadline - 5.0:
                break
            blocks_idx = greedy_cover(seed)
            blocks_idx = reduce_blocks(blocks_idx)
            if blocks_idx and len(blocks_idx) < best_len:
                best_blocks = blocks_idx
                best_len = len(blocks_idx)

    # Local search on best solution
    if best_blocks and time.time() < deadline - 5.0:
        improved = local_search(best_blocks, max_iter=200)
        if len(improved) < best_len:
            best_blocks = improved
            best_len = len(improved)

    # Final reduction pass
    if best_blocks and time.time() < deadline - 2.0:
        best_blocks = reduce_blocks(best_blocks)

    # Convert to actual blocks
    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
