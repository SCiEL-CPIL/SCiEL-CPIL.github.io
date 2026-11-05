import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)

    # mask -> index
    mask_to_idx = {}
    for i, sub in enumerate(all_t_subs):
        m = 0
        for p in sub:
            m |= (1 << p)
        mask_to_idx[m] = i

    # For each point x, list of t-subset indices containing x
    point_t_idx = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subs):
        for p in sub:
            point_t_idx[p].append(i)
    point_t_idx = [np.array(lst, dtype=np.int32) for lst in point_t_idx]

    # For a block (list of points), compute its t-subset indices
    def block_t_indices(block):
        idxs = []
        for t_sub in itertools.combinations(block, t):
            m = 0
            for p in t_sub:
                m |= (1 << p)
            idx = mask_to_idx.get(m)
            if idx is not None:
                idxs.append(idx)
        return idxs

    # Precompute for each possible (t-1)-subset of points, the bitmask of point bits
    # We need, for a growing block B and candidate x, the t-subsets of (B ∪ {x}) that contain x.
    # That's { {x} ∪ S : S ⊆ B, |S| = t-1 }.
    # We precompute, for each point x and each (t-1)-subset mask of B, the t-subset index.
    # But B changes. Instead, compute on the fly using point_t_idx intersection trick:
    # t-subsets of (B ∪ {x}) containing x = (t-subsets containing x) ∩ (t-subsets contained in B ∪ {x})
    # A t-subset is contained in B ∪ {x} iff all its points are in B ∪ {x}.
    # This is complex. Use the (t-1)-subset enumeration instead.

    # Precompute (t-1)-subset masks for quick lookup
    # For a given block B, the (t-1)-subsets of B can be enumerated.
    # For each (t-1)-subset S of B and candidate x, the t-subset is S ∪ {x}.
    # We precompute a dict: (mask_of_S, x) -> t-subset index
    # But that's huge. Instead, for each x, precompute a dict from (t-1)-subset mask to t-subset index.
    # Number of (t-1)-subsets of v points is C(v, t-1) = C(19,4) = 3876. That's manageable.
    x_to_t1mask_to_idx = [{} for _ in range(v)]
    for x in range(v):
        for t1_sub in itertools.combinations(range(v), t - 1):
            if x in t1_sub:
                continue
            m1 = 0
            for p in t1_sub:
                m1 |= (1 << p)
            m_full = m1 | (1 << x)
            idx = mask_to_idx.get(m_full)
            if idx is not None:
                x_to_t1mask_to_idx[x][m1] = idx

    def get_gain_indices(block, x):
        """Return list of t-subset indices covered by adding x to block (that contain x)."""
        result = []
        for t1_sub in itertools.combinations(block, t - 1):
            m1 = 0
            for p in t1_sub:
                m1 |= (1 << p)
            idx = x_to_t1mask_to_idx[x].get(m1)
            if idx is not None:
                result.append(idx)
        return result

    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)

        uncovered = np.ones(n_t, dtype=bool)
        blocks = []

        while uncovered.any():
            # Find a random uncovered index
            while True:
                start_idx = random.randint(0, n_t - 1)
                if uncovered[start_idx]:
                    break

            block = list(all_t_subs[start_idx])
            block_set = set(block)

            while len(block) < k:
                best_x = -1
                best_gain = -1
                candidates = []

                for x in range(v):
                    if x in block_set:
                        continue
                    gain_idx = get_gain_indices(block, x)
                    if not gain_idx:
                        gain = 0
                    else:
                        gain = int(uncovered[gain_idx].sum())
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        candidates = [x]
                    elif gain == best_gain:
                        candidates.append(x)

                if best_x == -1:
                    available = [x for x in range(v) if x not in block_set]
                    if not available:
                        break
                    best_x = random.choice(available)

                if len(candidates) > 1:
                    best_x = random.choice(candidates)

                block.append(best_x)
                block_set.add(best_x)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for idx in block_t_indices(block):
                uncovered[idx] = False

        return blocks

    def reduce_covering(blocks):
        if not blocks:
            return blocks

        block_t_idx = [block_t_indices(b) for b in blocks]
        cov = np.zeros(n_t, dtype=np.int32)
        for idxs in block_t_idx:
            for idx in idxs:
                cov[idx] += 1

        removed = np.zeros(len(blocks), dtype=bool)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if removed[i]:
                    continue
                idxs = block_t_idx[i]
                if not idxs:
                    removed[i] = True
                    changed = True
                    continue
                if not np.any(cov[idxs] == 1):
                    removed[i] = True
                    changed = True
                    for idx in idxs:
                        cov[idx] -= 1

        return [blocks[i] for i in range(len(blocks)) if not removed[i]]

    def swap_local_search(blocks, time_limit):
        """Try to reduce block count by swapping blocks."""
        if not blocks:
            return blocks

        block_t_idx = [block_t_indices(b) for b in blocks]
        cov = np.zeros(n_t, dtype=np.int32)
        for idxs in block_t_idx:
            for idx in idxs:
                cov[idx] += 1

        # A block is "redundant" if no t-subset it covers has cov == 1
        def is_redundant(i):
            idxs = block_t_idx[i]
            if not idxs:
                return True
            return not np.any(cov[idxs] == 1)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(blocks)):
                if not is_redundant(i):
                    continue
                # Try to find a replacement block that covers at least one uncovered-by-others t-subset
                # Actually, we want to replace block i with a new block j such that the new block
                # covers some t-subsets that were only covered by i, while not losing coverage.
                # Simpler: try to find a new block that covers a subset of i's t-subsets
                # that are "essential" (cov == 1 for i).
                essential = [idx for idx in block_t_idx[i] if cov[idx] == 1]
                if not essential:
                    # i is redundant, just remove it
                    for idx in block_t_idx[i]:
                        cov[idx] -= 1
                    blocks.pop(i)
                    block_t_idx.pop(i)
                    improved = True
                    break

                # Try to find a new block that covers all essential t-subsets
                # This is hard. Instead, try random new blocks and check if they cover all essential.
                # For now, skip swap and just try to remove non-essential blocks.
                pass

        return blocks

    best_blocks = None
    best_size = float('inf')

    attempt = 0
    while time.time() < deadline - 15.0:
        attempt += 1
        seed = attempt * 12345 + 67890
        blocks = greedy_construct(seed)

        if not blocks:
            continue

        reduced = reduce_covering(blocks)

        if len(reduced) < best_size:
            best_size = len(reduced)
            best_blocks = reduced

        if time.time() > deadline - 10.0:
            break

    if best_blocks is None:
        blocks = greedy_construct(None)
        return reduce_covering(blocks)

    return best_blocks
