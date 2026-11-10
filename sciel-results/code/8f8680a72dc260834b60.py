import time
import numpy as np
import random

N = 256
DEADLINE = 85.0  # Safety margin before 90s limit


def _count_3ap_vec(mask):
    """Count 3-APs using vectorized numpy operations.
    A 3-AP (a, b, c) with a < b < c, b-a = c-b means a+c = 2b.
    For each pair (a, c) with a < c and even gap, b = (a+c)//2.
    We count triples where all three are in the set.
    """
    n = len(mask)
    # For each b (middle element), count pairs (a, c) with a < b < c, c-b = b-a
    # i.e., a = b-d, c = b+d for d >= 1
    count = 0
    # Vectorize: for each b in selected, count d such that mask[b-d] and mask[b+d]
    idx = np.where(mask)[0]
    if len(idx) < 3:
        return 0
    # For each b in idx, count valid d
    for b in idx:
        # d ranges from 1 to min(b, n-1-b)
        max_d = min(b, n - 1 - b)
        if max_d < 1:
            continue
        # Check mask at positions b-d and b+d for d=1..max_d
        left = mask[:b][::-1]  # mask[b-1], mask[b-2], ..., mask[0]
        right = mask[b+1:]     # mask[b+1], mask[b+2], ..., mask[n-1]
        m = min(len(left), len(right))
        if m > 0:
            count += np.sum(left[:m] & right[:m])
    return int(count)


def _count_3ap_fast(mask):
    """Faster 3-AP count using convolution-like approach.
    For each difference d, count b such that mask[b-d] & mask[b] & mask[b+d].
    """
    n = len(mask)
    count = 0
    # Max difference
    for d in range(1, n // 2 + 1):
        # b ranges from d to n-1-d
        a_part = mask[:n - 2 * d]       # positions 0..n-2d-1 -> a = b-d
        b_part = mask[d:n - d]          # positions d..n-d-1 -> b
        c_part = mask[2 * d:]           # positions 2d..n-1 -> c = b+d
        count += np.sum(a_part & b_part & c_part)
    return int(count)


def _has_4ap_vec(mask, new_pos):
    """Check if adding new_pos creates a 4-AP. Vectorized check."""
    n = len(mask)
    # 4-AP: a, a+d, a+2d, a+3d
    # new_pos can be at position 0, 1, 2, or 3 in the progression
    # Position 0: new_pos = a, need a+d, a+2d, a+3d all in mask
    # Position 1: new_pos = a+d, need a, a+2d, a+3d in mask
    # Position 2: new_pos = a+2d, need a, a+d, a+3d in mask
    # Position 3: new_pos = a+3d, need a, a+d, a+2d in mask

    # Position 0: new_pos = a
    # a+d, a+2d, a+3d must be in mask
    max_d = (n - 1 - new_pos) // 3
    if max_d >= 1:
        # For d in 1..max_d, check mask[new_pos+d], mask[new_pos+2d], mask[new_pos+3d]
        for d in range(1, max_d + 1):
            if mask[new_pos + d] and mask[new_pos + 2 * d] and mask[new_pos + 3 * d]:
                return True

    # Position 1: new_pos = a + d, so a = new_pos - d
    # Need a, a+2d=new_pos+d, a+3d=new_pos+2d in mask
    max_d = new_pos // 1  # a = new_pos - d >= 0 => d <= new_pos
    max_d = min(max_d, (n - 1 - new_pos) // 2)
    if max_d >= 1:
        for d in range(1, max_d + 1):
            a = new_pos - d
            if mask[a] and mask[new_pos + d] and mask[new_pos + 2 * d]:
                return True

    # Position 2: new_pos = a + 2d, so a = new_pos - 2d
    # Need a, a+d=new_pos-d, a+3d=new_pos+d in mask
    max_d = new_pos // 2
    max_d = min(max_d, (n - 1 - new_pos) // 1)
    if max_d >= 1:
        for d in range(1, max_d + 1):
            a = new_pos - 2 * d
            if mask[a] and mask[new_pos - d] and mask[new_pos + d]:
                return True

    # Position 3: new_pos = a + 3d, so a = new_pos - 3d
    # Need a, a+d=new_pos-2d, a+2d=new_pos-d in mask
    max_d = new_pos // 3
    if max_d >= 1:
        for d in range(1, max_d + 1):
            a = new_pos - 3 * d
            if mask[a] and mask[a + d] and mask[a + 2 * d]:
                return True

    return False


def _has_4ap_fast(mask, new_pos):
    """Optimized 4-AP check for adding new_pos."""
    n = len(mask)

    # Position 0: new_pos = a
    max_d = (n - 1 - new_pos) // 3
    if max_d >= 1:
        # Vectorize: check all d at once
        d = np.arange(1, max_d + 1)
        positions1 = new_pos + d
        positions2 = new_pos + 2 * d
        positions3 = new_pos + 3 * d
        if np.any(mask[positions1] & mask[positions2] & mask[positions3]):
            return True

    # Position 1: new_pos = a + d
    max_d = min(new_pos, (n - 1 - new_pos) // 2)
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        a = new_pos - d
        pos2 = new_pos + d
        pos3 = new_pos + 2 * d
        if np.any(mask[a] & mask[pos2] & mask[pos3]):
            return True

    # Position 2: new_pos = a + 2d
    max_d = min(new_pos // 2, n - 1 - new_pos)
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        a = new_pos - 2 * d
        pos1 = new_pos - d
        pos3 = new_pos + d
        if np.any(mask[a] & mask[pos1] & mask[pos3]):
            return True

    # Position 3: new_pos = a + 3d
    max_d = new_pos // 3
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        a = new_pos - 3 * d
        pos1 = a + d  # = new_pos - 2d
        pos2 = a + 2 * d  # = new_pos - d
        if np.any(mask[a] & mask[pos1] & mask[pos2]):
            return True

    return False


def _marginal_3ap_gain(mask, new_pos):
    """Compute how many new 3-APs are created by adding new_pos.
    A new 3-AP must include new_pos.
    """
    n = len(mask)
    gain = 0

    # new_pos as first (a): need b, c with b-a = c-b, so b = (a+c)/2
    # a = new_pos, b = new_pos + d, c = new_pos + 2d
    max_d = (n - 1 - new_pos) // 2
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        b_pos = new_pos + d
        c_pos = new_pos + 2 * d
        gain += np.sum(mask[b_pos] & mask[c_pos])

    # new_pos as middle (b): need a, c with b-a = c-b
    # a = new_pos - d, c = new_pos + d
    max_d = min(new_pos, n - 1 - new_pos)
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        a_pos = new_pos - d
        c_pos = new_pos + d
        gain += np.sum(mask[a_pos] & mask[c_pos])

    # new_pos as last (c): need a, b with c-b = b-a
    # b = new_pos - d, a = new_pos - 2d
    max_d = new_pos // 2
    if max_d >= 1:
        d = np.arange(1, max_d + 1)
        b_pos = new_pos - d
        a_pos = new_pos - 2 * d
        gain += np.sum(mask[b_pos] & mask[a_pos])

    return int(gain)


def _density_greedy(start_time, deadline):
    """Density-driven greedy: always pick element with max marginal 3-AP gain."""
    mask = np.zeros(N, dtype=bool)
    remaining = set(range(N))

    while remaining:
        if time.time() > deadline - 5:
            break

        best_pos = -1
        best_gain = -1

        for pos in remaining:
            if time.time() > deadline - 5:
                break
            if _has_4ap_fast(mask, pos):
                continue
            gain = _marginal_3ap_gain(mask, pos)
            if gain > best_gain:
                best_gain = gain
                best_pos = pos

        if best_pos == -1:
            break

        mask[best_pos] = True
        remaining.discard(best_pos)

    return mask


def _simulated_annealing(mask, start_time, deadline, max_iters=50000):
    """Simulated annealing with vectorized operations."""
    current = mask.copy()
    current_score = _count_3ap_fast(current)
    best = current.copy()
    best_score = current_score

    T = 5.0
    cooling = 0.9995
    selected_idx = np.where(current)[0]
    unselected_idx = np.where(~current)[0]

    iter_count = 0
    while iter_count < max_iters:
        if time.time() > deadline - 3:
            break
        iter_count += 1

        # Choose a random move: add, remove, or swap
        move_type = random.random()
        accepted = False

        if move_type < 0.3 and len(unselected_idx) > 0:
            # Add a random unselected element
            pos = int(random.choice(unselected_idx))
            if _has_4ap_fast(current, pos):
                continue
            new_mask = current.copy()
            new_mask[pos] = True
            new_score = _count_3ap_fast(new_mask)
            delta = new_score - current_score
            if delta >= 0 or random.random() < np.exp(delta / max(T, 0.01)):
                current = new_mask
                current_score = new_score
                accepted = True
                # Update unselected
                unselected_idx = unselected_idx[unselected_idx != pos]
                selected_idx = np.concatenate([selected_idx, [pos]])
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        elif move_type < 0.5 and len(selected_idx) > 0:
            # Remove a random selected element
            pos = int(random.choice(selected_idx))
            new_mask = current.copy()
            new_mask[pos] = False
            new_score = _count_3ap_fast(new_mask)
            delta = new_score - current_score
            if delta >= 0 or random.random() < np.exp(delta / max(T, 0.01)):
                current = new_mask
                current_score = new_score
                accepted = True
                selected_idx = selected_idx[selected_idx != pos]
                unselected_idx = np.concatenate([unselected_idx, [pos]])
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        elif len(selected_idx) > 0 and len(unselected_idx) > 0:
            # Swap: remove one selected, add one unselected
            pos_remove = int(random.choice(selected_idx))
            pos_add = int(random.choice(unselected_idx))
            new_mask = current.copy()
            new_mask[pos_remove] = False
            new_mask[pos_add] = True
            # Check 4-AP constraint
            if _has_4ap_fast(new_mask, pos_add):
                continue
            new_score = _count_3ap_fast(new_mask)
            delta = new_score - current_score
            if delta >= 0 or random.random() < np.exp(delta / max(T, 0.01)):
                current = new_mask
                current_score = new_score
                accepted = True
                selected_idx = selected_idx[selected_idx != pos_remove]
                selected_idx = np.concatenate([selected_idx, [pos_add]])
                unselected_idx = unselected_idx[unselected_idx != pos_add]
                unselected_idx = np.concatenate([unselected_idx, [pos_remove]])
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()

        T *= cooling

    return best, best_score


def _multi_start_greedy(start_time, deadline, n_starts=5):
    """Run density greedy multiple times with different initial seeds."""
    best_mask = None
    best_score = -1

    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
        # Start with a random small seed
        seed_size = random.randint(3, 8)
        seed_positions = random.sample(range(N), seed_size)
        mask = np.zeros(N, dtype=bool)
        valid_seed = True
        for pos in seed_positions:
            if _has_4ap_fast(mask, pos):
                valid_seed = False
                break
            mask[pos] = True
        if not valid_seed:
            continue

        # Continue with density greedy
        remaining = set(range(N)) - set(seed_positions)
        while remaining:
            if time.time() > deadline - 10:
                break
            best_pos = -1
            best_gain = -1
            for pos in remaining:
                if time.time() > deadline - 10:
                    break
                if _has_4ap_fast(mask, pos):
                    continue
                gain = _marginal_3ap_gain(mask, pos)
                if gain > best_gain:
                    best_gain = gain
                    best_pos = pos
            if best_pos == -1:
                break
            mask[best_pos] = True
            remaining.discard(best_pos)

        score = _count_3ap_fast(mask)
        if score > best_score:
            best_score = score
            best_mask = mask.copy()

    return best_mask, best_score


def _local_search_improved(mask, start_time, deadline):
    """Improved local search: try all single-element additions and removals."""
    current = mask.copy()
    current_score = _count_3ap_fast(current)
    best = current.copy()
    best_score = current_score

    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False

        # Try adding each unselected element
        for pos in range(N):
            if time.time() > deadline - 5:
                break
            if current[pos]:
                continue
            if _has_4ap_fast(current, pos):
                continue
            new_mask = current.copy()
            new_mask[pos] = True
            new_score = _count_3ap_fast(new_mask)
            if new_score > current_score:
                current = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
                improved = True
                break

        if improved:
            continue

        # Try removing each selected element
        for pos in range(N):
            if time.time() > deadline - 5:
                break
            if not current[pos]:
                continue
            new_mask = current.copy()
            new_mask[pos] = False
            new_score = _count_3ap_fast(new_mask)
            if new_score > current_score:
                current = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
                improved = True
                break

    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    random.seed(42)

    # Phase 1: Density-driven greedy (deterministic)
    mask1 = _density_greedy(start_time, deadline)
    score1 = _count_3ap_fast(mask1)

    # Phase 2: Multi-start greedy with random seeds
    mask2, score2 = _multi_start_greedy(start_time, deadline, n_starts=8)
    if mask2 is not None and score2 > score1:
        mask1 = mask2
        score1 = score2

    # Phase 3: Local search improvement
    if time.time() < deadline - 15:
        mask1, score1 = _local_search_improved(mask1, start_time, deadline)

    # Phase 4: Simulated annealing from best
    if time.time() < deadline - 10:
        mask1, score1 = _simulated_annealing(mask1, start_time, deadline, max_iters=30000)

    # Phase 5: Final local search
    if time.time() < deadline - 5:
        mask1, score1 = _local_search_improved(mask1, start_time, deadline)

    # Convert to 1-indexed list
    result = sorted([int(i) + 1 for i in np.where(mask1)[0]])

    if len(result) == 0:
        return [1]

    return result
