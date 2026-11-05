"""Covering design C(v,k,t) via strong-greedy max-gain block selection (bitset accelerated)."""
import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin under 120s / 150s kill
    
    random.seed(42)

    # Map every t-subset to an integer id; use a Python int as a bitset for covered.
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # Build masks: for each point x, dict keyed by tuple of (t-1) other points -> bitmask
    masks = [dict() for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i, x in enumerate(s):
            others = tuple(s[:i] + s[i+1:])
            masks[x][others] = masks[x].get(others, 0) | bit

    covered = 0  # bitset of covered t-subsets
    blocks = []

    def count_bits(n):
        return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')

    def gain_of_add(x, B_set):
        g = 0
        B_list = list(B_set)
        for S in itertools.combinations(B_list, t - 1):
            key = tuple(S)
            m = masks[x].get(key)
            if m is not None:
                g += count_bits(m & ~covered)
        return g

    def build_block_from(seed_points):
        B = set(seed_points)
        available = [x for x in range(v) if x not in B]
        while len(B) < k:
            best_x = -1
            best_gain = -1
            for x in available:
                g = gain_of_add(x, B)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            B.add(best_x)
            available.remove(best_x)
        return B

    def compute_new_coverage(B):
        total = 0
        for s in itertools.combinations(sorted(B), t):
            if not (covered >> subset_to_index[s]) & 1:
                total += 1
        return total

    def find_uncovered_indices():
        uncov_mask = ((1 << n_subsets) - 1) & ~covered
        if uncov_mask == 0:
            return []
        uncovered_indices = []
        m = uncov_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            uncovered_indices.append(idx)
            m ^= lsb
        return uncovered_indices

    def select_best_block(uncovered_indices, sample_size, time_budget):
        """Select the best block by trying multiple seeds. Returns (best_block, best_gain)."""
        if not uncovered_indices:
            return None, -1
        
        # Always include a "worst-case" seed: the uncovered triplet that appears in
        # the fewest potential blocks (hardest to cover). We approximate by picking
        # randomly but also ensuring we try at least one from the tail.
        # For determinism + exploration: pick sample_size random + 1 forced worst.
        seeds = random.sample(uncovered_indices, min(sample_size, len(uncovered_indices)))
        # Add a deterministic worst-case seed: pick the one with max index (arbitrary but stable)
        # Actually, let's add the last few uncovered ones as they may be harder
        if len(uncovered_indices) > sample_size:
            # Add a few from the end to ensure we don't miss hard cases
            extra = uncovered_indices[-min(5, len(uncovered_indices)-sample_size):]
            for e in extra:
                if e not in seeds:
                    seeds.append(e)
        
        best_block = None
        best_gain_total = -1
        t_start = time.time()
        
        for idx in seeds:
            if time.time() - t_start > time_budget:
                break
            seed = all_subsets[idx]
            B = build_block_from(set(seed))
            total = compute_new_coverage(B)
            if total > best_gain_total:
                best_gain_total = total
                best_block = B
        
        return best_block, best_gain_total

    # Main greedy loop
    iteration = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break

        uncovered_indices = find_uncovered_indices()
        if not uncovered_indices:
            break

        # Time-adaptive sample size: use more seeds when we have more time
        remaining = deadline - elapsed
        # Budget per iteration: spend at most 2s on block selection, scale with remaining time
        # Early: more exploration. Late: less.
        if remaining > 60:
            sample_size = min(150, len(uncovered_indices))
            time_budget = 2.5
        elif remaining > 30:
            sample_size = min(100, len(uncovered_indices))
            time_budget = 2.0
        elif remaining > 10:
            sample_size = min(50, len(uncovered_indices))
            time_budget = 1.5
        else:
            sample_size = min(20, len(uncovered_indices))
            time_budget = 1.0

        best_block, best_gain = select_best_block(uncovered_indices, sample_size, time_budget)

        if best_block is None or best_gain <= 0:
            # No improvement possible with current seeds; try a single forced seed
            # Pick the first uncovered and force it
            seed = all_subsets[uncovered_indices[0]]
            best_block = build_block_from(set(seed))
            best_gain = compute_new_coverage(best_block)
            if best_block is None or best_gain <= 0:
                break

        bl = sorted(best_block)
        # Check for duplicate
        if bl not in blocks:
            for s in itertools.combinations(bl, t):
                covered |= 1 << subset_to_index[s]
            blocks.append(bl)
        
        iteration += 1

    # Post-processing: Multi-pass redundant block removal
    if blocks:
        for pass_num in range(3):
            block_triplets = []
            for b in blocks:
                tps = [subset_to_index[s] for s in itertools.combinations(b, t)]
                block_triplets.append(tps)
            
            triplet_count = [0] * n_subsets
            for tps in block_triplets:
                for idx in tps:
                    triplet_count[idx] += 1
            
            new_blocks = []
            for i, b in enumerate(blocks):
                is_redundant = True
                for idx in block_triplets[i]:
                    if triplet_count[idx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(b)
            
            if len(new_blocks) == len(blocks):
                break
            blocks = new_blocks

    # Local search polish: try swapping points in blocks to improve coverage
    # A swap replaces point x in block B with point y not in B.
    # Gain = new triplets covered - triplets lost (if any become uncovered).
    # Since we're minimizing block count, we only do this if it doesn't increase blocks.
    # Actually, local search on block composition can help find tighter coverings.
    # We'll try: for each block, try replacing each point with each other point,
    # and keep the change if it increases total coverage (without adding blocks).
    
    if time.time() < deadline - 5:
        # Recompute coverage from scratch for accuracy
        covered_new = 0
        for b in blocks:
            for s in itertools.combinations(b, t):
                covered_new |= 1 << subset_to_index[s]
        covered = covered_new
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for bi, b in enumerate(blocks):
                if time.time() > deadline - 5:
                    break
                b_set = set(b)
                for x in list(b):
                    for y in range(v):
                        if y in b_set:
                            continue
                        # Try swapping x with y
                        new_b = sorted((b_set - {x}) | {y})
                        # Compute gain: new triplets covered
                        new_triplets = set(subset_to_index[s] for s in itertools.combinations(new_b, t))
                        old_triplets = set(subset_to_index[s] for s in itertools.combinations(b, t))
                        
                        # Triplets that will be lost (in old but not in new)
                        lost = old_triplets - new_triplets
                        # Triplets that will be gained (in new but not in old)
                        gained = new_triplets - old_triplets
                        
                        # Check if lost triplets are covered by other blocks
                        can_lose = True
                        for idx in lost:
                            # Count how many blocks cover this triplet (excluding current)
                            count = 0
                            for bj in blocks:
                                if bj == b:
                                    continue
                                if idx in [subset_to_index[s] for s in itertools.combinations(bj, t)]:
                                    count += 1
                                    if count >= 1:
                                        break
                            if count < 1:
                                can_lose = False
                                break
                        
                        if not can_lose:
                            continue
                        
                        # Net gain = number of gained triplets not already covered
                        net_gain = 0
                        for idx in gained:
                            if not (covered >> idx) & 1:
                                net_gain += 1
                        
                        if net_gain > 0:
                            # Apply the swap
                            blocks[bi] = new_b
                            for idx in gained:
                                covered |= 1 << idx
                            improved = True
                            break  # Restart loop for this block after modification
                if improved:
                    break

    return blocks
