<<<<<<< SEARCH
"""Covering design C(v,k,t) via Max-SAT-style stochastic local search (WalkSAT-like)."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s kill

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # For each point x and each (t-1)-subset of other points, store the bitmask of
    # t-subsets that contain x and those others.
    masks = [dict() for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i, x in enumerate(s):
            others = tuple(s[:i] + s[i+1:])
            masks[x][others] = masks[x].get(others, 0) | bit

    def count_bits(n):
        return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')

    # Build a block from seed points using max-gain greedy
    def build_block_from(seed_points, covered_mask):
        B = set(seed_points)
        available = [x for x in range(v) if x not in B]
        while len(B) < k:
            best_x = -1
            best_gain = -1
            B_list = list(B)
            for x in available:
                g = 0
                for S in itertools.combinations(B_list, t - 1):
                    key = tuple(S)
                    m = masks[x].get(key)
                    if m is not None:
                        g += count_bits(m & ~covered_mask)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            B.add(best_x)
            available.remove(best_x)
        return B

    # Run initial greedy to get a starting solution
    def run_initial_greedy(rng_seed):
        rng = random.Random(rng_seed)
        cov = 0
        blks = []
        full_mask = (1 << n_subsets) - 1

        while True:
            if time.time() > deadline - 20.0:
                break
            uncov_mask = full_mask & ~cov
            if uncov_mask == 0:
                break

            # Find uncovered indices
            uncovered_indices = []
            m = uncov_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                uncovered_indices.append(idx)
                m ^= lsb

            if not uncovered_indices:
                break

            # Try multiple seeds to find best block
            sample_size = min(50, len(uncovered_indices))
            sampled_indices = rng.sample(uncovered_indices, sample_size)

            best_block = None
            best_gain = -1

            for idx in sampled_indices:
                seed = all_subsets[idx]
                B = build_block_from(set(seed), cov)
                if len(B) < k:
                    continue
                # Count new coverage
                new_cov = 0
                bl = sorted(B)
                for s in itertools.combinations(bl, t):
                    si = subset_to_index[s]
                    if not (cov >> si) & 1:
                        new_cov += 1
                if new_cov > best_gain:
                    best_gain = new_cov
                    best_block = B

            if best_block is None:
                break

            bl = sorted(best_block)
            for s in itertools.combinations(bl, t):
                cov |= 1 << subset_to_index[s]
            blks.append(bl)

        return blks, cov

    # Get initial solution
    initial_blocks, initial_cov = run_initial_greedy(42)
    best_blocks = list(initial_blocks)
    best_count = len(best_blocks)

    # Max-SAT style local search: try to remove or swap blocks to reduce count
    # while maintaining full coverage
    def is_valid_covering(blocks):
        """Check if blocks cover all t-subsets."""
        cov = 0
        for b in blocks:
            for s in itertools.combinations(sorted(b), t):
                cov |= 1 << subset_to_index[s]
        return cov == (1 << n_subsets) - 1

    def try_remove_block(blocks, idx, deadline):
        """Try removing block idx; return True if still valid."""
        if len(blocks) <= 1:
            return False
        # Quick check: does this block have any uniquely covered triplet?
        # Build coverage count
        n = len(blocks)
        # For efficiency, just test removal directly
        test_blocks = blocks[:idx] + blocks[idx+1:]
        cov = 0
        for b in test_blocks:
            for s in itertools.combinations(sorted(b), t):
                cov |= 1 << subset_to_index[s]
        return cov == (1 << n_subsets) - 1

    def try_swap_block(blocks, idx, new_block, deadline):
        """Try replacing block idx with new_block; return True if valid and no worse."""
        test_blocks = blocks[:idx] + [new_block] + blocks[idx+1:]
        cov = 0
        for b in test_blocks:
            for s in itertools.combinations(sorted(b), t):
                cov |= 1 << subset_to_index[s]
        return cov == (1 << n_subsets) - 1

    # Local search: iteratively try to remove blocks
    rng = random.Random(123)
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        n_blocks = len(best_blocks)
        # Shuffle order for variety
        order = list(range(n_blocks))
        rng.shuffle(order)
        for idx in order:
            if time.time() > deadline - 10.0:
                break
            if try_remove_block(best_blocks, idx, deadline):
                best_blocks = best_blocks[:idx] + best_blocks[idx+1:]
                best_count = len(best_blocks)
                improved = True
                # Restart the pass since indices shifted
                break

    # Additional local search: try swapping blocks with better alternatives
    if time.time() < deadline - 5.0:
        for _iter in range(5):
            if time.time() > deadline - 5.0:
                break
            n_blocks = len(best_blocks)
            order = list(range(n_blocks))
            rng.shuffle(order)
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                # Try to find a replacement block that covers at least as much
                # but allows removal of another block
                current_block = best_blocks[idx]
                # Find triplets uniquely covered by this block
                # Skip complex swap for now; just do removal passes
                pass

    # Final cleanup: multiple removal passes
    for _pass in range(5):
        if time.time() > deadline - 2.0:
            break
        changed = False
        for idx in range(len(best_blocks) - 1, -1, -1):
            if time.time() > deadline - 2.0:
                break
            if try_remove_block(best_blocks, idx, deadline):
                best_blocks = best_blocks[:idx] + best_blocks[idx+1:]
                changed = True
        if not changed:
            break

    return best_blocks
=======
"""Covering design C(v,k,t) via NumPy-accelerated greedy + simulated annealing."""
import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin under 150s kill

    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # NumPy uint64 array for coverage: n_subsets / 64 words
    n_words = (n_subsets + 63) // 64
    full_mask_np = np.ones(n_words, dtype=np.uint64)
    # Last word may have extra bits; mask them off
    extra = n_words * 64 - n_subsets
    if extra > 0:
        full_mask_np[-1] = (1 << (64 - extra)) - 1

    # Precompute for each block (as a frozenset) its coverage bitmask in numpy
    # We'll compute on-the-fly but cache

    # For each point x and each (t-1)-subset of other points, store the bitmask of
    # t-subsets that contain x and those others (as a list of (word_idx, bit_mask))
    masks = [dict() for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        word_idx = idx >> 6
        bit = 1 << (idx & 63)
        for i, x in enumerate(s):
            others = tuple(s[:i] + s[i+1:])
            if others not in masks[x]:
                masks[x][others] = []
            masks[x][others].append((word_idx, bit))

    def count_bits_list(lst):
        """Count total bits in a list of (word_idx, bit) pairs."""
        total = 0
        for _, b in lst:
            total += 1
        return total

    def count_bits_np(mask_np):
        """Count bits set in a numpy uint64 array."""
        return int(np.sum([bin(x).count('1') for x in mask_np]))

    # Build a block from seed points using max-gain greedy
    def build_block_from(seed_points, covered_np):
        B = set(seed_points)
        available = [x for x in range(v) if x not in B]
        while len(B) < k:
            best_x = -1
            best_gain = -1
            best_x2 = -1  # for tie-breaking
            B_list = sorted(B)
            for x in available:
                g = 0
                for S in itertools.combinations(B_list, t - 1):
                    key = tuple(S)
                    lst = masks[x].get(key)
                    if lst is not None:
                        for (wi, bit) in lst:
                            if not (covered_np[wi] & bit):
                                g += 1
                if g > best_gain:
                    best_gain = g
                    best_x = x
                    best_x2 = x
                elif g == best_gain and g > 0:
                    # Tie-break: prefer point that appears in more uncovered triplets overall
                    # (approximate: just pick randomly among ties for diversity)
                    if random.random() < 0.5:
                        best_x = x
            if best_x == -1:
                break
            B.add(best_x)
            available.remove(best_x)
        return B

    def block_to_np(block):
        """Convert a block (list of ints) to its numpy coverage bitmask."""
        mask = np.zeros(n_words, dtype=np.uint64)
        for s in itertools.combinations(sorted(block), t):
            si = subset_to_index[s]
            mask[si >> 6] |= (1 << (si & 63))
        return mask

    def is_full(mask_np):
        """Check if mask_np equals full_mask_np."""
        return bool(np.array_equal(mask_np, full_mask_np))

    def uncovered_count(mask_np):
        """Count number of uncovered subsets."""
        diff = full_mask_np & ~mask_np
        return int(np.sum([bin(x).count('1') for x in diff]))

    # Run initial greedy to get a starting solution
    def run_initial_greedy(rng_seed):
        rng = random.Random(rng_seed)
        cov = np.zeros(n_words, dtype=np.uint64)
        blks = []

        while time.time() < deadline - 25.0:
            uncov = full_mask_np & ~cov
            if not np.any(uncov):
                break

            # Find uncovered indices
            uncovered_indices = []
            for wi in range(n_words):
                m = int(uncov[wi])
                while m:
                    lsb = m & (-m)
                    idx = (wi << 6) + (lsb.bit_length() - 1)
                    if idx < n_subsets:
                        uncovered_indices.append(idx)
                    m ^= lsb

            if not uncovered_indices:
                break

            # Try multiple seeds to find best block
            sample_size = min(30, len(uncovered_indices))
            sampled_indices = rng.sample(uncovered_indices, sample_size)

            best_block = None
            best_gain = -1

            for idx in sampled_indices:
                seed = all_subsets[idx]
                B = build_block_from(set(seed), cov)
                if len(B) < k:
                    continue
                # Count new coverage
                bl = sorted(B)
                new_cov = 0
                for s in itertools.combinations(bl, t):
                    si = subset_to_index[s]
                    if not (cov[si >> 6] & (1 << (si & 63))):
                        new_cov += 1
                if new_cov > best_gain:
                    best_gain = new_cov
                    best_block = B

            if best_block is None:
                break

            bl = sorted(best_block)
            for s in itertools.combinations(bl, t):
                si = subset_to_index[s]
                cov[si >> 6] |= (1 << (si & 63))
            blks.append(bl)

        return blks, cov

    # Get initial solution
    initial_blocks, initial_cov = run_initial_greedy(42)
    best_blocks = list(initial_blocks)
    best_count = len(best_blocks)

    # Simulated annealing + removal passes
    rng = random.Random(123)

    def try_remove_block(blocks, idx):
        """Try removing block idx; return True if still valid."""
        if len(blocks) <= 1:
            return False
        # Compute coverage without this block
        cov = np.zeros(n_words, dtype=np.uint64)
        for i, b in enumerate(blocks):
            if i == idx:
                continue
            for s in itertools.combinations(sorted(b), t):
                si = subset_to_index[s]
                cov[si >> 6] |= (1 << (si & 63))
        return is_full(cov)

    def removal_pass(blocks):
        """Do one pass of removal; return modified blocks."""
        n = len(blocks)
        order = list(range(n))
        rng.shuffle(order)
        for idx in order:
            if time.time() > deadline - 15.0:
                break
            if try_remove_block(blocks, idx):
                blocks = blocks[:idx] + blocks[idx+1:]
                # Restart since indices shifted
                break
        return blocks

    # Phase 1: Aggressive removal passes
    for _ in range(10):
        if time.time() > deadline - 20.0:
            break
        old_len = len(best_blocks)
        best_blocks = removal_pass(best_blocks)
        if len(best_blocks) == old_len:
            break
        best_count = len(best_blocks)

    # Phase 2: Simulated annealing - accept temporary increases to escape local optima
    if time.time() < deadline - 15.0:
        current_blocks = list(best_blocks)
        current_count = len(current_blocks)
        T = 5.0  # initial temperature
        T_min = 0.1
        cooling = 0.995

        while time.time() < deadline - 15.0 and T > T_min:
            n_blocks = len(current_blocks)
            if n_blocks == 0:
                break

            # Randomly pick an operation
            op = rng.random()
            if op < 0.6:
                # Try removing a random block
                idx = rng.randint(0, n_blocks - 1)
                test_blocks = current_blocks[:idx] + current_blocks[idx+1:]
                # Check coverage
                cov = np.zeros(n_words, dtype=np.uint64)
                for b in test_blocks:
                    for s in itertools.combinations(sorted(b), t):
                        si = subset_to_index[s]
                        cov[si >> 6] |= (1 << (si & 63))
                if is_full(cov):
                    # Accept (improvement)
                    current_blocks = test_blocks
                    current_count = len(current_blocks)
                    if current_count < best_count:
                        best_blocks = list(current_blocks)
                        best_count = current_count
                else:
                    # Count uncovered
                    unc = uncovered_count(cov)
                    if unc <= 3:
                        # Accept with probability based on temperature
                        if rng.random() < np.exp(-unc / T):
                            current_blocks = test_blocks
                            current_count = len(current_blocks)
            elif op < 0.85:
                # Try swapping a block with a new random block
                idx = rng.randint(0, n_blocks - 1)
                # Generate a random block
                new_block = sorted(rng.sample(range(v), k))
                test_blocks = current_blocks[:idx] + [new_block] + current_blocks[idx+1:]
                cov = np.zeros(n_words, dtype=np.uint64)
                for b in test_blocks:
                    for s in itertools.combinations(sorted(b), t):
                        si = subset_to_index[s]
                        cov[si >> 6] |= (1 << (si & 63))
                if is_full(cov):
                    current_blocks = test_blocks
                    if current_count < best_count:
                        best_blocks = list(current_blocks)
                        best_count = current_count
                else:
                    unc = uncovered_count(cov)
                    if unc <= 3:
                        if rng.random() < np.exp(-unc / T):
                            current_blocks = test_blocks
            else:
                # Try adding a block (to escape) then removing two
                if n_blocks >= 2:
                    # Pick two blocks to potentially remove
                    i1, i2 = rng.sample(range(n_blocks), 2)
                    # Generate a new block that covers some uncovered parts
                    # Just use a random block for now
                    new_block = sorted(rng.sample(range(v), k))
                    test_blocks = current_blocks[:i1] + current_blocks[i1+1:i2] + \
                                  current_blocks[i2+1:] + [new_block]
                    cov = np.zeros(n_words, dtype=np.uint64)
                    for b in test_blocks:
                        for s in itertools.combinations(sorted(b), t):
                            si = subset_to_index[s]
                            cov[si >> 6] |= (1 << (si & 63))
                    if is_full(cov):
                        current_blocks = test_blocks
                        current_count = len(current_blocks)
                        if current_count < best_count:
                            best_blocks = list(current_blocks)
                            best_count = current_count

            T *= cooling

    # Phase 3: Final aggressive removal passes on best solution
    for _ in range(15):
        if time.time() > deadline - 3.0:
            break
        old_len = len(best_blocks)
        best_blocks = removal_pass(best_blocks)
        if len(best_blocks) == old_len:
            break
        best_count = len(best_blocks)

    # Phase 4: More removal passes with different random orderings
    for _ in range(20):
        if time.time() > deadline - 1.0:
            break
        n = len(best_blocks)
        if n == 0:
            break
        order = list(range(n))
        random.shuffle(order)
        improved = False
        for idx in order:
            if time.time() > deadline - 1.0:
                break
            if try_remove_block(best_blocks, idx):
                best_blocks = best_blocks[:idx] + best_blocks[idx+1:]
                best_count = len(best_blocks)
                improved = True
                break
        if not improved:
            break

    return best_blocks
>>>>>>> REPLACE
