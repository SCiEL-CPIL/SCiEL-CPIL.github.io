import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin under 150s kill

    # Map every t-subset to an integer id
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # Precompute masks for fast gain calculation
    # masks[x] is a dict: key is a (t-1)-tuple of points, value is a bitmask of t-subsets
    # that contain x and that (t-1)-tuple.
    masks = [dict() for _ in range(v)]
    
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i in range(t):
            others = s[:i] + s[i+1:]
            key = others 
            x = s[i]
            masks[x][key] = masks[x].get(key, 0) | bit

    covered = 0  # Bitset of covered t-subsets
    blocks = []

    # Helper: Count bits in an integer
    def popcount(n):
        return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')

    def gain_of_add(x, B_tuple):
        """
        Calculate the number of new t-subsets covered by adding x to the current block B_tuple.
        B_tuple is a tuple of points currently in the block.
        """
        g_mask = 0
        # Iterate over all (t-1) subsets of B_tuple
        for S in itertools.combinations(B_tuple, t - 1):
            key = S
            m = masks[x].get(key)
            if m is not None:
                g_mask |= m
        # Intersect with currently uncovered subsets
        return popcount(g_mask & ~covered)

    def build_block_from(seed_points):
        """
        Grow a block from initial set (seed_points) to size k by max-gain greedy.
        Returns the block as a list of ints.
        """
        B = list(seed_points)
        B_set = set(B)
        available = [x for x in range(v) if x not in B_set]
        
        while len(B) < k:
            best_x = -1
            best_gain = -1
            # Evaluate all available points
            for x in available:
                g = gain_of_add(x, tuple(B))
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_x == -1:
                break
            
            B.append(best_x)
            B_set.add(best_x)
            available.remove(best_x)
            
        return B

    def total_new_coverage(block):
        """
        Count how many t-subsets in the block are newly covered.
        """
        total = 0
        for s in itertools.combinations(sorted(block), t):
            idx = subset_to_index[s]
            if not (covered >> idx) & 1:
                total += 1
        return total

    # Main Loop: Strong Greedy with Randomized Restarting
    all_points = list(range(v))
    
    while True:
        if time.time() > deadline:
            break

        # Check if all covered
        uncov_mask = ((1 << n_subsets) - 1) & ~covered
        if uncov_mask == 0:
            break

        # Collect uncovered subset indices
        # We will use a mix of deterministic and random sampling to explore different parts of the search space
        uncovered_indices = []
        m = uncov_mask
        # First, collect a deterministic set (LSB first)
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            uncovered_indices.append(idx)
            m ^= lsb
            if len(uncovered_indices) >= 1000:
                break
        
        if not uncovered_indices:
            break

        # To improve diversity and potentially escape local minima of the greedy choice,
        # we can shuffle the indices we consider, or just iterate through them.
        # The current program iterates deterministically. Inspiration 2 suggests randomization.
        # Let's shuffle the list of indices to consider in this iteration to vary the "best" block found.
        # However, finding the *global* best among 1000 seeds is expensive.
        # A better strategy: 
        # 1. Pick a subset of seeds (e.g. 50-100)
        # 2. Build blocks from them
        # 3. Pick the best one
        # 4. If we have time, maybe do another round with different seeds?
        
        # For now, let's stick to the strong greedy: evaluate many seeds, pick the best.
        # But we can randomize the order in which we evaluate them, or just take a random subset.
        # Taking a random subset of size S from the uncovered ones allows us to cover different "areas" of the problem space.
        
        # Let's try a hybrid: 
        # - If we have plenty of time, evaluate more seeds.
        # - Randomly select a subset of the uncovered indices to evaluate to introduce stochasticity.
        
        # Number of seeds to evaluate per iteration
        # We want to balance quality (more seeds -> better block) vs speed (fewer seeds -> more iterations)
        # Let's aim for ~20-40 seeds per iteration to allow ~50-100 iterations in 112s.
        # Each build_block_from takes ~ (k-t) * v * (t-1 choose t-1) * popcount ~ 4 * 24 * 1 * fast = ~100 ops?
        # Actually gain_of_add is O(C(|B|, t-1)). For t=3, it's O(|B|). 
        # build_block_from calls gain_of_add (k-t) times, each scanning v candidates.
        # So build_block_from is O((k-t) * v * |B|). With k=7, t=3, v=24, |B| grows 3->7.
        # Roughly 4 * 24 * 5 = 480 operations per block.
        # Evaluating 50 seeds: 24000 operations. Very fast.
        # The bottleneck is likely Python overhead.
        # Let's try evaluating up to 100 seeds per iteration.
        
        # Randomly select seeds
        # If we have less than 100 uncovered, take all.
        n_seeds_to_eval = min(100, len(uncovered_indices))
        
        # To ensure we don't always pick the same "best" block due to deterministic ties or ordering,
        # we shuffle the indices we pick from.
        random.shuffle(uncovered_indices)
        
        # Pick the first n_seeds_to_eval from the shuffled list
        selected_indices = uncovered_indices[:n_seeds_to_eval]
        
        best_block = None
        best_gain_total = -1
        
        for idx in selected_indices:
            if time.time() > deadline:
                break
            
            seed = all_subsets[idx]
            B = build_block_from(list(seed))
            
            # Calculate total new coverage for this block
            total = total_new_coverage(B)
            
            if total > best_gain_total:
                best_gain_total = total
                best_block = B
        
        if best_block is None:
            break

        bl = sorted(best_block)
        
        # Check if this block is already in our list (shouldn't happen if we track covered, but safe)
        # More importantly, check if it adds ANY new coverage. If not, we are stuck.
        if best_gain_total == 0:
            # This can happen if the greedy block construction gets stuck in a loop of redundant blocks
            # or if the seeds are all already covered by other blocks but the bitset logic missed it?
            # No, we only pick uncovered seeds. But the *block* built from an uncovered seed might
            # not cover that specific seed if the greedy process changes the block structure?
            # No, the seed is a subset of the block. So the block MUST cover the seed.
            # Unless... the seed is already covered? 
            # We selected idx from uncovered_indices. So the seed t-subset is uncovered.
            # The block contains the seed. So the block covers the seed.
            # So total_new_coverage MUST be at least 1.
            # If it is 0, there is a bug.
            # Let's break to avoid infinite loop.
            break

        # Mark covered
        for s in itertools.combinations(bl, t):
            covered |= 1 << subset_to_index[s]
            
        blocks.append(bl)

    return blocks
