import itertools
import random
import time
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_bitsets(v, k, t):
    """
    Precompute:
      - all k-blocks (list of tuples)
      - for each block, a bitset (list of np.uint64) representing which t-sets it covers
      - n_tsets
    We map each t-set to an integer index 0..n_tsets-1.
    """
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Number of uint64 words needed
    n_words = (n_tsets + 63) // 64

    # block_bitsets[b] = np.array of n_words uint64
    block_bitsets = np.zeros((n_blocks, n_words), dtype=np.uint64)

    for b_idx, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            word = ts_idx >> 6
            bit = ts_idx & 63
            block_bitsets[b_idx, word] |= (np.uint64(1) << np.uint64(bit))

    # Also precompute: for each t-set index, which blocks cover it (for fast candidate generation)
    # tset_coverers[ts_idx] = list of block indices
    tset_coverers = [[] for _ in range(n_tsets)]
    for b_idx in range(n_blocks):
        words = block_bitsets[b_idx]
        for w in range(n_words):
            word_val = words[w]
            while word_val:
                # find lowest set bit
                lb = (word_val & (-word_val)).bit_length() - 1
                ts_idx = (w << 6) + lb
                if ts_idx < n_tsets:
                    tset_coverers[ts_idx].append(b_idx)
                word_val &= word_val - 1

    return all_blocks, n_blocks, n_tsets, n_words, block_bitsets, tset_coverers

def _coverage_vector(block_indices, block_bitsets, n_words):
    """Return a numpy array of uint64 words representing the union coverage."""
    cov = np.zeros(n_words, dtype=np.uint64)
    for b in block_indices:
        cov |= block_bitsets[b]
    return cov

def _count_uncovered(cov, n_words, n_tsets):
    """Count how many t-sets are NOT covered by the bitset."""
    total = 0
    for w in range(n_words):
        word = cov[w]
        # count zeros in the word
        total += 64 - np.bit_count(word)
    # Adjust for the last word if it has extra bits
    if n_tsets % 64 != 0:
        last = n_words - 1
        extra = 64 - (n_tsets % 64)
        # The extra bits in the last word are always considered covered (or we just mask)
        # Actually, we should only count the valid bits.
        # Let's recompute properly for the last word
        last_word = cov[last]
        valid_bits = n_tsets % 64
        # Mask out the invalid bits
        mask = (np.uint64(1) << np.uint64(valid_bits)) - np.uint64(1)
        last_word &= mask
        total -= 64 - np.bit_count(cov[last])  # subtract previous count
        total += valid_bits - np.bit_count(last_word)
    return total

def _is_valid(cov, n_words, n_tsets):
    """Check if all t-sets are covered."""
    return _count_uncovered(cov, n_words, n_tsets) == 0

def _simulated_annealing(all_blocks, n_blocks, n_tsets, n_words, block_bitsets, tset_coverers, rng, time_limit):
    """
    Simulated annealing over the set of blocks.
    State: a set of block indices.
    Moves: 
      1. Remove a random block.
      2. Add a random block.
      3. Swap: remove one, add one.
      4. 2-exchange: remove two, add two.
    """
    deadline = time.time() + time_limit
    
    # Initialize with a greedy solution
    # Greedy: repeatedly pick the block that covers the most uncovered t-sets
    uncovered = np.ones(n_words, dtype=np.uint64)
    # Set only the valid bits
    if n_tsets % 64 != 0:
        last = n_words - 1
        valid_bits = n_tsets % 64
        mask = (np.uint64(1) << np.uint64(valid_bits)) - np.uint64(1)
        uncovered[last] &= mask
    else:
        pass # all bits valid

    current = []
    current_cov = np.zeros(n_words, dtype=np.uint64)
    
    # Greedy initialization
    while True:
        if time.time() > deadline - 10:
            break
        uncovered_count = _count_uncovered(current_cov, n_words, n_tsets)
        if uncovered_count == 0:
            break
        
        # Find an uncovered t-set
        # Pick a random uncovered t-set
        # We need to find an uncovered t-set index
        # Let's just pick a random t-set index and check if it's uncovered
        # For efficiency, we can pick a random word and then a random bit
        best_block = -1
        best_gain = -1
        
        # To speed up, we can sample a few random uncovered t-sets
        for _ in range(10):
            ts_idx = rng.randint(0, n_tsets - 1)
            word = ts_idx >> 6
            bit = ts_idx & 63
            if not (current_cov[word] & (np.uint64(1) << np.uint64(bit))):
                # This t-set is uncovered
                # Find the block that covers this t-set and has the highest gain
                candidates = tset_coverers[ts_idx]
                for b in candidates:
                    if b in current:
                        continue
                    gain = np.bit_count(block_bitsets[b] & ~current_cov)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b
        
        if best_block == -1 or best_gain <= 0:
            # No improvement possible with this method, break
            break
        
        current.append(best_block)
        current_cov |= block_bitsets[best_block]
    
    best = current[:]
    best_uncovered = _count_uncovered(current_cov, n_words, n_tsets)
    best_size = len(current)
    
    # Annealing
    # Temperature schedule
    t_start = 1.0
    t_end = 0.001
    # We'll use a simple cooling schedule
    # Number of steps is determined by time
    step_count = 0
    total_steps = 100000  # Will be limited by time
    
    # Precompute a list of all block indices for random selection
    all_block_indices = list(range(n_blocks))
    
    while time.time() < deadline:
        # Cooling
        progress = min(1.0, time.time() - (deadline - time_limit)) / time_limit
        temperature = t_start * (t_end / t_start) ** progress
        
        # Choose a move
        move_type = rng.random()
        new_current = current[:]
        new_cov = current_cov.copy()
        
        if move_type < 0.3:
            # Remove a block
            if len(current) == 0:
                continue
            i = rng.randint(0, len(current) - 1)
            b = current[i]
            # Check if removing it keeps coverage (or use energy)
            # We'll allow removal even if it breaks coverage, but with high energy cost
            new_cov = new_cov & ~block_bitsets[b]
            new_current.pop(i)
        elif move_type < 0.6:
            # Add a block
            b = rng.choice(all_block_indices)
            if b not in current:
                new_cov = new_cov | block_bitsets[b]
                new_current.append(b)
        elif move_type < 0.85:
            # Swap
            if len(current) == 0:
                continue
            i = rng.randint(0, len(current) - 1)
            old_b = current[i]
            new_b = rng.choice(all_block_indices)
            if new_b == old_b or new_b in current:
                continue
            new_cov = new_cov & ~block_bitsets[old_b]
            new_cov = new_cov | block_bitsets[new_b]
            new_current[i] = new_b
        else:
            # 2-exchange
            if len(current) < 2:
                continue
            i, j = rng.sample(range(len(current)), 2)
            old_b1, old_b2 = current[i], current[j]
            new_b1, new_b2 = rng.sample(all_block_indices, 2)
            if new_b1 in current or new_b2 in current or new_b1 == new_b2:
                continue
            new_cov = new_cov & ~block_bitsets[old_b1]
            new_cov = new_cov & ~block_bitsets[old_b2]
            new_cov = new_cov | block_bitsets[new_b1]
            new_cov = new_cov | block_bitsets[new_b2]
            new_current[i] = new_b1
            new_current[j] = new_b2
        
        # Calculate energy (uncovered count)
        new_uncovered = _count_uncovered(new_cov, n_words, n_tsets)
        new_size = len(new_current)
        
        # We want to minimize size, but only among valid solutions.
        # If invalid, we penalize heavily.
        # Energy = size + 100 * uncovered
        old_uncovered = _count_uncovered(current_cov, n_words, n_tsets)
        old_size = len(current)
        
        old_energy = old_size + 100 * old_uncovered
        new_energy = new_size + 100 * new_uncovered
        
        delta = new_energy - old_energy
        
        # Acceptance
        if delta <= 0:
            accept = True
        else:
            # Metropolis criterion
            if temperature > 0:
                accept = rng.random() < np.exp(-delta / temperature)
            else:
                accept = False
        
        if accept:
            current = new_current
            current_cov = new_cov
            # Update best
            if new_uncovered == 0 and new_size < best_size:
                best_size = new_size
                best = current[:]
                best_uncovered = 0
        
        step_count += 1
        if step_count % 10000 == 0:
            # Periodic check
            pass
    
    return best

def _local_search_refine(blocks_idx, all_blocks, n_blocks, n_tsets, n_words, block_bitsets, tset_coverers, rng, time_limit):
    """
    Local search: try to remove redundant blocks, then try swaps to reduce size.
    """
    deadline = time.time() + time_limit
    current = list(blocks_idx)
    
    if not current:
        return current
    
    current_cov = _coverage_vector(current, block_bitsets, n_words)
    
    # First, remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            b = current[i]
            # Check if removing b keeps coverage
            # We can check if b is the only cover for any t-set
            # This is expensive, so we'll just try removing it and check
            new_cov = current_cov & ~block_bitsets[b]
            if _count_uncovered(new_cov, n_words, n_tsets) == 0:
                current.pop(i)
                current_cov = new_cov
                changed = True
                # Restart the loop since indices changed
                break
    
    # Now try swaps to reduce size further
    # This is tricky because swapping doesn't change size.
    # We need to find a way to remove a block and add a block that covers more.
    # Or, we can try to find a block that is "less useful" and replace it with one that is "more useful",
    # but since we already removed redundant blocks, all blocks are necessary.
    # So local search here is mainly about redundancy removal.
    
    return current

def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0  # Safety margin before 150s limit
    
    # Precompute
    all_blocks, n_blocks, n_tsets, n_words, block_bitsets, tset_coverers = _precompute_bitsets(v, k, t)
    
    # Run simulated annealing with multiple seeds
    best_blocks = None
    best_size = float('inf')
    
    # We'll run SA for a fixed amount of time per seed
    n_seeds = 5
    time_per_seed = (total_budget - 10.0) / n_seeds
    
    for seed in range(n_seeds):
        if time.time() - start_time > total_budget - 10.0:
            break
        rng = random.Random(seed)
        blocks_idx = _simulated_annealing(
            all_blocks, n_blocks, n_tsets, n_words, block_bitsets, tset_coverers, rng, time_per_seed
        )
        
        if len(blocks_idx) < best_size:
            best_size = len(blocks_idx)
            best_blocks = blocks_idx[:]
    
    if best_blocks is None:
        # Fallback
        rng = random.Random(42)
        best_blocks = _simulated_annealing(
            all_blocks, n_blocks, n_tsets, n_words, block_bitsets, tset_coverers, rng, 30.0
        )
        best_size = len(best_blocks)
    
    # Local search refinement
    if time.time() - start_time < total_budget - 5.0:
        rng = random.Random(123)
        remaining_time = min(30.0, total_budget - (time.time() - start_time) - 5.0)
        improved = _local_search_refine(
            best_blocks, all_blocks, n_blocks, n_tsets, n_words, block_bitsets, tset_coverers, rng, remaining_time
        )
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)
    
    # Final verification
    # Rebuild coverage and check
    final_cov = _coverage_vector(best_blocks, block_bitsets, n_words)
    if _count_uncovered(final_cov, n_words, n_tsets) != 0:
        # Something went wrong; rebuild with a simple greedy
        rng = random.Random(789)
        best_blocks = _simulated_annealing(
            all_blocks, n_blocks, n_tsets, n_words, block_bitsets, tset_coverers, rng, 20.0
        )
    
    # Convert to list of lists
    result = [list(all_blocks[idx]) for idx in best_blocks]
    
    return result

# EVOLVE-BLOCK-END
