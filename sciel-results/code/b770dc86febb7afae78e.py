import itertools
import random
import time
import math

def construct(v, k, t):
    # Precompute all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_indices = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute for each t-subset, the list of k-subsets (blocks) that contain it
    # For v=24, k=7, t=3: C(24,7)=346104 blocks, C(24,3)=2024 t-subsets
    # Each t-subset is in C(v-t, k-t) = C(21, 4) = 5985 blocks
    # Total pairs: 2024 * 5985 = ~12M, manageable
    
    # Instead of storing all blocks, generate candidates on-the-fly
    # But for local search, we need to efficiently find blocks containing a t-subset
    
    def generate_blocks_containing(t_sub):
        """Generate all k-subsets containing the given t-subset."""
        t_set = set(t_sub)
        remaining = [x for x in range(v) if x not in t_set]
        for ext in itertools.combinations(remaining, k - t):
            yield tuple(sorted(t_sub + ext))
    
    def count_t_subsets(block):
        """Count how many t-subsets are in this block."""
        return math.comb(k, t)
    
    def initial_random_covering(num_blocks):
        """Create a random covering with approximately num_blocks blocks."""
        blocks = set()
        covered = [0] * num_t
        
        # First, ensure all t-subsets are covered by adding blocks
        uncovered = list(range(num_t))
        random.shuffle(uncovered)
        
        for idx in uncovered:
            ts = all_t_subsets[idx]
            # Pick a random block containing this t-subset
            candidates = list(generate_blocks_containing(ts))
            block = random.choice(candidates)
            blocks.add(block)
            
            # Update coverage
            for tsub in itertools.combinations(block, t):
                t_idx = t_subset_indices[tsub]
                if covered[t_idx] == 0:
                    covered[t_idx] = 1
        
        return list(blocks)
    
    def is_valid_covering(blocks):
        """Check if the blocks form a valid covering."""
        covered = [0] * num_t
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                covered[t_subset_indices[tsub]] = 1
        return all(c == 1 for c in covered)
    
    def get_uncovered_t_subsets(blocks):
        """Return list of uncovered t-subset indices."""
        covered = [0] * num_t
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                covered[t_subset_indices[tsub]] = 1
        return [i for i in range(num_t) if covered[i] == 0]
    
    def try_remove_block(blocks, block_idx):
        """Try to remove block at block_idx. Returns True if removed."""
        if len(blocks) <= 1:
            return False
        
        # Check if removing this block still leaves a valid covering
        new_blocks = blocks[:block_idx] + blocks[block_idx+1:]
        
        # Quick check: count how many t-subsets are only covered by this block
        # If any t-subset is only covered by this block, we can't remove it
        covered_count = [0] * num_t
        for i, block in enumerate(blocks):
            if i == block_idx:
                continue
            for tsub in itertools.combinations(block, t):
                covered_count[t_subset_indices[tsub]] += 1
        
        for tsub in itertools.combinations(blocks[block_idx], t):
            t_idx = t_subset_indices[tsub]
            if covered_count[t_idx] == 0:
                return False
        
        # Remove it
        del blocks[block_idx]
        return True
    
    def local_search(blocks, max_iter=5000):
        """Perform local search to minimize number of blocks."""
        best_blocks = list(blocks)
        best_len = len(best_blocks)
        current_blocks = list(blocks)
        
        # Track coverage counts for efficiency
        def compute_coverage_counts(blks):
            counts = [0] * num_t
            for block in blks:
                for tsub in itertools.combinations(block, t):
                    counts[t_subset_indices[tsub]] += 1
            return counts
        
        for iteration in range(max_iter):
            changed = False
            
            # Try to remove a random block
            if len(current_blocks) > 1:
                idx = random.randrange(len(current_blocks))
                # Check if we can remove it
                counts = compute_coverage_counts(current_blocks)
                can_remove = True
                for tsub in itertools.combinations(current_blocks[idx], t):
                    t_idx = t_subset_indices[tsub]
                    if counts[t_idx] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    del current_blocks[idx]
                    changed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
            
            # Try to swap: remove one block, add another that covers more uncovered
            if len(current_blocks) > 2:
                # Pick a random block to potentially replace
                remove_idx = random.randrange(len(current_blocks))
                
                # Find t-subsets only covered by this block
                counts = compute_coverage_counts(current_blocks)
                essential_ts = []
                for tsub in itertools.combinations(current_blocks[remove_idx], t):
                    t_idx = t_subset_indices[tsub]
                    if counts[t_idx] == 1:
                        essential_ts.append(tsub)
                
                if essential_ts:
                    # We need a block that covers all essential_ts
                    # Find blocks that contain all essential t-subsets
                    # This is expensive, so we'll skip for now
                    pass
                else:
                    # No essential t-subsets, try to replace with a better block
                    # Find a block that covers more uncovered t-subsets
                    uncovered = [i for i in range(num_t) if counts[i] == 0]
                    if uncovered:
                        # Pick a random uncovered t-subset
                        target_idx = random.choice(uncovered)
                        target_ts = all_t_subsets[target_idx]
                        
                        # Generate candidate blocks containing target_ts
                        candidates = list(generate_blocks_containing(target_ts))
                        if candidates:
                            # Pick the one that covers the most uncovered t-subsets
                            best_candidate = None
                            best_gain = -1
                            for cand in random.sample(candidates, min(50, len(candidates))):
                                gain = 0
                                for tsub in itertools.combinations(cand, t):
                                    t_idx = t_subset_indices[tsub]
                                    if counts[t_idx] == 0:
                                        gain += 1
                                if gain > best_gain:
                                    best_gain = gain
                                    best_candidate = cand
                            
                            if best_candidate is not None:
                                new_blocks = current_blocks[:remove_idx] + current_blocks[remove_idx+1:]
                                new_blocks.append(best_candidate)
                                
                                # Verify validity
                                if is_valid_covering(new_blocks):
                                    current_blocks = new_blocks
                                    changed = True
                                    if len(current_blocks) < best_len:
                                        best_len = len(current_blocks)
                                        best_blocks = list(current_blocks)
            
            if not changed and iteration > 100:
                # Stuck, try a bigger perturbation
                # Randomly replace a few blocks
                if len(current_blocks) > 5:
                    for _ in range(3):
                        idx = random.randrange(len(current_blocks))
                        counts = compute_coverage_counts(current_blocks)
                        
                        # Find uncovered t-subsets
                        uncovered = [i for i in range(num_t) if counts[i] == 0]
                        if not uncovered:
                            # Try to remove
                            can_remove = True
                            for tsub in itertools.combinations(current_blocks[idx], t):
                                t_idx = t_subset_indices[tsub]
                                if counts[t_idx] <= 1:
                                    can_remove = False
                                    break
                            if can_remove:
                                del current_blocks[idx]
                                if len(current_blocks) < best_len:
                                    best_len = len(current_blocks)
                                    best_blocks = list(current_blocks)
                        else:
                            # Replace with a block covering an uncovered t-subset
                            target_idx = random.choice(uncovered)
                            target_ts = all_t_subsets[target_idx]
                            candidates = list(generate_blocks_containing(target_ts))
                            if candidates:
                                new_block = random.choice(candidates)
                                current_blocks[idx] = new_block
                                if is_valid_covering(current_blocks) and len(current_blocks) < best_len:
                                    best_len = len(current_blocks)
                                    best_blocks = list(current_blocks)
                
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = list(current_blocks)
        
        return best_blocks
    
    # Start with a reasonably sized covering
    start_time = time.time()
    best_blocks = None
    best_len = float('inf')
    
    # Initial guess: start with ~80 blocks
    initial_size = 85
    
    # Run multiple local search instances
    while time.time() - start_time < 110:
        # Create initial random covering
        blocks = initial_random_covering(initial_size)
        
        # Local search to minimize
        improved = local_search(blocks, max_iter=2000)
        
        if len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved
            if best_len <= 69:
                break
    
    if best_blocks is None:
        # Fallback
        blocks = initial_random_covering(initial_size)
        best_blocks = local_search(blocks, max_iter=1000)
    
    return best_blocks
