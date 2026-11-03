import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 140.0  # Safety margin for 150s kill
    
    # Precompute all t-subsets (triples)
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    triple_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # State
    covered = np.zeros(n_t_subsets, dtype=bool)
    selected_blocks = []
    selected_tuples = set()
    
    # Helper: Get list of uncovered triples
    def get_uncovered():
        return np.where(~covered)[0]
    
    # Helper: Generate a block that maximally covers a specific uncovered triple
    # and generally covers many uncovered triples.
    def generate_adaptive_block(rng):
        # Pick a random uncovered triple to anchor the block
        unc = get_uncovered()
        if len(unc) == 0:
            return None
        anchor_idx = rng.choice(unc)
        anchor_triple = all_t_subsets[anchor_idx]
        
        block = list(anchor_triple)
        block_set = set(block)
        
        # Greedily fill the rest of the block
        # We want to maximize the number of *newly* covered triples.
        # A triple is covered if it is a subset of the block.
        # When adding point p to block B, the new triples covered are those
        # of the form (p, x, y) where {x, y} subset of B.
        
        remaining_points = [x for x in range(v) if x not in block_set]
        
        # Precompute subsets of current block for fast gain calculation
        # Since k is small (7), we can just recompute or maintain.
        
        while len(block) < k:
            best_gain = -1
            best_point = -1
            
            # Current (t-1)-subsets of the block
            # For t=3, we look at pairs in the block.
            block_subsets = list(itertools.combinations(block, t-1))
            
            for p in remaining_points:
                gain = 0
                for sub in block_subsets:
                    # Form new t-subset
                    new_t_sub = tuple(sorted(sub + (p,)))
                    # Check if it's uncovered
                    if new_t_sub in triple_to_idx:
                        idx = triple_to_idx[new_t_sub]
                        if not covered[idx]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
                    
            if best_point == -1:
                # No gain, pick random
                best_point = rng.choice(remaining_points)
            
            block.append(best_point)
            block_set.add(best_point)
            remaining_points.remove(best_point)
            
        return tuple(sorted(block))

    # Main Construction Loop: Adaptive Greedy
    # Instead of a fixed pool, we generate blocks one by one,
    # ensuring each new block covers at least one currently uncovered triple.
    
    rng = random.Random(12345)
    
    # To improve quality, we can try multiple "candidates" for the next block
    # and pick the one with the highest total gain.
    # Given time constraints, we'll generate a small batch of candidates per step.
    
    max_steps = 200 # Safety limit on number of blocks
    step = 0
    
    while time.time() < deadline - 20.0:
        unc = get_uncovered()
        if len(unc) == 0:
            break
        if step >= max_steps:
            break
            
        # Generate a batch of candidate blocks
        # The size of the batch depends on remaining time.
        # Early on, we can afford more. Later, we need to be faster.
        batch_size = 20 if step < 20 else 10
        
        best_candidate = None
        best_gain_total = -1
        
        for _ in range(batch_size):
            cand = generate_adaptive_block(rng)
            if cand is None:
                continue
            
            # Calculate total gain of this candidate
            # Gain = number of triples in cand that are currently uncovered
            gain = 0
            for sub in itertools.combinations(cand, t):
                idx = triple_to_idx[sub]
                if not covered[idx]:
                    gain += 1
            
            if gain > best_gain_total:
                best_gain_total = gain
                best_candidate = cand
                
        if best_candidate is None:
            break
            
        # Select the best candidate
        selected_blocks.append(list(best_candidate))
        selected_tuples.add(best_candidate)
        
        # Update covered state
        for sub in itertools.combinations(best_candidate, t):
            covered[triple_to_idx[sub]] = True
            
        step += 1
        
    # If we didn't cover everything (should be rare with adaptive greedy),
    # fall back to a simple greedy fill or accept the result if close.
    # Actually, the adaptive greedy *should* cover everything because it always
    # picks a block covering an uncovered triple.
    
    # Local Search Phase: Try to remove or swap blocks to reduce count.
    # We have time for this if construction was fast.
    
    if time.time() < deadline - 5.0:
        # 1-Swap Local Search
        # Try to remove a block if all its triples are covered by other blocks.
        # Then try to swap a block with another that covers more unique triples?
        # Or simply: iterate through blocks, check if redundant.
        
        # Precompute triple indices for each selected block for fast checking
        # n_selected is small (~70-80)
        
        # Redundancy Check
        # For each block, check if removing it leaves any triple uncovered.
        # This is expensive if done naively.
        # Better: Count coverage frequency for each triple.
        # If a triple is covered by only 1 block, that block is essential.
        
        # Let's build a coverage count array.
        coverage_counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b in selected_blocks:
            for sub in itertools.combinations(b, t):
                coverage_counts[triple_to_idx[sub]] += 1
                
        # A block is redundant if all its triples have count > 1.
        # If we remove a redundant block, counts decrease by 1 for its triples.
        # We need to check if any triple drops to 0.
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Iterate in reverse to remove indices safely
            for i in range(len(selected_blocks) - 1, -1, -1):
                b = selected_blocks[i]
                # Check if b is redundant
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    idx = triple_to_idx[sub]
                    if coverage_counts[idx] == 1:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    # Remove b
                    # Update coverage counts
                    for sub in itertools.combinations(b, t):
                        coverage_counts[triple_to_idx[sub]] -= 1
                    selected_blocks.pop(i)
                    improved = True
                    # Break to restart loop as indices changed?
                    # Or continue? If we continue, we might remove another redundant one.
                    # But removing one might make another non-redundant.
                    # So we must break and restart the scan.
                    break
        
        # Optional: 1-Swap (Replace block i with block j)
        # This is more complex and might not yield gains in short time.
        # Given the "adaptive greedy" usually produces near-optimal counts for this size,
        # redundancy removal is the most effective quick win.
        
    # Final Validation
    # Ensure all triples are covered
    if not covered.all():
        # This should not happen, but if it does, we have an invalid solution.
        # We should try to fix it quickly.
        unc = get_uncovered()
        if len(unc) > 0:
            # Quick fix: add random blocks until covered
            while len(unc) > 0 and time.time() < deadline - 1.0:
                # Pick a random uncovered triple
                idx = rng.choice(unc)
                anchor = all_t_subsets[idx]
                # Generate a block around it
                b = generate_adaptive_block(rng)
                if b:
                    if b not in selected_tuples:
                        selected_blocks.append(list(b))
                        selected_tuples.add(b)
                        for sub in itertools.combinations(b, t):
                            covered[triple_to_idx[sub]] = True
                unc = get_uncovered()
                
    return selected_blocks

# Fallback if the main logic fails or times out catastrophically
def _fallback_greedy(v, k, t, deadline):
    """
    Standard greedy if the pool strategy fails or times out early.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline - 2:
        # Pick a random uncovered subset
        start_subset = random.choice(list(uncovered))
        block = list(start_subset)
        block_set = set(block)
        
        while len(block) < k:
            best_gain = -1
            best_point = -1
            candidates = [x for x in range(v) if x not in block_set]
            
            if len(block) >= t - 1:
                block_subsets = set(itertools.combinations(block, t - 1))
            else:
                block_subsets = set()
                
            for x in candidates:
                count = 0
                for sub in block_subsets:
                    new_t_sub = tuple(sorted(sub + (x,)))
                    if new_t_sub in uncovered:
                        count += 1
                if count > best_gain:
                    best_gain = count
                    best_point = x
                    
            if best_point == -1:
                best_point = random.choice(candidates) if candidates else -1
                if best_point == -1: break
                
            block.append(best_point)
            block_set.add(best_point)
            
        if len(block) == k:
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
    return blocks
