import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # seconds
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered_initial = set(all_t_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    def get_gain(block, uncovered_set):
        """Calculate the number of uncovered t-subsets covered by this block."""
        gain = 0
        for combo in itertools.combinations(block, t):
            if combo in uncovered_set:
                gain += 1
        return gain

    def run_trial(trial_idx, num_candidates=50, alpha=1.0, use_max=False):
        """
        Run a single stochastic greedy trial.
        """
        uncovered = set(uncovered_initial)
        blocks = []
        rng = random.Random(trial_idx * 12345 + 67890)
        
        while uncovered:
            if time.time() - start_time > time_limit - 2:
                return None, False
            
            # 1. Pick a random uncovered subset to anchor candidates
            # This ensures we don't get stuck in a region where no candidates are generated for hard subsets
            anchor = rng.choice(list(uncovered))
            
            # 2. Generate candidate blocks containing the anchor
            candidates = []
            # We need k - t more points from the remaining v - t points
            other_points = [x for x in range(v) if x not in anchor]
            
            for _ in range(num_candidates):
                # Randomly pick k-t points from other_points
                # Note: anchor is a tuple, convert to set for fast lookup if needed, 
                # but here we just combine.
                extra = rng.sample(other_points, k - t)
                # Combine anchor and extra, then sort to create a canonical block
                block = tuple(sorted(anchor + extra))
                candidates.append(block)
            
            # 3. Calculate gains for candidates
            gains = []
            for block in candidates:
                g = get_gain(block, uncovered)
                gains.append(g)
            
            # 4. Select a block
            if not candidates:
                # Should not happen
                break
                
            if use_max:
                max_gain = max(gains)
                # Pick one of the max gain blocks randomly
                max_indices = [i for i, g in enumerate(gains) if g == max_gain]
                idx = rng.choice(max_indices)
            else:
                # Weighted selection
                # Filter out zero gains to avoid wasting steps, 
                # but if all are 0 (shouldn't happen if anchor is uncovered), pick random
                non_zero = [(i, g) for i, g in enumerate(gains) if g > 0]
                if not non_zero:
                    idx = rng.randint(0, len(candidates) - 1)
                else:
                    # Use weights proportional to g^alpha
                    indices = [i for i, g in non_zero]
                    weights = [g**alpha for i, g in non_zero]
                    idx = rng.choices(indices, weights=weights, k=1)[0]
            
            selected_block = candidates[idx]
            
            # 5. Add block and update uncovered
            blocks.append(list(selected_block))
            for combo in itertools.combinations(selected_block, t):
                uncovered.discard(combo)
                
        return blocks, True

    # Strategy: Run many trials with varying parameters
    # We can increase num_candidates if time permits, or vary alpha.
    
    trial_count = 0
    while time.time() - start_time < time_limit - 5:
        # Vary parameters to explore different parts of the solution space
        # Use trial_count to determine parameters
        num_cands = 30 + (trial_count % 5) * 10  # 30, 40, 50, 60, 70
        alpha = 1.0 + (trial_count % 3) * 0.5    # 1.0, 1.5, 2.0
        use_max = (trial_count % 10 == 0)        # 1 in 10 is pure greedy
        
        blocks, success = run_trial(trial_count, num_cands, alpha, use_max)
        
        if success and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # Optional: print or log improvement
            # print(f"Trial {trial_count}: Found cover with {len(blocks)} blocks")
            
        trial_count += 1
        
    # Fallback
    if best_blocks is None:
        # Simple greedy fallback
        uncovered = set(uncovered_initial)
        blocks = []
        rng = random.Random(0)
        while uncovered:
            anchor = min(uncovered) # Deterministic
            other_points = [x for x in range(v) if x not in anchor]
            # Just pick one random completion
            extra = rng.sample(other_points, k - t)
            block = tuple(sorted(anchor + extra))
            blocks.append(list(block))
            for combo in itertools.combinations(block, t):
                uncovered.discard(combo)
        best_blocks = blocks
        
    return best_blocks
