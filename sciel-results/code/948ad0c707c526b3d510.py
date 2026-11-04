import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute incidence structures
    # block_to_subsets[b_idx]: list of subset indices covered by block b_idx
    # subset_to_blocks[s_idx]: list of block indices covering subset s_idx
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for faster indexing where beneficial
    # But keep lists for iteration over sparse sets
    block_to_subsets_arr = [np.array(s, dtype=np.int32) for s in block_to_subsets]
    subset_to_blocks_arr = [np.array(s, dtype=np.int32) for s in subset_to_blocks]
    
    # Precompute block sizes (number of t-subsets covered)
    block_sizes = np.array([len(s) for s in block_to_subsets], dtype=np.int32)

    def get_gains(cover_count, uncovered_mask):
        """
        Calculate the gain (number of newly covered subsets) for each block.
        Gain[b] = sum(1 for s in block_to_subsets[b] if cover_count[s] == 0)
        """
        gains = np.zeros(num_blocks, dtype=np.int32)
        if not np.any(uncovered_mask):
            return gains
            
        # Iterate over uncovered subsets
        # This is O(num_uncovered * avg_blocks_per_subset)
        uncovered_indices = np.where(uncovered_mask)[0]
        for s_idx in uncovered_indices:
            blocks_covering = subset_to_blocks_arr[s_idx]
            gains[blocks_covering] += 1
            
        return gains

    def run_greedy(cover_count, random_seed=None, noise_scale=0.0, temperature=0.0):
        """
        Greedy repair.
        cover_count: current coverage counts (numpy array).
        Returns list of new block indices added.
        """
        rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng()
        
        uncovered_mask = (cover_count == 0).astype(bool)
        if not np.any(uncovered_mask):
            return []
            
        gains = get_gains(cover_count, uncovered_mask)
        selected_blocks = []
        uncovered_count = int(np.sum(uncovered_mask))
        
        while uncovered_count > 0:
            if noise_scale > 0 or temperature > 0:
                if temperature > 0:
                    # Soft selection: probability proportional to exp(gain / temperature)
                    # But we need to handle 0 gains carefully. 
                    # If all gains are 0, we break.
                    if np.max(gains) == 0:
                        break
                    # Shift gains to be positive for softmax
                    shifted = gains.astype(np.float64)
                    # Normalize by max to avoid overflow
                    max_gain = np.max(shifted)
                    if max_gain > 0:
                        shifted -= max_gain
                    probs = np.exp(shifted / temperature)
                    probs /= np.sum(probs)
                    best_idx = np.random.choice(num_blocks, p=probs)
                else:
                    noise = rng.uniform(0, noise_scale, size=num_blocks)
                    scores = gains + noise
                    best_idx = np.argmax(scores)
            else:
                best_idx = np.argmax(gains)
            
            if gains[best_idx] == 0:
                # No block can cover any new subset? This shouldn't happen if uncovered_count > 0
                # unless the structure is impossible, but for covering designs it's always possible.
                # If it happens, break to avoid infinite loop.
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            # Find which of these subsets are currently uncovered
            newly_covered = b_subs[uncovered_mask[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            # Update cover_count and uncovered_mask
            cover_count[newly_covered] += 1
            uncovered_mask[newly_covered] = False
            
            # Update gains for all blocks that cover these newly covered subsets
            # A block's gain decreases by 1 for each newly covered subset it covers
            for s in newly_covered:
                blocks_covering_s = subset_to_blocks_arr[s]
                gains[blocks_covering_s] -= 1
                
            uncovered_count -= len(newly_covered)
            
        return selected_blocks

    def minimize(blocks_indices, random_seed=None):
        """
        Remove redundant blocks.
        Returns a new list of block indices.
        """
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[block_to_subsets_arr[b_idx]] += 1
            
        changed = True
        while changed:
            changed = False
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                b_subs = block_to_subsets_arr[b_idx]
                # If any subset in this block is covered ONLY by this block, it's not redundant
                if not np.any(cover_count[b_subs] == 1):
                    redundant_blocks.append(i)
                    
            if redundant_blocks:
                # Remove a random redundant block to break symmetry
                remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
                b_idx_to_remove = current_blocks[remove_pos]
                
                cover_count[block_to_subsets_arr[b_idx_to_remove]] -= 1
                current_blocks.pop(remove_pos)
                changed = True
                
        return current_blocks

    def perturb_and_repair(best_blocks, num_remove, random_seed):
        """
        Remove num_remove random blocks, repair, minimize.
        """
        if len(best_blocks) <= num_remove:
            return None
            
        rng = random.Random(random_seed)
        
        # Select blocks to remove
        remove_indices = rng.sample(range(len(best_blocks)), num_remove)
        blocks_to_remove_set = set(best_blocks[i] for i in remove_indices)
        
        # Create new list without removed blocks
        remaining_blocks = [b for b in best_blocks if b not in blocks_to_remove_set]
        
        # Calculate cover count for remaining
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in remaining_blocks:
            cover_count[block_to_subsets_arr[b_idx]] += 1
            
        # Repair
        repair_res = run_greedy(cover_count, random_seed=random_seed+1, noise_scale=0.0)
        
        # Combine
        new_blocks = remaining_blocks + repair_res
        
        # Minimize
        min_res = minimize(new_blocks, random_seed=random_seed+2)
        
        return min_res

    def swap_improve(blocks_indices, random_seed):
        """
        Try to replace one block with another to reduce total size or maintain size with better structure.
        Here we try to find a block B in the set and a block C not in the set such that
        replacing B with C keeps the cover valid and potentially allows further reduction.
        Actually, simple swap doesn't reduce size. It's for diversification.
        For minimization, we rely on minimize().
        """
        pass

    best_blocks = None
    best_len = float('inf')
    
    trial = 0
    
    # Phase 1: Initial Greedy Searches
    # We want to find a good starting point quickly.
    while time.time() < deadline - 40.0:  # Leave 40s for local search
        # Run greedy with some noise to find diverse starting points
        # Low noise for first few trials to get a solid baseline
        if trial < 3:
            noise = 0.0
            temp = 0.0
        else:
            noise = 0.0
            temp = 0.5 + (trial % 5) * 0.5  # Use temperature-based selection for diversity
            
        cover_count_zero = np.zeros(num_subsets, dtype=np.int32)
        greedy_res = run_greedy(cover_count_zero, random_seed=trial, noise_scale=noise, temperature=temp)
        
        min_res = minimize(greedy_res, random_seed=trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        trial += 1
        
        # Early stopping if we find a very good solution?
        # The lower bound is 264. If we get close, we should spend more time polishing.
        # But we don't know the exact best.
        
    if best_blocks is None:
        # Fallback
        cover_count_zero = np.zeros(num_subsets, dtype=np.int32)
        best_blocks = minimize(run_greedy(cover_count_zero, random_seed=0, noise_scale=0.0), random_seed=0)
        best_len = len(best_blocks)

    # Phase 2: Local Search (Perturbation)
    # Focus on small perturbations to improve the current best.
    no_improve_count = 0
    while time.time() < deadline:
        # Try different perturbation sizes
        # Smaller perturbations are more likely to succeed but explore less
        # Weight towards small perturbations
        r = random.random()
        if r < 0.5:
            num_remove = 1
        elif r < 0.8:
            num_remove = 2
        elif r < 0.95:
            num_remove = 3
        else:
            num_remove = 5
            
        new_candidate = perturb_and_repair(best_blocks, num_remove, trial + 10000)
        
        if new_candidate is not None:
            if len(new_candidate) < best_len:
                best_len = len(new_candidate)
                best_blocks = new_candidate
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        trial += 1

    # Final Polish: Try to remove more blocks with a more aggressive minimize?
    # The current minimize is already quite thorough (repeated passes).
    # One more pass on the very best solution.
    if best_blocks is not None:
        final_polish = minimize(best_blocks, random_seed=99999)
        if len(final_polish) < best_len:
            best_blocks = final_polish
            best_len = len(final_polish)

    return [list(all_blocks[i]) for i in best_blocks]
