import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s (or 150s) kill

    # Precompute all t-subsets and map to indices for fast lookup
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubsets)}
    n_tsubs = len(all_tsubsets)
    
    # Precompute for each t-subset, the list of all k-subsets that contain it
    # For each t-subset, we need to choose (k-t) additional points from (v-t) remaining
    # We'll store these as a list of tuples (the k-subsets)
    
    # Actually, for efficiency in the greedy step, it's better to precompute
    # for each k-subset, the list of t-subsets it covers.
    # Then for the greedy step, given an anchor t-subset, we need all k-subsets containing it.
    
    # Let's precompute: for each t-subset, the list of k-subsets containing it.
    # This might be memory heavy: C(24,3) = 2024 t-subsets, each contained in C(21,4) = 5985 k-subsets.
    # Total entries: 2024 * 5985 ~ 12 million. Each entry is a tuple of 7 ints.
    # This might be too much memory or time to precompute.
    
    # Alternative: precompute all k-subsets and for each, the set of t-subsets it covers.
    # C(24,7) = 346104 k-subsets. Each covers C(7,3) = 35 t-subsets.
    # Total t-subset references: 346104 * 35 ~ 12 million.
    # We can store this as a list of arrays of t-subset indices.
    
    all_ksubsets = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubsets)
    
    # For each k-subset, compute the indices of the t-subsets it covers
    ksub_tsub_indices = []
    for ksub in all_ksubsets:
        tsubs_in_ksub = list(itertools.combinations(ksub, t))
        indices = [tsub_to_idx[tsub] for tsub in tsubs_in_ksub]
        ksub_tsub_indices.append(indices)
    
    # For the greedy step, given an anchor t-subset, we need all k-subsets containing it.
    # We can precompute a mapping from t-subset index to list of k-subset indices.
    tsub_to_ksubs = defaultdict(list)
    for ksub_idx, tsub_indices in enumerate(ksub_tsub_indices):
        for tsub_idx in tsub_indices:
            tsub_to_ksubs[tsub_idx].append(ksub_idx)
    
    best_blocks = None
    best_len = float('inf')
    
    def _greedy_cover_fast(v, k, t, all_tsubsets, tsub_to_idx, deadline, seed_val):
        rng = random.Random(seed_val)
        
        uncovered_indices = set(range(len(all_tsubsets)))
        blocks = []
        
        while uncovered_indices:
            if time.time() > deadline:
                return None, False
            
            # Pick a random uncovered t-subset to anchor our search
            anchor_idx = rng.choice(list(uncovered_indices))
            
            # Get all k-subsets containing this anchor t-subset
            candidate_ksub_indices = tsub_to_ksubs[anchor_idx]
            
            if not candidate_ksub_indices:
                return None, False
            
            best_gain = -1
            best_ksub_idx = None
            best_tiebreakers = []
            
            # Iterate over candidate k-subsets and compute gain
            for ksub_idx in candidate_ksub_indices:
                # Calculate gain: how many uncovered t-subsets does this k-subset cover?
                gain = 0
                for tsub_idx in ksub_tsub_indices[ksub_idx]:
                    if tsub_idx in uncovered_indices:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_ksub_idx = ksub_idx
                    best_tiebreakers = [ksub_idx]
                elif gain == best_gain and gain > 0:
                    best_tiebreakers.append(ksub_idx)
            
            if best_ksub_idx is None or best_gain == 0:
                return None, False
                
            # If there are tiebreakers, pick one randomly
            chosen_ksub_idx = rng.choice(best_tiebreakers)
            chosen_block = list(all_ksubsets[chosen_ksub_idx])
            
            blocks.append(chosen_block)
            
            # Update uncovered set
            for tsub_idx in ksub_tsub_indices[chosen_ksub_idx]:
                uncovered_indices.discard(tsub_idx)
        
        return blocks, True

    def _remove_redundant(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline):
        """Remove blocks that don't contribute uniquely to the cover."""
        if not blocks:
            return blocks
        
        # Compute cover count for each t-subset
        cover_count = np.zeros(len(all_tsubsets), dtype=np.int32)
        block_tsub_indices = []
        for block in blocks:
            tsubs = list(itertools.combinations(block, t))
            indices = [tsub_to_idx[tsub] for tsub in tsubs]
            block_tsub_indices.append(indices)
            for idx in indices:
                cover_count[idx] += 1
        
        improved = list(blocks)
        improved_tsub_indices = list(block_tsub_indices)
        cover_count_local = cover_count.copy()
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(improved) - 1, -1, -1):
                # Check if this block is redundant (all its t-subsets are covered by others)
                redundant = True
                for tsub_idx in improved_tsub_indices[i]:
                    if cover_count_local[tsub_idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    for tsub_idx in improved_tsub_indices[i]:
                        cover_count_local[tsub_idx] -= 1
                    improved.pop(i)
                    improved_tsub_indices.pop(i)
                    changed = True
        
        return improved

    def _local_improve(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline):
        if not blocks:
            return blocks
        
        # Remove redundant blocks
        improved = _remove_redundant(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline)
        
        return improved

    # Main loop
    trial = 0
    while time.time() < deadline:
        trial += 1
        
        # Use different seeds for diversity
        seed = trial
        
        blocks, valid = _greedy_cover_fast(v, k, t, all_tsubsets, tsub_to_idx, deadline, seed)
        
        if valid:
            # Local improvement
            improved_blocks = _local_improve(blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline)
            
            if len(improved_blocks) < best_len:
                best_len = len(improved_blocks)
                best_blocks = improved_blocks
                
            # Early stop if we hit the known best (76)
            if best_len <= 76:
                break

    if best_blocks is None:
        # Fallback
        blocks, valid = _greedy_cover_fast(v, k, t, all_tsubsets, tsub_to_idx, deadline, 0)
        if valid:
            best_blocks = blocks
            best_blocks = _local_improve(best_blocks, v, k, t, all_tsubsets, tsub_to_idx, deadline)
        else:
            best_blocks = []
            
    return best_blocks
