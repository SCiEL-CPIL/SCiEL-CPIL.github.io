"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def _precompute_covering_data(v, k, t):
    """
    Precompute data structures for fast greedy covering.
    Returns (all_t_subsets, t_subset_to_blocks_potential)
    """
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map from t-subset to list of blocks (k-subsets) that contain it
    # We'll build this lazily or precompute if feasible
    # C(19,4) = 3876, C(19,7) = 50388
    # For each t-subset, how many k-subsets contain it? C(v-t, k-t) = C(15,3) = 455
    # Total entries: 3876 * 455 = 1,763,580. This is manageable.
    
    # Actually, for greedy, we need to know for a partial block, which uncovered t-subsets it covers.
    # Better approach: maintain uncovered set, and for each candidate point, compute gain.
    
    # Precompute: for each t-subset, store it as a frozenset for fast lookup
    t_subsets_set = set(all_t_subsets)
    
    # Precompute for each (t-1)-subset, the list of t-subsets extending it
    # This helps when adding a point to a partial block
    # For a partial block B of size s < k, adding point x covers t-subsets 
    # that are subsets of B ∪ {x} and not subsets of B.
    # A t-subset S is covered by B ∪ {x} iff S ⊆ B ∪ {x}.
    # S is already covered by B iff S ⊆ B.
    # So new t-subsets covered are those S with S ⊆ B ∪ {x} and S ⊄ B.
    # This means x ∈ S and S \ {x} ⊆ B, and |S| = t.
    # So S = (S \ {x}) ∪ {x} where S \ {x} is a (t-1)-subset of B.
    
    # Precompute: for each (t-1)-subset, list of points x such that (t-1-subset ∪ {x}) is a valid t-subset
    # i.e., for each (t-1)-tuple, the set of x not in it such that the t-set exists (always true for x in 0..v-1)
    
    # Actually simpler: just iterate over all t-subsets of the current block ∪ {x} that include x.
    # For a block B of size s, adding x, the new t-subsets are combinations of (t-1) elements from B plus x.
    # Number: C(s, t-1). For s up to k-1=6, t=4, C(6,3)=20. Very fast.
    
    return all_t_subsets, t_subsets_set


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second deadline
    
    # Precompute all t-subsets
    all_t_subsets = set(itertools.combinations(range(v), t))
    
    # Greedy covering with precomputed gain
    uncovered = set(all_t_subsets)
    blocks = []
    
    while uncovered:
        if time.time() > deadline - 5.0:
            # Safety margin: return what we have (may be incomplete, but let's try to finish current block)
            pass
        
        # Pick the first uncovered t-subset to start a new block
        # Actually, for better results, we could pick a t-subset that is "hardest" to cover
        # but for simplicity, start with min
        seed_subset = min(uncovered)
        block = list(seed_subset)
        
        while len(block) < k:
            if time.time() > deadline - 1.0:
                break
            
            # Find point x not in block that maximizes number of uncovered t-subsets covered
            best_x = None
            best_gain = -1
            
            # Precompute (t-1)-subsets of current block for gain calculation
            block_set = set(block)
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Count uncovered t-subsets that include x and (t-1) elements from block
                gain = 0
                for rest in t_minus_1_subsets:
                    t_sub = tuple(sorted(rest + (x,)))
                    if t_sub in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is not None and best_gain > 0:
                block.append(best_x)
                block_set.add(best_x)
            elif best_x is not None:
                # No gain, just add arbitrary point to fill block
                block.append(best_x)
                block_set.add(best_x)
            else:
                break
        
        block.sort()
        blocks.append(block)
        
        # Remove all t-subsets covered by this block
        for t_sub in itertools.combinations(block, t):
            uncovered.discard(t_sub)
    
    return blocks


# EVOLVE-BLOCK-END
