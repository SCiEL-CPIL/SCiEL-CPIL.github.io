"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _precompute_flip_masks(n, max_dist):
    """Precompute masks for flipping exactly d bits, for d in 0..max_dist."""
    masks = [[] for _ in range(max_dist + 1)]
    masks[0].append(0)
    for d in range(1, max_dist + 1):
        for m in masks[d - 1]:
            for i in range(n):
                if not (m & (1 << i)):
                    masks[d].append(m | (1 << i))
    return masks

def construct():
    """Construct a large binary code of length 17 with minimum distance 6."""
    deadline = time.time() + 55.0
    N = 17
    MIN_DIST = 6
    MAX_DIST_FORBID = MIN_DIST - 1  # 5
    
    # Precompute flip masks
    flip_masks = _precompute_flip_masks(N, MAX_DIST_FORBID)
    
    # Total number of possible words
    total_words = 1 << N
    
    best_code = []
    rng = random.Random(42)
    
    # Run multiple restarts with different random orderings
    while time.time() < deadline:
        # Create a random permutation of all words
        # This is O(2^17) which is fast
        candidates = list(range(total_words))
        rng.shuffle(candidates)
        
        forbidden = np.zeros(total_words, dtype=bool)
        code = []
        
        for w in candidates:
            if not forbidden[w]:
                code.append(w)
                forbidden[w] = True
                # Mark all neighbors within distance < MIN_DIST as forbidden
                for d in range(1, MAX_DIST_FORBID + 1):
                    for m in flip_masks[d]:
                        nb = w ^ m
                        forbidden[nb] = True
                        
        if len(code) > len(best_code):
            best_code = code
            
        # If we found a very large code, we might stop early?
        # Theoretical max is 340. If we get > 300, it's good.
        # But let's keep searching for a bit longer to see if we can do better.
        # Or just keep going until deadline.
        
    return best_code
# EVOLVE-BLOCK-END
