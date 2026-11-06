"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def construct():
    """
    Construct a binary error-correcting code of length 18 with minimum distance 6.
    Uses a greedy sieve approach seeded with a large initial code, then attempts
    to improve via local search (swapping codewords for better candidates).
    """
    N = 18
    MIN_DIST = 6
    DEADLINE = 85.0  # Safety margin before 90s kill
    start_time = time.time()
    
    size = 1 << N
    
    # Precompute all masks of weight < MIN_DIST (0 to 5)
    # There are sum_{i=0}^{5} C(18, i) = 1 + 18 + 153 + 816 + 3060 + 8568 = 12616
    masks_list = [i for i in range(size) if i.bit_count() < MIN_DIST]
    masks = np.array(masks_list, dtype=np.uint32)
    
    def mark_forbidden(c, forbidden):
        """Mark all words within Hamming distance < MIN_DIST from c as forbidden."""
        neighbors = (c ^ masks) % size
        forbidden[neighbors] = True
    
    def unmark_forbidden(c, forbidden):
        """Unmark all words within Hamming distance < MIN_DIST from c.
        Only safe if no other codeword has this word in its ball."""
        neighbors = (c ^ masks) % size
        forbidden[neighbors] = False
    
    def get_forbidden_set(c):
        """Get the set of forbidden neighbors for word c."""
        return set((c ^ m) % size for m in masks_list)
    
    # Phase 1: Greedy construction with multiple orderings
    # We'll try several different orderings and keep the best result
    
    best_code = []
    best_size = 0
    
    # Ordering 1: Natural order (0, 1, 2, ...)
    def greedy_natural():
        forbidden = np.zeros(size, dtype=bool)
        code = [0]
        mark_forbidden(0, forbidden)
        
        for c in range(1, size):
            if time.time() - start_time > DEADLINE - 10:
                break
            if not forbidden[c]:
                code.append(c)
                mark_forbidden(c, forbidden)
        
        return code
    
    # Ordering 2: Weight-sorted (by Hamming weight, then by value)
    def greedy_weight_sorted():
        forbidden = np.zeros(size, dtype=bool)
        code = [0]
        mark_forbidden(0, forbidden)
        
        # Sort candidates by weight, then by value
        candidates = list(range(1, size))
        candidates.sort(key=lambda x: (x.bit_count(), x))
        
        for c in candidates:
            if time.time() - start_time > DEADLINE - 10:
                break
            if not forbidden[c]:
                code.append(c)
                mark_forbidden(c, forbidden)
        
        return code
    
    # Ordering 3: Random
    def greedy_random(seed):
        rng = np.random.default_rng(seed)
        forbidden = np.zeros(size, dtype=bool)
        code = [0]
        mark_forbidden(0, forbidden)
        
        candidates = np.arange(1, size, dtype=np.int64)
        rng.shuffle(candidates)
        
        for c in candidates:
            if time.time() - start_time > DEADLINE - 10:
                break
            c = int(c)
            if not forbidden[c]:
                code.append(c)
                mark_forbidden(c, forbidden)
        
        return code
    
    # Try natural order first
    code = greedy_natural()
    best_code = code
    best_size = len(code)
    
    # Try weight-sorted if time permits
    if time.time() - start_time < DEADLINE - 15:
        code = greedy_weight_sorted()
        if len(code) > best_size:
            best_code = code
            best_size = len(code)
    
    # Try a few random orderings if time permits
    if time.time() - start_time < DEADLINE - 20:
        for seed in [42, 123, 456]:
            if time.time() - start_time > DEADLINE - 15:
                break
            code = greedy_random(seed)
            if len(code) > best_size:
                best_code = code
                best_size = len(code)
    
    # Phase 2: Local search - try to improve by swapping codewords
    # For each codeword in the code, try replacing it with a candidate that
    # is currently forbidden but would allow adding more words overall.
    
    if time.time() - start_time < DEADLINE - 5 and len(best_code) > 0:
        code_set = set(best_code)
        forbidden = np.zeros(size, dtype=bool)
        for c in best_code:
            mark_forbidden(c, forbidden)
        
        # Try local search: for each codeword, try removing it and adding
        # a different candidate in its place
        improved = True
        max_iterations = 50  # Limit local search iterations
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE - 5:
                break
            if not improved:
                break
            
            improved = False
            # Try swapping each codeword with a random candidate
            rng = np.random.default_rng(iteration + 1000)
            
            for idx in range(len(best_code)):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                old_cw = best_code[idx]
                
                # Try a few random candidates to replace old_cw
                for attempt in range(3):
                    if time.time() - start_time > DEADLINE - 5:
                        break
                    
                    new_cw = int(rng.integers(0, size))
                    if new_cw == old_cw or new_cw in code_set:
                        continue
                    
                    # Check if new_cw is compatible with all other codewords
                    compatible = True
                    for other_idx in range(len(best_code)):
                        if other_idx == idx:
                            continue
                        other_cw = best_code[other_idx]
                        d = (new_cw ^ other_cw).bit_count()
                        if d < MIN_DIST:
                            compatible = False
                            break
                    
                    if not compatible:
                        continue
                    
                    # Swap: remove old_cw, add new_cw
                    # This is a neutral swap (same size), so we only do it
                    # if it helps in subsequent greedy steps
                    # For now, just accept swaps that don't hurt
                    # Actually, let's do a more sophisticated approach:
                    # Try to add a new word after removing an old one
                    
                    # Remove old_cw
                    best_code[idx] = new_cw
                    code_set.remove(old_cw)
                    code_set.add(new_cw)
                    improved = True
                    break
    
    # Phase 3: Final greedy pass to add any remaining compatible words
    if time.time() - start_time < DEADLINE - 2:
        forbidden = np.zeros(size, dtype=bool)
        for c in best_code:
            mark_forbidden(c, forbidden)
        
        # Try to add more words
        added = True
        while added and time.time() - start_time < DEADLINE - 2:
            added = False
            for c in range(size):
                if not forbidden[c]:
                    best_code.append(c)
                    mark_forbidden(c, forbidden)
                    added = True
                    break
    
    return best_code

# EVOLVE-BLOCK-END
