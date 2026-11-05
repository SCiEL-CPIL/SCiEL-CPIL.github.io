import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {trip: i for i, trip in enumerate(triples)}
    n_triples = len(triples)
    
    blocks = list(itertools.combinations(points, k))
    n_blocks = len(blocks)
    
    # Number of 64-bit words needed
    n_words = (n_triples + 63) // 64
    
    # Precompute block masks as numpy array (n_blocks, n_words)
    # Initialize to 0
    block_masks = np.zeros((n_blocks, n_words), dtype=np.uint64)
    
    for i, b in enumerate(blocks):
        for trip in itertools.combinations(b, t):
            idx = triple_to_idx[trip]
            word_idx = idx // 64
            bit_idx = idx % 64
            block_masks[i, word_idx] |= (1 << bit_idx)
            
    # Initialize uncovered mask
    # All 1s for valid bits
    uncovered = np.ones(n_words, dtype=np.uint64)
    # Clear invalid bits in the last word
    if n_triples % 64 != 0:
        valid_bits = n_triples % 64
        uncovered[-1] = (1 << valid_bits) - 1
        
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    max_time = 100 # Leave buffer
    
    # Run multiple greedy instances
    for run in range(50):
        if time.time() - start_time > max_time:
            break
            
        # Copy uncovered mask
        curr_uncovered = uncovered.copy()
        selected = []
        
        while True:
            # Check if uncovered is zero
            # np.any(curr_uncovered != 0)
            if not np.any(curr_uncovered):
                break
                
            # Calculate gains
            # Vectorized AND
            inters = curr_uncovered[None, :] & block_masks
            # Count bits
            # np.bitwise_count is available in numpy >= 1.17? 
            # Actually, np.bitwise_count was added in NumPy 1.17? 
            # Wait, np.bitwise_count is not a standard function in older versions.
            # Use np.unpackbits? No, that's for 0/1 arrays.
            # Let's use a lookup table or just sum of bits?
            # np.bit_count is not in numpy.
            # We can use:
            # gains = np.sum(np.bitwise_count(inters), axis=1) -> Error: no bitwise_count
            # 
            # Alternative: Convert to list of ints and use int.bit_count?
            # inters is (346k, 32). 
            # Flattening and using list comprehension might be slow.
            #
            # Let's check if we can use a trick.
            # For uint64, we can use:
            # x = inters
            # x = x - ((x >> 1) & 0x5555555555555555)
            # ... popcount algorithm
            # This is complex to vectorize correctly without errors.
            
            # Simpler: Use Python's int.bit_count on the flattened array?
            # No, let's stick to a simpler heuristic if numpy popcount is not easy.
            
            # Actually, we can just compute the gain by summing the number of set bits.
            # We can precompute a popcount table for 8-bit values and use it?
            # Or just use the fact that we are in Python.
            
            # Let's try a different vectorized approach:
            # We only need the ARGMAX.
            # We can compute `inters` and then `gains`.
            # If we can't vectorize popcount easily, we can fall back to:
            # gains = np.array([row.bit_count() for row in inters]) -> inters rows are arrays, not ints.
            
            # Let's use a helper function or just accept the overhead of converting to Python ints if needed.
            # But 346k conversions per step * 76 steps * 50 runs = 1.3B conversions. Too slow.
            
            # Let's use the bitwise popcount algorithm for uint64 in numpy.
            # Reference: https://stackoverflow.com/questions/11351076/
            
            def popcount(x):
                # x is uint64 array
                x = x - ((x >> 1) & 0x5555555555555555)
                x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
                x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
                x = x + (x >> 8)
                x = x + (x >> 16)
                x = x + (x >> 32)
                return x & 0x7F # Max 64 bits set, so 7 bits enough
            
            # Apply popcount to inters (346k, 32)
            # This will do the operations element-wise.
            gains = popcount(inters).sum(axis=1)
            
            max_gain = np.max(gains)
            
            if max_gain == 0:
                break # Should not happen if uncovered is not empty
            
            candidates = np.where(gains == max_gain)[0]
            chosen = random.choice(candidates)
            
            selected.append(chosen)
            curr_uncovered &= ~block_masks[chosen]
            
        # Local Search: Redundancy Removal
        # Use simple Python logic for the small number of selected blocks
        if selected:
            # Calculate coverage for each triple
            # We can use the masks
            # total_coverage_mask_counts?
            # Let's just count how many times each triple is covered.
            # n_triples is 2024.
            # selected has ~76 blocks.
            # coverage_counts = np.zeros(n_triples, dtype=np.int32)
            # for b_idx in selected:
            #     mask = block_masks[b_idx]
            #     # Iterate words and bits?
            #     # Or just use the precomputed list of triples for this block?
            #     # We didn't store the list of triples per block, only masks.
            #     # We can extract triples from mask.
            
            # Faster: Precompute `block_triples` list of lists?
            # We did that in the seed. Let's do it here too for local search.
            
            # Actually, we can just check redundancy using the masks.
            # A block is redundant if all its triples are covered by other blocks.
            # i.e., for each triple in block i, coverage_count > 1.
            
            # Let's compute coverage_counts array.
            coverage_counts = np.zeros(n_triples, dtype=np.int32)
            
            for b_idx in selected:
                mask = block_masks[b_idx]
                for w in range(n_words):
                    m = mask[w]
                    while m:
                        lsb = m & -m
                        bit_idx = lsb.bit_length() - 1
                        triple_idx = w * 64 + bit_idx
                        if triple_idx < n_triples:
                            coverage_counts[triple_idx] += 1
                        m ^= lsb
            
            # Remove redundant
            changed = True
            while changed:
                changed = False
                for i in range(len(selected)):
                    b_idx = selected[i]
                    mask = block_masks[b_idx]
                    redundant = True
                    for w in range(n_words):
                        m = mask[w]
                        while m:
                            lsb = m & -m
                            bit_idx = lsb.bit_length() - 1
                            triple_idx = w * 64 + bit_idx
                            if triple_idx < n_triples:
                                if coverage_counts[triple_idx] == 1:
                                    redundant = False
                                    break
                            m ^= lsb
                        if not redundant:
                            break
                    
                    if redundant:
                        # Remove
                        mask = block_masks[b_idx]
                        for w in range(n_words):
                            m = mask[w]
                            while m:
                                lsb = m & -m
                                bit_idx = lsb.bit_length() - 1
                                triple_idx = w * 64 + bit_idx
                                if triple_idx < n_triples:
                                    coverage_counts[triple_idx] -= 1
                                m ^= lsb
                        selected.pop(i)
                        changed = True
                        break
            
            if len(selected) < best_count:
                best_count = len(selected)
                best_blocks = [blocks[i] for i in selected]
                
    if best_blocks is None:
        # Fallback
        return []
        
    return best_blocks
