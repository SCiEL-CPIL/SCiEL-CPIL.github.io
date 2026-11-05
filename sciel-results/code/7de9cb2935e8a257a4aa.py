import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit (evaluator kills at 150s)
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 2. Efficient Representation using Bitsets
    # Instead of a dense (num_blocks, num_triples) matrix which is ~700MB,
    # we represent each block's covered triples as a list of uint64 bitmasks.
    # num_triples = C(24,3) = 2024.
    # 2024 bits / 64 bits per word = 32 words.
    # So each block is represented by an array of 32 uint64s.
    # Total memory: 346,104 blocks * 32 words * 8 bytes/word ≈ 88 MB.
    # This is much smaller and allows faster bitwise operations for "AND" with uncovered mask.
    
    words_per_block = (num_triples + 63) // 64
    # block_bitsets[i] is a numpy array of shape (words_per_block,) dtype uint64
    block_bitsets = np.zeros((num_blocks, words_per_block), dtype=np.uint64)
    
    # Also store the list of triple indices for each block for fast pruning checks
    block_triples = []
    
    for i, b in enumerate(all_blocks):
        tris = [triple_to_idx[tri] for tri in itertools.combinations(b, t)]
        block_triples.append(tris)
        
        # Set bits in the bitset
        for tri_idx in tris:
            word_idx = tri_idx // 64
            bit_idx = tri_idx % 64
            block_bitsets[i, word_idx] |= (1 << bit_idx)
            
    # 3. Multi-Start Randomized Greedy
    # The deterministic greedy often gets stuck in a local optimum where the "best" next block
    # prevents a better global configuration. Randomized greedy (simulated annealing style or
    # simply random sampling from top-K) allows escaping these traps.
    
    best_blocks = None
    best_count = float('inf')
    
    # We will run multiple iterations.
    # Each iteration:
    # 1. Randomized Greedy Construction
    # 2. Redundancy Pruning
    
    # To fit in time, we estimate how many iterations we can run.
    # Greedy step: ~80 blocks * (346k blocks * 32 words ops) -> 80 * 11M ops ~ 900M ops.
    # This might be slow in pure Python loops over blocks, but we can vectorize the gain calculation?
    # No, vectorizing gain calculation requires summing bits.
    # Alternative: Keep the dense matrix for gain calculation if memory allows, 
    # but use bitsets for pruning?
    # Actually, the previous program used a dense matrix and scored 1.09.
    # The bottleneck is the gain calculation: `inc_matrix @ uncovered`.
    # 346k x 2024 matrix multiplication.
    
    # Let's try a hybrid:
    # Use the dense matrix for the greedy step (fast gain via dot product).
    # Use bitsets or lists for pruning (fast membership checks).
    # But the memory for dense matrix is the risk.
    # Let's try to build the dense matrix again but ensure we don't blow up.
    # If MemoryError, fallback to a slower but memory-safe method.
    
    try:
        # Dense incidence matrix for fast gain calculation
        # (num_blocks, num_triples) uint8
        inc_matrix = np.zeros((num_blocks, num_triples), dtype=np.uint8)
        for i in range(num_blocks):
            inc_matrix[i, block_triples[i]] = 1
            
    except MemoryError:
        # Fallback: We cannot use dense matrix. 
        # We will use a slower greedy that calculates gain on the fly or uses a sparse structure.
        # For now, let's assume memory is available (88MB for bitsets, 700MB for dense).
        # 700MB is often okay in modern eval environments, but risky.
        # If we fail, we return a simple random valid covering if possible, or best effort.
        inc_matrix = None
        # If inc_matrix is None, we need a different greedy.
        # Let's implement a "Sparse Greedy" using the block_triples lists.
        # Gain = sum(uncovered[j] for j in block_triples[i])
        # This is O(C(k,t)) per block. 35 * 346k = 12M ops per step.
        # 80 steps * 12M = 960M ops. Might take 10-20s per run.
        # We can only afford ~5-10 runs.
        
    best_result = None
    
    # Number of restarts depends on time.
    # Let's try 10 restarts.
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        uncovered = np.ones(num_triples, dtype=np.uint8)
        selected_blocks = []
        
        # Randomized Greedy
        # We pick a block with probability proportional to its gain, 
        # but with a "temperature" that allows picking sub-optimal blocks.
        # Or simpler: Pick randomly from the top 10% of blocks by gain?
        # Or: Gain + Noise?
        
        # Let's use: Score = Gain + random.uniform(0, max_gain * 0.2)
        # This biases towards high gain but allows exploration.
        
        while np.any(uncovered):
            if time.time() > deadline - 15:
                break
                
            if inc_matrix is not None:
                # Vectorized gain calculation
                gains = inc_matrix @ uncovered
                gains = gains.astype(np.float64)
                
                # Add noise
                # To avoid picking blocks with 0 gain, we only consider positive gains?
                # If all gains are 0, we are stuck (should not happen).
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                    
                # Add noise to escape local optima
                # Noise scale: 10-20% of max gain
                noise = np.random.uniform(0, max_gain * 0.2, size=num_blocks)
                scores = gains + noise
                
                # Mask out blocks that are already selected?
                # In set cover, we don't usually re-select, but since gains become 0, 
                # they won't be picked unless noise is huge and other gains are 0.
                # To be safe, we can set scores of selected blocks to -inf.
                for idx in selected_blocks:
                    scores[idx] = -1.0
                    
                best_idx = np.argmax(scores)
                best_gain = gains[best_idx]
                
                if best_gain == 0:
                    break
                    
                selected_blocks.append(best_idx)
                
                # Update uncovered
                mask = inc_matrix[best_idx] == 1
                uncovered[mask] = 0
                
            else:
                # Sparse fallback (slower)
                best_score = -1
                best_idx = -1
                best_gain = 0
                
                # Calculate gains for all blocks
                # This is the bottleneck.
                # Optimization: Only check blocks that cover at least one uncovered triple?
                # Hard to do efficiently without index.
                # Let's just iterate.
                for i, b_tris in enumerate(block_triples):
                    if i in selected_blocks:
                        continue
                    gain = 0
                    for tri_idx in b_tris:
                        if uncovered[tri_idx]:
                            gain += 1
                            
                    if gain == 0:
                        continue
                    
                    # Noise
                    # We don't have max_gain easily here without a full pass.
                    # Let's just use random selection among top 5?
                    # Or simple greedy with random tie-breaking?
                    # For simplicity in fallback, let's do pure greedy with random tie-break.
                    # This is less effective but works.
                    score = gain
                    # Add small random noise to break ties
                    score += random.random() * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_idx = i
                        best_gain = gain
                        
                if best_idx == -1:
                    break
                    
                selected_blocks.append(best_idx)
                
                for tri_idx in block_triples[best_idx]:
                    uncovered[tri_idx] = 0
                    
        # Redundancy Pruning
        if selected_blocks:
            # Rebuild counts
            counts = np.zeros(num_triples, dtype=np.int32)
            for idx in selected_blocks:
                if inc_matrix is not None:
                    counts[inc_matrix[idx]] += 1
                else:
                    for tri_idx in block_triples[idx]:
                        counts[tri_idx] += 1
                        
            new_selected = []
            # Iterate multiple times to remove cascading redundancies
            # But limit to 2 passes for time
            for _ in range(2):
                if time.time() > deadline - 10:
                    break
                next_selected = []
                for idx in selected_blocks:
                    if inc_matrix is not None:
                        b_tris_idx = np.where(inc_matrix[idx] == 1)[0]
                    else:
                        b_tris_idx = block_triples[idx]
                        
                    # Check if all triples in this block are covered by >1 blocks
                    # i.e., counts[tri] > 1 for all tri in b_tris
                    if np.all(counts[b_tris_idx] > 1):
                        # Remove
                        counts[b_tris_idx] -= 1
                    else:
                        next_selected.append(idx)
                        
                if len(next_selected) == len(selected_blocks):
                    break
                selected_blocks = next_selected
                # Recalculate counts for next pass? 
                # No, we updated counts in place. But if we removed a block, 
                # counts for its triples decreased. 
                # However, the condition `counts > 1` was checked against the 
                # counts *including* the current block. 
                # If we remove it, we decrement. 
                # This logic is sound for a single pass.
                # For the second pass, we should recalculate counts from scratch 
                # or be very careful. 
                # Let's recalculate counts from scratch for the second pass to be safe.
                if len(selected_blocks) < len(new_selected):
                    new_selected = selected_blocks
                    # Recalculate counts for the next iteration of the while loop?
                    # The loop above only runs 2 times.
                    # Let's just do one pass of pruning to save time, 
                    # or recalculate counts properly.
                    
            # Let's do a proper multi-pass pruning with count recalculation
            # It's safer.
            # Pass 1
            counts = np.zeros(num_triples, dtype=np.int32)
            for idx in selected_blocks:
                if inc_matrix is not None:
                    counts[inc_matrix[idx]] += 1
                else:
                    for tri_idx in block_triples[idx]:
                        counts[tri_idx] += 1
                        
            new_selected = []
            for idx in selected_blocks:
                if inc_matrix is not None:
                    b_tris_idx = np.where(inc_matrix[idx] == 1)[0]
                else:
                    b_tris_idx = block_triples[idx]
                    
                if np.all(counts[b_tris_idx] > 1):
                    counts[b_tris_idx] -= 1
                else:
                    new_selected.append(idx)
                    
            selected_blocks = new_selected
            
            # Pass 2 (if time permits)
            if time.time() < deadline - 5 and len(selected_blocks) > 0:
                counts = np.zeros(num_triples, dtype=np.int32)
                for idx in selected_blocks:
                    if inc_matrix is not None:
                        counts[inc_matrix[idx]] += 1
                    else:
                        for tri_idx in block_triples[idx]:
                            counts[tri_idx] += 1
                            
                new_selected = []
                for idx in selected_blocks:
                    if inc_matrix is not None:
                        b_tris_idx = np.where(inc_matrix[idx] == 1)[0]
                    else:
                        b_tris_idx = block_triples[idx]
                        
                    if np.all(counts[b_tris_idx] > 1):
                        counts[b_tris_idx] -= 1
                    else:
                        new_selected.append(idx)
                        
                selected_blocks = new_selected

        # Update best
        if len(selected_blocks) < best_count:
            best_count = len(selected_blocks)
            best_blocks = selected_blocks.copy()
            
    # Convert indices to blocks
    if best_blocks is None:
        # Fallback: Return a valid covering if we have one, else empty (invalid)
        # But we should always find one.
        return []
        
    final_blocks = [all_blocks[idx] for idx in best_blocks]
    
    return final_blocks
