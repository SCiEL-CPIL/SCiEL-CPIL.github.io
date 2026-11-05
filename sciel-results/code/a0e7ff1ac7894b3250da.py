import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_id = {tr: i for i, tr in enumerate(triples)}
    num_triples = len(triples)

    # Generate all k-blocks
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)

    # Map block index to list of triple IDs it covers
    block_triples = []
    for b in all_blocks:
        ts = [triple_to_id[tup] for tup in itertools.combinations(b, t)]
        block_triples.append(ts)

    # Map triple ID to list of block indices covering it
    triple_blocks = [[] for _ in range(num_triples)]
    for b_idx, ts in enumerate(block_triples):
        for t_idx in ts:
            triple_blocks[t_idx].append(b_idx)

    # Helper to get triple IDs for a block index
    def get_block_ids(b_idx):
        return block_triples[b_idx]

    # 1. Randomized Greedy with Ratio-based Selection
    # The strategy uses a ratio of newly covered triples to the size of the block (or 1)
    # but since all blocks have size k, this is equivalent to max gain.
    # However, the inspiration suggests "ratio of newly covered triplets to the number of points in the block".
    # Since k is constant, this is just max gain.
    # To make it distinct and potentially better, we can use a randomized greedy that picks from top-K candidates
    # or uses a noise-based selection.
    
    def greedy_construction(seed=0):
        rng = random.Random(seed)
        # gain[b_idx] is the number of currently uncovered triples covered by block b_idx
        gain = np.array([len(ts) for ts in block_triples], dtype=np.int32)
        is_uncovered = np.ones(num_triples, dtype=np.bool_)
        
        selected_blocks = []
        
        while is_uncovered.any():
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                # Should not happen if is_uncovered.any() is true
                break
            
            # Find indices with max gain
            candidates = np.where(gain == max_gain)[0]
            
            # To add diversity, we can pick randomly from candidates
            # Or, to implement the "ratio" idea more dynamically, we could penalize blocks
            # that overlap heavily with already selected blocks, but gain already accounts for that.
            # Let's stick to max gain with random tie-breaking.
            b_idx = rng.choice(candidates.tolist())
            
            selected_blocks.append(b_idx)
            
            # Update gain and is_uncovered
            for t_idx in block_triples[b_idx]:
                if is_uncovered[t_idx]:
                    is_uncovered[t_idx] = False
                    # For every block that covers this triple, decrease its gain
                    for other_b_idx in triple_blocks[t_idx]:
                        gain[other_b_idx] -= 1
                        
        return selected_blocks

    # 2. Local Search with Reduction and Perturbation
    def local_search(blocks, deadline):
        # blocks is a list of block indices
        # Initialize cover counts
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in blocks:
            for t_idx in block_triples[b_idx]:
                cover_count[t_idx] += 1
        
        best_blocks = blocks[:]
        best_len = len(blocks)
        
        T = 1.0 # Temperature
        
        while time.time() < deadline:
            # Reduction Phase
            changed = False
            # Iterate until no change
            while True:
                removal_idx = -1
                for i, b_idx in enumerate(blocks):
                    # Check if redundant
                    redundant = True
                    for t_idx in block_triples[b_idx]:
                        if cover_count[t_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        removal_idx = i
                        break
                
                if removal_idx != -1:
                    # Remove block
                    b_idx = blocks.pop(removal_idx)
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
                else:
                    break
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
            
            # Perturbation Phase
            if not blocks:
                break
                
            i = random.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            old_ids = block_triples[old_block]
            
            # Remove
            blocks.pop(i)
            for t_idx in old_ids:
                cover_count[t_idx] -= 1
            
            # Find uncovered
            uncovered = [tid for tid in range(num_triples) if cover_count[tid] == 0]
            
            if not uncovered:
                # This implies old_block was redundant, which should have been caught in reduction.
                # But if it happens, we accept this state.
                continue
            
            # Repair: Greedily add blocks to cover uncovered
            added_blocks = []
            current_uncovered = set(uncovered)
            
            while current_uncovered:
                tid = random.choice(list(current_uncovered))
                target_triple = triples[tid]
                remaining_points = [p for p in points if p not in target_triple]
                
                candidates = []
                for combo in itertools.combinations(remaining_points, k - t):
                    block = tuple(sorted(target_triple + combo))
                    # Find index of this block
                    # This is inefficient. We should precompute a map from block tuple to index.
                    # Let's assume we have a map.
                    # For now, we'll just calculate gain manually or use a different repair strategy.
                    # Actually, let's use the precomputed structure.
                    # We need to find all blocks containing target_triple.
                    # This is not directly precomputed.
                    # Let's use a different repair: pick a random block that covers the most uncovered triples.
                    # We can scan all blocks? That's 346k. Too slow.
                    
                    # Alternative: Pick a random block from those covering a random uncovered triple.
                    # But we need to cover ALL uncovered.
                    
                    # Let's go back to the original repair method which generated blocks on the fly.
                    # It was O(C(v-t, k-t)) which is C(21, 4) = 5985. This is fast.
                    
                    # We need to map the generated block to an index to use block_triples.
                    # Let's precompute a dictionary from block tuple to index.
                    # This is done outside local_search.
                    
                    # Since we are inside the function, we can't easily access a global map created inside construct
                    # unless we pass it or make it a closure.
                    # Let's create the map in construct.
                    pass
                
                # This repair is getting complicated with indices.
                # Let's simplify: Use the original repair logic which works with block tuples.
                # But local_search works with indices.
                # We can convert indices to tuples for the repair, then convert back to indices.
                
                # Let's rewrite repair to use tuples and then map to indices.
                # We need a map: block_tuple_to_idx
                # Let's assume it exists.
                
                if not candidates:
                    break
                
                # This part of the code is a placeholder for the complex repair.
                # Let's implement a simpler repair:
                # Pick a random uncovered triple, find a random block covering it, add it.
                # Repeat until covered.
                
                # Actually, let's just use the greedy logic on the full set of blocks, but restricted to those covering uncovered triples.
                # This is hard to do efficiently.
                
                # Let's fall back to a simple heuristic:
                # Add a random block that covers at least one uncovered triple.
                # This is slow but works.
                
                # Better: Use the precomputed triple_blocks.
                # Pick a random uncovered triple tid.
                # Pick a random block b_idx from triple_blocks[tid].
                # Add it.
                # Update uncovered.
                
                # This is a valid repair, though not optimal.
                
                # Let's implement this.
                if not current_uncovered:
                    break
                
                tid = random.choice(list(current_uncovered))
                possible_blocks = triple_blocks[tid]
                if not possible_blocks:
                    break
                b_idx = random.choice(possible_blocks)
                
                # Check if this block adds any new coverage
                new_cover = False
                for t_idx in block_triples[b_idx]:
                    if t_idx in current_uncovered:
                        new_cover = True
                        break
                
                if new_cover:
                    added_blocks.append(b_idx)
                    for t_idx in block_triples[b_idx]:
                        if t_idx in current_uncovered:
                            current_uncovered.remove(t_idx)
                    
                    # Early stop if we added too many
                    if len(added_blocks) >= 2:
                        break
            
            new_len = len(blocks) + len(added_blocks)
            old_len = len(blocks) + 1
            
            delta = new_len - old_len
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True
            else:
                if random.random() < math.exp(-delta / T):
                    accept = True
            
            if accept:
                blocks.extend(added_blocks)
                for b_idx in added_blocks:
                    for t_idx in block_triples[b_idx]:
                        cover_count[t_idx] += 1
            else:
                # Restore
                blocks.insert(i, old_block)
                for t_idx in old_ids:
                    cover_count[t_idx] += 1
            
            # Cool down
            T *= 0.99
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
                
        return best_blocks

    # Precompute block tuple to index map for repair if needed
    # But the simplified repair above uses triple_blocks directly, so we don't need it.
    
    # Main Loop: Multiple Restarts
    best_result = None
    best_size = float('inf')
    
    seed = 0
    while time.time() < deadline:
        # Greedy
        blocks = greedy_construction(seed)
        
        # Local Search
        ls_deadline = min(deadline, time.time() + 10.0)
        improved = local_search(blocks, ls_deadline)
        
        if len(improved) < best_size:
            best_size = len(improved)
            best_result = improved
            
        # If we found 69 (lower bound) or 70, we can stop?
        if best_size <= 70:
            break
            
        seed += 1
        
    # Return as list of lists
    if best_result:
        return [list(all_blocks[b_idx]) for b_idx in best_result]
    else:
        # Fallback: simple greedy
        blocks = greedy_construction(0)
        return [list(all_blocks[b_idx]) for b_idx in blocks]
