import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin for 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets (triples) and map them to indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Generate all k-subsets (blocks) and map them to indices
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}
    num_blocks = len(all_blocks)
    
    # Precompute for each block, which triples it covers
    # block_triples[i] = list of triple indices covered by block i
    block_triples = [[] for _ in range(num_blocks)]
    # Precompute for each triple, which blocks cover it
    # triple_blocks[j] = list of block indices covering triple j
    triple_blocks = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[b_idx].append(t_idx)
            triple_blocks[t_idx].append(b_idx)
            
    # 2. Greedy Construction
    # uncovered: set of triple indices that are not yet covered
    uncovered = set(range(num_triples))
    solution_blocks = [] # List of block indices
    
    while uncovered:
        if time.time() > deadline - 10: # Stop if we don't have enough time for local search
            break
            
        # Pick an uncovered triple to force cover
        # Using min for determinism, or random for diversity? 
        # Let's use the first one for speed, or random to avoid bias.
        # Random is better for escaping local minima in greedy? 
        # Greedy is usually deterministic. Let's stick to deterministic for reproducibility of the initial state, 
        # but the local search will provide diversity.
        target_triple = next(iter(uncovered))
        
        # Get candidate blocks that cover this triple
        candidates = triple_blocks[target_triple]
        
        best_block_idx = -1
        max_gain = -1
        
        for b_idx in candidates:
            # Calculate gain: number of uncovered triples covered by this block
            gain = 0
            for t_idx in block_triples[b_idx]:
                if t_idx in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
                # Early termination if we cover all possible? 
                # Max possible is len(block_triples[b_idx]) which is C(k,t)
                if gain == len(block_triples[b_idx]):
                    break
                    
        if best_block_idx == -1:
            # Should not happen if uncovered is not empty
            break
            
        solution_blocks.append(best_block_idx)
        # Update uncovered
        for t_idx in block_triples[best_block_idx]:
            uncovered.discard(t_idx)
            
    # If we ran out of time during greedy, we might have an invalid cover.
    # We must ensure validity. 
    # If uncovered is not empty, we need to finish it quickly.
    if uncovered:
         # Force finish: just add blocks until covered, ignoring optimality
         while uncovered and time.time() < deadline - 5:
             target_triple = next(iter(uncovered))
             # Pick first candidate
             b_idx = triple_blocks[target_triple][0]
             solution_blocks.append(b_idx)
             for t_idx in block_triples[b_idx]:
                 uncovered.discard(t_idx)
         
         # If still uncovered (extreme timeout), we might return invalid. 
         # But greedy should be fast enough.

    # Convert solution to set of block indices for faster lookup
    solution_set = set(solution_blocks)
    
    # 3. Local Search
    # Initialize cover counts
    cover_count = [0] * num_triples
    for b_idx in solution_set:
        for t_idx in block_triples[b_idx]:
            cover_count[t_idx] += 1
            
    def is_redundant(b_idx):
        for t_idx in block_triples[b_idx]:
            if cover_count[t_idx] <= 1:
                return False
        return True

    def remove_block(b_idx):
        solution_set.remove(b_idx)
        for t_idx in block_triples[b_idx]:
            cover_count[t_idx] -= 1

    def add_block(b_idx):
        solution_set.add(b_idx)
        for t_idx in block_triples[b_idx]:
            cover_count[t_idx] += 1

    # Local search loop
    while time.time() < deadline - 5:
        progress = False
        
        # Pass 1: Remove redundant blocks
        # Iterate over a copy of the set to allow removal
        for b_idx in list(solution_set):
            if is_redundant(b_idx):
                remove_block(b_idx)
                progress = True
        
        # Pass 2: Swaps
        # Try to swap a block B with B' such that B' covers all essential triples of B
        for b_idx in list(solution_set):
            # Find essential triples for b_idx
            essential = []
            for t_idx in block_triples[b_idx]:
                if cover_count[t_idx] == 1:
                    essential.append(t_idx)
            
            if not essential:
                # Should have been removed in Pass 1
                continue
                
            # Find intersection of blocks covering all essential triples
            # Start with blocks covering the first essential triple
            # Sort essential by number of candidate blocks to minimize intersection size?
            # triple_blocks[t_idx] size is constant ~6000.
            
            # Optimization: If len(essential) > 3, intersection is likely empty or very small.
            # If len(essential) is large, it's hard to find a replacement.
            
            candidates = set(triple_blocks[essential[0]])
            
            for t_idx in essential[1:]:
                # Intersect with next
                # Using set intersection
                candidates.intersection_update(triple_blocks[t_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
                
            # Remove the current block from candidates
            candidates.discard(b_idx)
            
            if not candidates:
                continue
                
            # Choose best candidate
            # Heuristic: Maximize overlap with existing coverage (to make it redundant later)
            # Or minimize "new" uncovered triples? No, we maintain validity.
            # Let's pick the one that maximizes the number of triples t in B' where cover_count[t] > 1.
            best_candidate = -1
            max_overlap = -1
            
            # To speed up, we can just pick the first one if we are in a hurry.
            # But let's try a few.
            # If candidates is large, iterating all is slow.
            # Let's limit to checking a random subset if candidates is too big?
            # Or just pick one at random?
            # Random selection might be better for escaping local minima.
            
            if len(candidates) > 100:
                # Pick a random candidate
                best_candidate = random.choice(list(candidates))
            else:
                for cand in candidates:
                    overlap = 0
                    for t_idx in block_triples[cand]:
                        if cover_count[t_idx] > 1:
                            overlap += 1
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_candidate = cand
                        
            if best_candidate != -1:
                # Perform swap
                remove_block(b_idx)
                add_block(best_candidate)
                progress = True
                
    # Final check for validity (should be valid if we started with valid and did valid swaps/removals)
    # Construct the final list of blocks
    final_blocks = [all_blocks[b_idx] for b_idx in solution_set]
    
    # Sort blocks for consistent output
    final_blocks.sort()
    
    return final_blocks
