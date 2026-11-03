import itertools
import random
import heapq
import numpy as np

def construct(v, k, t):
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    triple_to_index = {trip: i for i, trip in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Pre-compute all blocks and their triple lists
    # And reverse mapping: triple -> list of blocks
    all_blocks = []
    block_to_triples = []
    triple_to_blocks = [[] for _ in range(num_triples)]
    
    for block in itertools.combinations(points, k):
        block_idx = len(all_blocks)
        all_blocks.append(block)
        trips = []
        for trip in itertools.combinations(block, t):
            idx = triple_to_index[trip]
            trips.append(idx)
            triple_to_blocks[idx].append(block_idx)
        block_to_triples.append(trips)
        
    num_blocks = len(all_blocks)
    
    # Pre-compute static degree of each triple (number of blocks covering it)
    # This helps in selecting the "most constrained" uncovered triple
    triple_degree = [len(triple_to_blocks[i]) for i in range(num_triples)]
    
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 10 # Fewer restarts needed if heuristic is stronger
    
    for restart in range(num_restarts):
        random.seed(restart + 100) # Different seed range
        
        uncovered = set(range(num_triples))
        blocks = []
        
        # Greedy Construction with "Most Constrained Triple" heuristic
        # We use a heap to efficiently find the uncovered triple with min degree.
        # Note: degree is static, so we can just iterate and find min, 
        # or maintain a heap of (degree, triple_idx) and lazy-delete.
        
        # Since num_triples is ~2000, finding min by scanning is O(2000) which is fast.
        # Scanning 2000 items 76 times is ~150k ops. Negligible.
        
        while uncovered:
            # Find uncovered triple with minimum degree
            min_deg = float('inf')
            target_idx = -1
            for t_idx in uncovered:
                deg = triple_degree[t_idx]
                if deg < min_deg:
                    min_deg = deg
                    target_idx = t_idx
            
            # Find best block covering target_idx
            best_block_idx = -1
            max_covered = -1
            
            for b_idx in triple_to_blocks[target_idx]:
                covered = 0
                for t_idx in block_to_triples[b_idx]:
                    if t_idx in uncovered:
                        covered += 1
                
                if covered > max_covered:
                    max_covered = covered
                    best_block_idx = b_idx
                    if covered == len(block_to_triples[b_idx]):
                        break
            
            if best_block_idx != -1:
                blocks.append(best_block_idx)
                for t_idx in block_to_triples[best_block_idx]:
                    uncovered.discard(t_idx)
        
        # Refinement 1: Deletion
        triple_count = [0] * num_triples
        for b_idx in blocks:
            for t_idx in block_to_triples[b_idx]:
                triple_count[t_idx] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks:
                essential = False
                for t_idx in block_to_triples[b_idx]:
                    if triple_count[t_idx] == 1:
                        essential = True
                        break
                if not essential:
                    for t_idx in block_to_triples[b_idx]:
                        triple_count[t_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
        
        # Refinement 2: 1-Swap Local Search
        # Try to swap blocks to enable further deletions
        in_cover = set(blocks)
        improved = True
        max_swap_iters = 20
        while improved and max_swap_iters > 0:
            improved = False
            max_swap_iters -= 1
            
            # Re-count for deletion check after swaps
            triple_count = [0] * num_triples
            for b_idx in blocks:
                for t_idx in block_to_triples[b_idx]:
                    triple_count[t_idx] += 1
            
            # First, try simple deletions again
            new_blocks = []
            for b_idx in blocks:
                essential = False
                for t_idx in block_to_triples[b_idx]:
                    if triple_count[t_idx] == 1:
                        essential = True
                        break
                if not essential:
                    for t_idx in block_to_triples[b_idx]:
                        triple_count[t_idx] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
            in_cover = set(blocks)
            
            if not blocks:
                break

            # Now try swaps
            # For each block in cover, see if we can swap it with another block
            # such that the new set is still a cover.
            # We only swap if it doesn't increase size, but might help structure.
            # Actually, swapping keeps size same. It only helps if it allows a future deletion.
            # To check if a swap is "safe", we need the new set to be a cover.
            
            for i, b_old in enumerate(blocks):
                # Temporarily remove b_old
                temp_blocks = blocks[:i] + blocks[i+1:]
                temp_in_cover = set(temp_blocks)
                
                # Find uncovered triples
                # We need to know which triples are covered ONLY by b_old
                # i.e., triples t in b_old where count(t) == 1 in current cover
                critical_triples = []
                for t_idx in block_to_triples[b_old]:
                    if triple_count[t_idx] == 1:
                        critical_triples.append(t_idx)
                
                if not critical_triples:
                    # b_old is redundant, should have been deleted in previous pass
                    continue
                
                # Find a block b_new not in cover that covers all critical_triples
                # Candidate blocks must cover at least one critical triple
                candidates = set()
                for t_idx in critical_triples:
                    for b_idx in triple_to_blocks[t_idx]:
                        if b_idx not in temp_in_cover:
                            candidates.add(b_idx)
                
                found_swap = False
                for b_new in candidates:
                    # Check if b_new covers ALL critical_triples
                    new_triples = set(block_to_triples[b_new])
                    if all(t in new_triples for t in critical_triples):
                        # Perform swap
                        blocks[i] = b_new
                        temp_in_cover.add(b_new)
                        temp_in_cover.remove(b_old)
                        
                        # Update triple_count
                        for t_idx in block_to_triples[b_old]:
                            triple_count[t_idx] -= 1
                        for t_idx in block_to_triples[b_new]:
                            triple_count[t_idx] += 1
                        
                        in_cover = temp_in_cover
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    # Break out of for loop to restart scan? 
                    # Or just continue? Restarting is safer for consistency.
                    break
            
            if found_swap:
                # If we swapped, we should re-evaluate deletions
                continue
            else:
                break # No swaps found
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(all_blocks[b]) for b in blocks]
            
    return best_blocks
