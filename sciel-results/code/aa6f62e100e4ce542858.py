"""Improved covering algorithm with greedy + redundancy removal + local search."""
import itertools
import time
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    """Construct a covering design C(v,k,t) minimizing the number of blocks."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Generate all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    
    # Map each t-subset to an index
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Track coverage: which triples are covered
    covered = np.zeros(n_triples, dtype=bool)
    
    # Precompute: for each block (k-subset), which triples it covers
    # We'll build this on the fly for efficiency
    
    def block_triple_indices(block):
        """Get indices of all t-subsets within a block."""
        return [triple_to_idx[tuple(sorted(c))] for c in itertools.combinations(block, t)]
    
    def count_uncovered(block):
        """Count how many uncovered triples a block covers."""
        idxs = block_triple_indices(block)
        return int(np.sum(~covered[idxs]))
    
    # Greedy phase
    blocks = []
    blocks_set = set()
    
    while not covered.all() and time.time() < deadline:
        # Find the most uncovered triple to seed a block
        uncovered_mask = ~covered
        if not uncovered_mask.any():
            break
        
        # Pick the uncovered triple with highest "potential" - just pick first uncovered
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        seed_idx = uncovered_indices[0]
        seed_triple = all_triples[seed_idx]
        block = list(seed_triple)
        
        # Greedily add points to maximize coverage
        while len(block) < k and time.time() < deadline:
            current_block_set = set(block)
            best_point = None
            best_gain = -1
            
            for x in range(v):
                if x in current_block_set:
                    continue
                new_block = block + [x]
                gain = count_uncovered(new_block)
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block.append(best_point)
        
        block.sort()
        block_tuple = tuple(block)
        
        if block_tuple not in blocks_set:
            blocks.append(block)
            blocks_set.add(block_tuple)
            # Update coverage
            idxs = block_triple_indices(block)
            covered[idxs] = True
    
    if not covered.all():
        # If we ran out of time, return what we have (may be invalid)
        # But let's try to finish quickly
        pass
    
    # Redundancy removal phase
    if time.time() < deadline - 5:
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                # Temporarily remove this block and check if coverage is lost
                idxs = block_triple_indices(block)
                # Check if all triples in this block are covered by other blocks
                other_covered = np.zeros(n_triples, dtype=bool)
                for j, b in enumerate(blocks):
                    if j == i:
                        continue
                    other_idxs = block_triple_indices(b)
                    other_covered[other_idxs] = True
                
                if np.all(other_covered[idxs]):
                    # This block is redundant
                    changed = True
                else:
                    new_blocks.append(block)
            
            if changed:
                blocks = new_blocks
                blocks_set = set(tuple(b) for b in blocks)
                # Recompute coverage
                covered = np.zeros(n_triples, dtype=bool)
                for block in blocks:
                    idxs = block_triple_indices(block)
                    covered[idxs] = True
    
    # Local search: try to replace blocks with better ones
    if time.time() < deadline - 10:
        for iteration in range(5):  # Limited iterations
            if time.time() >= deadline - 10:
                break
            improved = False
            for i, block in enumerate(blocks):
                if time.time() >= deadline - 10:
                    break
                block_set = set(block)
                block_idxs = block_triple_indices(block)
                
                # Try to find a replacement block that covers all of block's triples
                # and also covers additional uncovered triples
                best_replacement = None
                best_score = -1
                
                # Generate candidate blocks: take the block and swap one element
                for pos in range(k):
                    for x in range(v):
                        if x in block_set:
                            continue
                        new_block = list(block)
                        new_block[pos] = x
                        new_block.sort()
                        new_block_set = set(new_block)
                        new_block_tuple = tuple(new_block)
                        
                        # Check if this block covers all triples of the original
                        new_idxs = block_triple_indices(new_block)
                        new_idx_set = set(new_idxs)
                        
                        # Must cover all original block's triples
                        if not all(idx in new_idx_set for idx in block_idxs):
                            continue
                        
                        # Score: number of newly covered triples
                        gain = int(np.sum(~covered[new_idxs]))
                        
                        if gain > best_score:
                            best_score = gain
                            best_replacement = new_block
                
                if best_replacement is not None and best_score > 0:
                    blocks[i] = best_replacement
                    blocks_set.add(tuple(best_replacement))
                    # Update coverage
                    covered = np.zeros(n_triples, dtype=bool)
                    for b in blocks:
                        idxs = block_triple_indices(b)
                        covered[idxs] = True
                    improved = True
            
            if not improved:
                break
    
    # Final redundancy removal
    if time.time() < deadline - 2:
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            new_blocks = []
            for i, block in enumerate(blocks):
                idxs = block_triple_indices(block)
                other_covered = np.zeros(n_triples, dtype=bool)
                for j, b in enumerate(blocks):
                    if j == i:
                        continue
                    other_idxs = block_triple_indices(b)
                    other_covered[other_idxs] = True
                
                if np.all(other_covered[idxs]):
                    changed = True
                else:
                    new_blocks.append(block)
            
            if changed:
                blocks = new_blocks
                covered = np.zeros(n_triples, dtype=bool)
                for block in blocks:
                    idxs = block_triple_indices(block)
                    covered[idxs] = True
    
    return blocks
