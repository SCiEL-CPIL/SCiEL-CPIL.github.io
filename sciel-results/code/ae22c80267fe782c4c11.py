import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a Dynamic Scarcity Greedy approach.
    This is an Information-Theoretic method that prioritizes covering triples
    with the fewest remaining available blocks (highest scarcity).
    """
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    
    # Map each triple to the list of blocks that cover it
    # triple_to_blocks[t_idx] = list of block tuples
    triple_to_blocks = defaultdict(list)
    for block in all_blocks:
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            triple_to_blocks[idx].append(block)
            
    # Initial scarcity: number of blocks covering each triple
    scarcity = [0] * len(all_triples)
    for i in range(len(all_triples)):
        scarcity[i] = len(triple_to_blocks[i])
        
    uncovered = set(range(len(all_triples)))
    selected_blocks = set()
    blocks = []
    
    # To speed up "is this block selected", we use a set of tuples.
    
    while uncovered:
        # Find the uncovered triple with minimum scarcity
        # We can sample or just iterate. With 2024 triples, iteration is fine.
        min_scarcity = float('inf')
        best_triple_idx = -1
        
        # Optimization: If we have many iterations, we might want a priority queue.
        # But scarcity changes dynamically.
        # For now, simple scan.
        for idx in uncovered:
            if scarcity[idx] < min_scarcity:
                min_scarcity = scarcity[idx]
                best_triple_idx = idx
                
        if best_triple_idx == -1:
            break
            
        # Get candidate blocks for this triple
        candidates = triple_to_blocks[best_triple_idx]
        
        best_block = None
        best_gain = -1
        
        # We need to pick the block that maximizes the weighted gain
        # Weight = 1 / scarcity[T] for each uncovered T in the block
        
        for block in candidates:
            if block in selected_blocks:
                continue
                
            gain = 0.0
            # Calculate gain
            for triple in itertools.combinations(block, t):
                t_idx = triple_to_idx[triple]
                if t_idx in uncovered:
                    # Avoid division by zero, though scarcity should be >= 1
                    # unless we messed up logic.
                    gain += 1.0 / scarcity[t_idx]
                    
            if gain > best_gain:
                best_gain = gain
                best_block = block
                
        if best_block is None:
            # Fallback: pick a random candidate not selected
            for block in candidates:
                if block not in selected_blocks:
                    best_block = block
                    break
            if best_block is None:
                # Should not happen
                break
                
        # Select the block
        selected_blocks.add(best_block)
        blocks.append(list(best_block))
        
        # Update scarcity and uncovered set
        for triple in itertools.combinations(best_block, t):
            t_idx = triple_to_idx[triple]
            # Decrement scarcity for this triple
            # Note: This block is now "used", so it's one less option
            scarcity[t_idx] -= 1
            # Remove from uncovered
            uncovered.discard(t_idx)
            
    # Local Search: Remove redundant blocks
    blocks = _local_search(v, k, t, blocks)
    
    return blocks

def _local_search(v, k, t, blocks):
    """
    Local search to reduce the number of blocks by removing redundant ones.
    """
    blocks = list(blocks)
    all_triples = set(itertools.combinations(range(v), t))
    
    def is_valid(block_list):
        covered = set()
        for b in block_list:
            for triple in itertools.combinations(b, t):
                covered.add(triple)
        return covered == all_triples
        
    # Pass 1: Remove redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            trial = blocks[:i] + blocks[i+1:]
            if is_valid(trial):
                blocks = trial
                changed = True
                break
                
    return blocks
