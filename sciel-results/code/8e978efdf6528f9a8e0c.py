import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s margin for safety
    
    # Precompute all triplets
    all_triplets = list(itertools.combinations(range(v), t))
    triplet_to_idx = {triplet: i for i, triplet in enumerate(all_triplets)}
    n_triplets = len(all_triplets)
    
    # For each point, precompute which triplets contain it
    point_triplets = defaultdict(list)
    for i, triplet in enumerate(all_triplets):
        for p in triplet:
            point_triplets[p].append(i)
    
    best_blocks = None
    best_size = float('inf')
    
    def greedy_run():
        uncovered = [True] * n_triplets
        blocks = []
        
        while True:
            # Check if all covered
            if not any(uncovered):
                break
            
            # Pick a random uncovered triplet
            uncovered_indices = [i for i, u in enumerate(uncovered) if u]
            if not uncovered_indices:
                break
            seed_idx = random.choice(uncovered_indices)
            seed_triplet = all_triplets[seed_idx]
            
            # Build block containing seed_triplet
            block = list(seed_triplet)
            block_set = set(block)
            
            # Greedily add k - t points
            for _ in range(k - t):
                best_gain = -1
                best_point = -1
                best_gain_ties = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Calculate gain: number of uncovered triplets containing x and 2 points from block
                    gain = 0
                    for a, b in itertools.combinations(block, 2):
                        triplet = tuple(sorted((x, a, b)))
                        idx = triplet_to_idx.get(triplet)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        best_gain_ties = [x]
                    elif gain == best_gain and gain > 0:
                        best_gain_ties.append(x)
                
                if best_point != -1:
                    # Random tie-breaking
                    if len(best_gain_ties) > 1:
                        best_point = random.choice(best_gain_ties)
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    break
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                idx = triplet_to_idx.get(triplet)
                if idx is not None:
                    uncovered[idx] = False
        
        return blocks
    
    def reduce_blocks(blocks):
        if not blocks:
            return blocks
        
        block_sets = [set(b) for b in blocks]
        changed = True
        while changed:
            changed = False
            random.shuffle(block_sets)
            for i in range(len(block_sets)):
                if i >= len(block_sets):
                    break
                b = block_sets[i]
                is_redundant = True
                for triplet in itertools.combinations(b, t):
                    covered = False
                    for j in range(len(block_sets)):
                        if i == j:
                            continue
                        if set(triplet).issubset(block_sets[j]):
                            covered = True
                            break
                    if not covered:
                        is_redundant = False
                        break
                
                if is_redundant:
                    block_sets.pop(i)
                    changed = True
                    break
        
        return [list(b) for b in block_sets]
    
    def local_search(blocks, max_iter=50):
        """Try to improve by removing blocks that become redundant after adding others"""
        blocks = [list(b) for b in blocks]
        block_sets = [set(b) for b in blocks]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try removing each block and see if we can compensate
            improved = False
            for i in range(len(block_sets)):
                if time.time() > deadline:
                    break
                
                # Temporarily remove block i
                removed = block_sets.pop(i)
                
                # Find uncovered triplets
                uncovered = []
                for triplet in itertools.combinations(removed, t):
                    covered = False
                    for j in range(len(block_sets)):
                        if set(triplet).issubset(block_sets[j]):
                            covered = True
                            break
                    if not covered:
                        uncovered.append(triplet)
                
                if not uncovered:
                    # Block was redundant
                    improved = True
                    break
                
                # Try to add a new block that covers as many uncovered as possible
                best_new_block = None
                best_coverage = 0
                
                # Try random blocks containing some uncovered triplets
                for _ in range(20):
                    if not uncovered:
                        break
                    seed = random.choice(uncovered)
                    new_block = list(seed)
                    new_block_set = set(new_block)
                    
                    for _ in range(k - t):
                        best_gain = -1
                        best_point = -1
                        for x in range(v):
                            if x in new_block_set:
                                continue
                            gain = 0
                            for triplet in uncovered:
                                if x in triplet and set(triplet) & new_block_set:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_point = x
                        if best_point != -1:
                            new_block.append(best_point)
                            new_block_set.add(best_point)
                    
                    new_block.sort()
                    # Count how many uncovered triplets this covers
                    coverage = 0
                    for triplet in uncovered:
                        if set(triplet).issubset(new_block_set):
                            coverage += 1
                    
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_new_block = new_block
                
                if best_new_block is not None and best_coverage >= len(uncovered):
                    # Add the new block
                    block_sets.append(set(best_new_block))
                    improved = True
                    break
                else:
                    # Restore
                    block_sets.insert(i, removed)
            
            if not improved:
                break
        
        return [list(b) for b in block_sets]
    
    # Main loop
    while time.time() < deadline:
        # Run greedy
        blocks = greedy_run()
        
        # Reduce
        reduced_blocks = reduce_blocks(blocks)
        
        # Local search
        improved_blocks = local_search(reduced_blocks)
        
        if len(improved_blocks) < best_size:
            best_size = len(improved_blocks)
            best_blocks = improved_blocks
        
        # If we're close to optimal, spend more time on local search
        if best_size <= 78 and time.time() < deadline - 30:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                more_improved = local_search(best_blocks, max_iter=100)
                if len(more_improved) < best_size:
                    best_size = len(more_improved)
                    best_blocks = more_improved
    
    if best_blocks is None:
        return greedy_run()
    
    return best_blocks
