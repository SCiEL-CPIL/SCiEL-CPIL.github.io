import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precomputation
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    
    # Map triple index to mask (uint64)
    triple_masks = np.array([sum(1 << p for p in tri) for tri in all_triples], dtype=np.uint64)
    
    best_blocks = None
    best_score = float('inf')
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        uncovered_mask = np.ones(num_triples, dtype=bool)
        blocks = []
        
        while np.any(uncovered_mask):
            # Pick a random uncovered triple as seed
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            seed_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            seed_triple = all_triples[seed_idx]
            
            block = list(seed_triple)
            block_mask = np.uint64(sum(1 << p for p in seed_triple))
            
            # Greedily add points
            while len(block) < k:
                gains = np.zeros(v)
                for x in range(v):
                    if x in block:
                        continue
                    new_mask = block_mask | np.uint64(1 << x)
                    # Count uncovered triples t such that t is subset of new_mask
                    # Condition: (triple_mask & ~new_mask) == 0
                    covered = (triple_masks & ~new_mask) == 0
                    gain = np.sum(uncovered_mask & covered)
                    gains[x] = gain
                
                max_gain = np.max(gains)
                # Random tie-breaking
                candidates = [i for i in range(v) if gains[i] == max_gain and i not in block]
                if not candidates:
                    break
                best_x = random.choice(candidates)
                
                block.append(best_x)
                block_mask |= np.uint64(1 << best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered mask
            covered = (triple_masks & ~block_mask) == 0
            uncovered_mask[covered] = False
        
        # Post-processing: Remove redundant blocks
        if blocks:
            coverage_counts = np.zeros(num_triples, dtype=int)
            for b in blocks:
                b_mask = np.uint64(sum(1 << p for p in b))
                covered = (triple_masks & ~b_mask) == 0
                coverage_counts[covered] += 1
            
            new_blocks = []
            for b in blocks:
                b_mask = np.uint64(sum(1 << p for p in b))
                covered = (triple_masks & ~b_mask) == 0
                # If any triple is covered ONLY by this block, we must keep it
                if np.any(coverage_counts[covered] == 1):
                    new_blocks.append(b)
                else:
                    # This block can be removed
                    coverage_counts[covered] -= 1
            
            blocks = new_blocks
            
            if len(blocks) < best_score:
                best_score = len(blocks)
                best_blocks = [list(b) for b in blocks]
                
    if best_blocks is None:
        # Fallback
        uncovered_mask = np.ones(num_triples, dtype=bool)
        blocks = []
        while np.any(uncovered_mask):
            uncovered_indices = np.where(uncovered_mask)[0]
            seed_idx = uncovered_indices[0]
            seed_triple = all_triples[seed_idx]
            block = list(seed_triple)
            block_mask = np.uint64(sum(1 << p for p in seed_triple))
            while len(block) < k:
                gains = np.zeros(v)
                for x in range(v):
                    if x in block:
                        continue
                    new_mask = block_mask | np.uint64(1 << x)
                    covered = (triple_masks & ~new_mask) == 0
                    gain = np.sum(uncovered_mask & covered)
                    gains[x] = gain
                max_gain = np.max(gains)
                candidates = [i for i in range(v) if gains[i] == max_gain and i not in block]
                best_x = min(candidates)
                block.append(best_x)
                block_mask |= np.uint64(1 << best_x)
            block.sort()
            blocks.append(block)
            covered = (triple_masks & ~block_mask) == 0
            uncovered_mask[covered] = False
        best_blocks = blocks

    return best_blocks
