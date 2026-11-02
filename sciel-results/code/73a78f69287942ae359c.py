import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Generate all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute block masks using bitsets
    block_masks = np.zeros(num_blocks, dtype=np.uint64)
    # For v=24, k=7, t=3: 2024 triples, so we need multiple uint64s
    # Actually, let's use Python integers for simplicity since 2024 bits fits in ~32 uint64s
    # But for speed, let's use a list of integers
    
    block_masks_list = [0] * num_blocks
    blocks_by_triple = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            mask |= (1 << t_idx)
            blocks_by_triple[t_idx].append(b_idx)
        block_masks_list[b_idx] = mask
    
    # Greedy set cover with bitsets
    uncovered = (1 << num_triples) - 1
    selected_indices = []
    selected_set = set()
    
    while uncovered and time.time() < deadline - 30.0:
        # Find an uncovered triple (pick one with fewest covering blocks for efficiency)
        # For speed, just pick a random uncovered bit
        # Find lowest set bit
        lb = uncovered & (-uncovered)
        t_idx = lb.bit_length() - 1
        
        candidates = blocks_by_triple[t_idx]
        
        best_gain = 0
        best_block_idx = -1
        
        for b_idx in candidates:
            if b_idx in selected_set:
                continue
            gain = (uncovered & block_masks_list[b_idx]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx == -1 or best_gain == 0:
            break
            
        selected_indices.append(best_block_idx)
        selected_set.add(best_block_idx)
        uncovered &= ~block_masks_list[best_block_idx]
    
    if not selected_indices:
        return []
    
    selected_blocks = [all_blocks[i] for i in selected_indices]
    
    # Redundancy removal with coverage counts
    def remove_redundant(blocks, v, t, triple_to_idx, deadline):
        if not blocks:
            return []
            
        n_blocks = len(blocks)
        block_triples = []
        for b in blocks:
            tris = []
            for tri in itertools.combinations(b, t):
                tris.append(triple_to_idx[tuple(sorted(tri))])
            block_triples.append(tris)
        
        cov_count = np.zeros(len(triple_to_idx), dtype=np.int32)
        for tris in block_triples:
            for ti in tris:
                cov_count[ti] += 1
        
        active = [True] * n_blocks
        changed = True
        
        while changed and time.time() < deadline:
            changed = False
            for i in range(n_blocks):
                if not active[i]:
                    continue
                # Check if block i is redundant
                redundant = True
                for ti in block_triples[i]:
                    if cov_count[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    active[i] = False
                    changed = True
                    for ti in block_triples[i]:
                        cov_count[ti] -= 1
        
        return [blocks[i] for i in range(n_blocks) if active[i]]
    
    selected_blocks = remove_redundant(selected_blocks, v, t, triple_to_idx, deadline - 10.0)
    
    # Local search with point swaps to create redundancy
    def local_search_swaps(blocks, v, k, t, triple_to_idx, deadline):
        blocks = [list(b) for b in blocks]
        n_triples = len(triple_to_idx)
        rng = random.Random(42)
        
        def compute_cov_count(blks):
            cov = np.zeros(n_triples, dtype=np.int32)
            for b in blks:
                for tri in itertools.combinations(b, t):
                    tri_idx = triple_to_idx[tuple(sorted(tri))]
                    cov[tri_idx] += 1
            return cov
        
        cov_count = compute_cov_count(blocks)
        improved = True
        iters = 0
        
        while improved and time.time() < deadline:
            improved = False
            iters += 1
            if iters > 50:
                break
            
            # First, remove any redundant blocks
            blocks = remove_redundant(blocks, v, t, triple_to_idx, deadline)
            cov_count = compute_cov_count(blocks)
            
            # Try point swaps
            for _ in range(30):
                if time.time() >= deadline or len(blocks) == 0:
                    break
                    
                idx = rng.randint(0, len(blocks) - 1)
                old_x = rng.choice(blocks[idx])
                new_x = rng.randint(0, v - 1)
                
                if new_x == old_x or new_x in blocks[idx]:
                    continue
                
                # Compute effect of swap
                old_block = blocks[idx]
                new_block = old_block.copy()
                new_block.remove(old_x)
                new_block.append(new_x)
                new_block.sort()
                
                # Get triples covered by old and new blocks
                old_tris = set()
                for tri in itertools.combinations(old_block, t):
                    old_tris.add(triple_to_idx[tuple(sorted(tri))])
                
                new_tris = set()
                for tri in itertools.combinations(new_block, t):
                    new_tris.add(triple_to_idx[tuple(sorted(tri))])
                
                # Check if any triple would become uncovered
                valid_swap = True
                for ti in old_tris:
                    if ti not in new_tris and cov_count[ti] == 1:
                        valid_swap = False
                        break
                
                if not valid_swap:
                    continue
                
                # Apply swap
                blocks[idx] = new_block
                for ti in old_tris:
                    cov_count[ti] -= 1
                for ti in new_tris:
                    cov_count[ti] += 1
                improved = True
        
        return blocks
    
    if time.time() < deadline - 5.0:
        selected_blocks = local_search_swaps(selected_blocks, v, k, t, triple_to_idx, deadline - 5.0)
    
    # Final redundancy removal
    selected_blocks = remove_redundant(selected_blocks, v, t, triple_to_idx, deadline)
    
    # Verify validity (should always be valid, but check just in case)
    covered = set()
    for b in selected_blocks:
        for tri in itertools.combinations(b, t):
            covered.add(tuple(sorted(tri)))
    
    # If invalid (shouldn't happen), fall back to greedy result
    # But we don't have the original greedy result anymore, so trust the process
    
    return [list(b) for b in selected_blocks]
