import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135  # 15s margin before 150s kill
    
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_set = set(all_tsubs)
    n_tsubs = len(all_tsubs)
    
    # Map t-subset to index for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    best_blocks = None
    best_len = float('inf')
    
    def block_tsubs(block):
        return set(itertools.combinations(block, t))
    
    def block_tsub_indices(block):
        return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    def remove_redundant(blocks):
        """Remove redundant blocks iteratively."""
        blocks = [list(b) for b in blocks]
        if not blocks:
            return blocks
        
        # Build coverage count using indices
        cov = np.zeros(n_tsubs, dtype=np.int32)
        block_idx_lists = []
        for b in blocks:
            idxs = block_tsub_indices(b)
            block_idx_lists.append(idxs)
            for idx in idxs:
                cov[idx] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if i >= len(blocks):
                    break
                # Check if all t-subsets in block i have coverage > 1
                removable = True
                for idx in block_idx_lists[i]:
                    if cov[idx] == 1:
                        removable = False
                        break
                if removable:
                    for idx in block_idx_lists[i]:
                        cov[idx] -= 1
                    blocks.pop(i)
                    block_idx_lists.pop(i)
                    changed = True
                    break
        
        return blocks
    
    def greedy_random_pivot():
        """Randomized greedy construction."""
        uncovered = set(all_tsubs_set)
        blocks = []
        
        while uncovered:
            pivot_triple = random.choice(list(uncovered))
            pivot_points = set(pivot_triple)
            other_points = [p for p in range(v) if p not in pivot_points]
            
            best_block = None
            best_gain = -1
            
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                gain = sum(1 for tri in itertools.combinations(block, t) if tri in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is not None:
                blocks.append(list(best_block))
                for tri in itertools.combinations(best_block, t):
                    if tri in uncovered:
                        uncovered.remove(tri)
                        
        return blocks
    
    def simulated_annealing(blocks, max_time=30):
        """SA to minimize block count by swapping/removing blocks."""
        if time.time() > start_time + max_time:
            return blocks
        
        blocks = [list(b) for b in blocks]
        n = len(blocks)
        
        # Precompute t-subset indices for each block
        block_tris = [block_tsub_indices(b) for b in blocks]
        
        # Build coverage count
        cov = np.zeros(n_tsubs, dtype=np.int32)
        for tris in block_tris:
            for idx in tris:
                cov[idx] += 1
        
        # Check initial validity
        if np.any(cov == 0):
            return blocks  # Not valid, return as-is
        
        T = 100.0  # Initial temperature
        cooling = 0.97
        best_blocks = [list(b) for b in blocks]
        best_n = n
        no_improve_count = 0
        
        while time.time() < start_time + max_time and T > 0.5:
            if n < best_n:
                best_n = n
                best_blocks = [list(b) for b in blocks]
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count > 500:
                break
            
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.4 and n > 1:
                # Try removing a block
                i = random.randint(0, n - 1)
                removable = True
                for idx in block_tris[i]:
                    if cov[idx] == 1:
                        removable = False
                        break
                
                if removable:
                    for idx in block_tris[i]:
                        cov[idx] -= 1
                    blocks.pop(i)
                    block_tris.pop(i)
                    n -= 1
                else:
                    # Try to replace block i
                    only_tris = [idx for idx in block_tris[i] if cov[idx] == 1]
                    
                    if not only_tris:
                        for idx in block_tris[i]:
                            cov[idx] -= 1
                        blocks.pop(i)
                        block_tris.pop(i)
                        n -= 1
                    else:
                        # Collect points that must be in replacement
                        must_include = set()
                        for idx in only_tris:
                            must_include.update(all_tsubs[idx])
                        
                        if len(must_include) <= k:
                            remaining_points = [p for p in range(v) if p not in must_include]
                            need = k - len(must_include)
                            
                            if need >= 0 and need <= len(remaining_points):
                                found = False
                                n_tries = 30
                                for _ in range(n_tries):
                                    if need == 0:
                                        candidate = tuple(sorted(must_include))
                                    else:
                                        extra = random.sample(remaining_points, need)
                                        candidate = tuple(sorted(must_include | set(extra)))
                                    
                                    cand_tris = block_tsub_indices(candidate)
                                    cand_tris_set = set(cand_tris)
                                    
                                    # Check if candidate covers all only_tris
                                    covers_all = all(idx in cand_tris_set for idx in only_tris)
                                    
                                    if covers_all:
                                        # Calculate net gain in coverage
                                        old_tris_set = set(block_tris[i])
                                        new_only = cand_tris_set - old_tris_set
                                        removed_only = old_tris_set - cand_tris_set
                                        
                                        # New blocks that become uncovered if we remove old
                                        loss = sum(1 for idx in removed_only if cov[idx] == 1)
                                        # Gain from new blocks that were uncovered
                                        gain = sum(1 for idx in new_only if cov[idx] == 0)
                                        
                                        delta = gain - loss
                                        
                                        if delta >= 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                                            # Apply swap
                                            for idx in block_tris[i]:
                                                cov[idx] -= 1
                                            for idx in cand_tris:
                                                cov[idx] += 1
                                            blocks[i] = list(candidate)
                                            block_tris[i] = cand_tris
                                            found = True
                                            break
                                
                                if not found:
                                    pass
                
            elif move_type < 0.7:
                # Try replacing a block with one that covers more unique triples
                i = random.randint(0, n - 1)
                only_tris = [idx for idx in block_tris[i] if cov[idx] == 1]
                
                if only_tris:
                    must_include = set()
                    for idx in only_tris:
                        must_include.update(all_tsubs[idx])
                    
                    if len(must_include) <= k:
                        remaining_points = [p for p in range(v) if p not in must_include]
                        need = k - len(must_include)
                        
                        if need >= 0 and need <= len(remaining_points):
                            for _ in range(15):
                                if need == 0:
                                    candidate = tuple(sorted(must_include))
                                else:
                                    extra = random.sample(remaining_points, need)
                                    candidate = tuple(sorted(must_include | set(extra)))
                                
                                cand_tris = block_tsub_indices(candidate)
                                cand_tris_set = set(cand_tris)
                                covers_all = all(idx in cand_tris_set for idx in only_tris)
                                
                                if covers_all:
                                    old_tris_set = set(block_tris[i])
                                    new_only = cand_tris_set - old_tris_set
                                    removed_only = old_tris_set - cand_tris_set
                                    
                                    loss = sum(1 for idx in removed_only if cov[idx] == 1)
                                    gain = sum(1 for idx in new_only if cov[idx] == 0)
                                    
                                    if gain >= loss:
                                        for idx in block_tris[i]:
                                            cov[idx] -= 1
                                        for idx in cand_tris:
                                            cov[idx] += 1
                                        blocks[i] = list(candidate)
                                        block_tris[i] = cand_tris
                                        break
            
            else:
                # Try add-then-remove: add a block, then remove a redundant one
                if n < 150:
                    # Find triples with low coverage
                    low_cov_indices = np.where(cov <= 1)[0]
                    if len(low_cov_indices) > 0:
                        target_idx = random.choice(low_cov_indices)
                        target_tri = all_tsubs[target_idx]
                        target_points = set(target_tri)
                        other_points = [p for p in range(v) if p not in target_points]
                        
                        best_block = None
                        best_gain = -1
                        for _ in range(30):
                            combo = random.sample(other_points, k - t)
                            block = tuple(sorted(target_points | set(combo)))
                            tris = block_tsub_indices(block)
                            gain = sum(1 for idx in tris if cov[idx] <= 1)
                            if gain > best_gain:
                                best_gain = gain
                                best_block = block
                                best_tris = tris
                        
                        if best_block is not None and best_gain > 0:
                            for idx in best_tris:
                                cov[idx] += 1
                            blocks.append(list(best_block))
                            block_tris.append(best_tris)
                            n += 1
                            
                            # Try to remove a redundant block
                            removed = False
                            for i in range(n - 1, -1, -1):
                                removable = True
                                for idx in block_tris[i]:
                                    if cov[idx] == 1:
                                        removable = False
                                        break
                                if removable:
                                    for idx in block_tris[i]:
                                        cov[idx] -= 1
                                    blocks.pop(i)
                                    block_tris.pop(i)
                                    n -= 1
                                    removed = True
                                    break
                            
                            if not removed:
                                # Revert the addition
                                for idx in best_tris:
                                    cov[idx] -= 1
                                blocks.pop()
                                block_tris.pop()
                                n -= 1
            
            T *= cooling
        
        return best_blocks
    
    def local_search(blocks, max_time=30):
        """Attempt to reduce block count via removals and SA."""
        if time.time() > start_time + max_time:
            return blocks
        
        blocks = list(blocks)
        
        # First pass: remove redundant blocks
        blocks = remove_redundant(blocks)
        
        # Second pass: simulated annealing
        if time.time() < start_time + max_time - 5:
            blocks = simulated_annealing(blocks, max_time=max_time - 5)
        
        # Final pass: remove redundant again
        blocks = remove_redundant(blocks)
        
        return blocks
    
    iterations = 0
    while time.time() < deadline:
        blocks = greedy_random_pivot()
        blocks = remove_redundant(blocks)
        
        # Apply local search if we have time
        if time.time() < deadline - 15:
            sa_time = min(25, deadline - time.time() - 5)
            if sa_time > 5:
                blocks = local_search(blocks, max_time=sa_time)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        iterations += 1
        
        # If we've done enough iterations, spend remaining time on SA
        if iterations >= 4 and time.time() < deadline - 10:
            if best_blocks is not None:
                remaining = deadline - time.time() - 5
                if remaining > 10:
                    best_blocks = local_search(best_blocks, max_time=remaining)
                    if len(best_blocks) < best_len:
                        best_len = len(best_blocks)
                break
    
    if best_blocks is None:
        uncovered = set(all_tsubs_set)
        blocks = []
        while uncovered:
            pivot_triple = min(uncovered)
            pivot_points = set(pivot_triple)
            other_points = [p for p in range(v) if p not in pivot_points]
            best_block = None
            best_gain = -1
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_points | set(combo)))
                gain = sum(1 for tri in itertools.combinations(block, t) if tri in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            blocks.append(list(best_block))
            for tri in itertools.combinations(best_block, t):
                if tri in uncovered:
                    uncovered.remove(tri)
        best_blocks = remove_redundant(blocks)

    return best_blocks
