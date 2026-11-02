import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    total_tsubs = len(list(itertools.combinations(range(v), t)))
    all_tsubs = set(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_len = float('inf')
    
    def block_tsubs(block):
        return set(itertools.combinations(block, t))
    
    def coverage_map(blocks):
        """Map each t-subset to the number of blocks covering it."""
        cov = defaultdict(int)
        for b in blocks:
            for tri in block_tsubs(b):
                cov[tri] += 1
        return cov
    
    def is_valid_fast(blocks, cov=None):
        if cov is None:
            cov = coverage_map(blocks)
        for tri in all_tsubs:
            if cov.get(tri, 0) == 0:
                return False
        return True
    
    def remove_redundant(blocks):
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            cov = coverage_map(blocks)
            for i in range(len(blocks)):
                if cov.get(tuple(blocks[i]), 0) is not None:
                    pass
                # Check if removing block i still covers everything
                for tri in block_tsubs(blocks[i]):
                    if cov[tri] == 1:
                        break
                else:
                    # All triples in block i are covered by other blocks
                    blocks.pop(i)
                    changed = True
                    break
        return blocks
    
    def greedy_random_pivot():
        uncovered = set(all_tsubs)
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
        
        # Precompute t-subsets for each block
        block_tris = [block_tsubs(b) for b in blocks]
        
        # Build coverage count
        cov = defaultdict(int)
        for tris in block_tris:
            for tri in tris:
                cov[tri] += 1
        
        # Check initial validity
        for tri in all_tsubs:
            if cov.get(tri, 0) == 0:
                return blocks  # Not valid, return as-is
        
        T = 50.0  # Initial temperature
        cooling = 0.95
        best_blocks = [list(b) for b in blocks]
        best_n = n
        
        while time.time() < start_time + max_time and T > 0.1:
            if n < best_n:
                best_n = n
                best_blocks = [list(b) for b in blocks]
            
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.3 and n > 1:
                # Try removing a block
                i = random.randint(0, n - 1)
                # Check if block i can be removed
                removable = True
                for tri in block_tris[i]:
                    if cov[tri] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove block i
                    for tri in block_tris[i]:
                        cov[tri] -= 1
                    blocks.pop(i)
                    block_tris.pop(i)
                    n -= 1
                else:
                    # Can't remove, try swapping
                    i = random.randint(0, n - 1)
                    # Find triples only covered by block i
                    only_tris = [tri for tri in block_tris[i] if cov[tri] == 1]
                    
                    if not only_tris:
                        # Block is redundant, remove it
                        for tri in block_tris[i]:
                            cov[tri] -= 1
                        blocks.pop(i)
                        block_tris.pop(i)
                        n -= 1
                    else:
                        # Try to find a replacement block
                        must_include = set()
                        for tri in only_tris:
                            must_include.update(tri)
                        
                        if len(must_include) <= k:
                            remaining_points = [p for p in range(v) if p not in must_include]
                            need = k - len(must_include)
                            
                            if need >= 0 and need <= len(remaining_points):
                                # Try a few random candidates
                                found = False
                                for _ in range(20):
                                    if need == 0:
                                        candidate = tuple(sorted(must_include))
                                    else:
                                        extra = random.sample(remaining_points, need)
                                        candidate = tuple(sorted(must_include | set(extra)))
                                    
                                    cand_tris = block_tsubs(candidate)
                                    
                                    # Check if candidate covers all only_tris
                                    covers_all = all(tri in cand_tris for tri in only_tris)
                                    
                                    if covers_all:
                                        # Calculate gain: new triples covered
                                        new_gain = 0
                                        for tri in cand_tris:
                                            if tri not in block_tris[i]:
                                                # Check if this triple is covered by other blocks
                                                other_cov = cov[tri] - (1 if tri in block_tris[i] else 0)
                                                if other_cov == 0:
                                                    new_gain += 1
                                        
                                        # Accept with probability based on temperature
                                        delta = new_gain - len(only_tris)  # Should be >= 0
                                        if delta >= 0 or random.random() < np.exp(-delta / T):
                                            # Apply swap
                                            for tri in block_tris[i]:
                                                cov[tri] -= 1
                                            for tri in cand_tris:
                                                cov[tri] += 1
                                            blocks[i] = list(candidate)
                                            block_tris[i] = cand_tris
                                            found = True
                                            break
                                
                                if not found:
                                    # Revert (nothing changed)
                                    pass
                
            elif move_type < 0.6 and n > 1:
                # Try swapping two blocks
                i, j = random.sample(range(n), 2)
                # This is complex, so just try a simple version
                # Try replacing block i with a new block that covers its unique triples
                only_tris_i = [tri for tri in block_tris[i] if cov[tri] == 1]
                
                if only_tris_i:
                    must_include = set()
                    for tri in only_tris_i:
                        must_include.update(tri)
                    
                    if len(must_include) <= k:
                        remaining_points = [p for p in range(v) if p not in must_include]
                        need = k - len(must_include)
                        
                        if need >= 0 and need <= len(remaining_points):
                            for _ in range(10):
                                if need == 0:
                                    candidate = tuple(sorted(must_include))
                                else:
                                    extra = random.sample(remaining_points, need)
                                    candidate = tuple(sorted(must_include | set(extra)))
                                
                                cand_tris = block_tsubs(candidate)
                                covers_all = all(tri in cand_tris for tri in only_tris_i)
                                
                                if covers_all:
                                    # Check if this improves coverage
                                    new_gain = 0
                                    for tri in cand_tris:
                                        if tri not in block_tris[i]:
                                            other_cov = cov[tri] - (1 if tri in block_tris[i] else 0)
                                            if other_cov == 0:
                                                new_gain += 1
                                    
                                    if new_gain >= len(only_tris_i):
                                        for tri in block_tris[i]:
                                            cov[tri] -= 1
                                        for tri in cand_tris:
                                            cov[tri] += 1
                                        blocks[i] = list(candidate)
                                        block_tris[i] = cand_tris
                                        break
            
            else:
                # Try adding a block temporarily then removing a different one
                if n < 200:  # Don't let it grow too much
                    # Find most uncovered triple
                    min_cov_tris = [(cov.get(tri, 0), tri) for tri in all_tsubs if cov.get(tri, 0) <= 2]
                    if min_cov_tris:
                        min_cov_tris.sort()
                        target_tri = min_cov_tris[0][1]
                        target_points = set(target_tri)
                        other_points = [p for p in range(v) if p not in target_points]
                        
                        best_block = None
                        best_gain = -1
                        for _ in range(50):
                            combo = random.sample(other_points, k - t)
                            block = tuple(sorted(target_points | set(combo)))
                            gain = sum(1 for tri in itertools.combinations(block, t) if cov.get(tri, 0) <= 1)
                            if gain > best_gain:
                                best_gain = gain
                                best_block = block
                        
                        if best_block is not None and best_gain > 0:
                            # Add this block
                            new_tris = block_tsubs(best_block)
                            for tri in new_tris:
                                cov[tri] += 1
                            blocks.append(list(best_block))
                            block_tris.append(new_tris)
                            n += 1
                            
                            # Now try to remove a redundant block
                            for i in range(n - 1, -1, -1):  # Check from end (most recently added first)
                                removable = True
                                for tri in block_tris[i]:
                                    if cov[tri] == 1:
                                        removable = False
                                        break
                                if removable:
                                    for tri in block_tris[i]:
                                        cov[tri] -= 1
                                    blocks.pop(i)
                                    block_tris.pop(i)
                                    n -= 1
                                    break
            
            T *= cooling
        
        # Return best found
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
            blocks = local_search(blocks, max_time=min(20, deadline - time.time() - 5))
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        iterations += 1
        
        # If we've done enough iterations, spend remaining time on SA
        if iterations >= 5 and time.time() < deadline - 10:
            if best_blocks is not None:
                best_blocks = local_search(best_blocks, max_time=deadline - time.time() - 5)
                if len(best_blocks) < best_len:
                    best_len = len(best_blocks)
                break
    
    if best_blocks is None:
        uncovered = set(all_tsubs)
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
