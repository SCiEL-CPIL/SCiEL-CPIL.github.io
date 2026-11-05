import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142  # 8s safety margin for 150s limit
    
    # Pre-compute all t-subsets (triples)
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # For v=24, k=7, t=3:
    # n_triples = C(24,3) = 2024
    # Each block covers C(7,3) = 35 triples
    
    def greedy_construct():
        """Greedy set cover: repeatedly pick the block covering the most uncovered triples."""
        uncovered = set(range(n_triples))  # Use indices for speed
        blocks = []
        
        while uncovered:
            best_block = None
            best_gain = -1
            
            # Sample blocks to find a good one quickly
            # For C(24,7)=346104, we can't check all each iteration.
            # Instead, we'll use a randomized greedy approach with multiple passes.
            
            # Actually, for the first few blocks, let's be more careful.
            # For later blocks, we can be faster.
            
            num_samples = 200 if len(blocks) < 10 else 50
            
            for _ in range(num_samples):
                block = list(random.sample(range(v), k))
                block.sort()
                
                gain = 0
                for tri in itertools.combinations(block, t):
                    if triple_to_idx[tri] in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None:
                break
            
            blocks.append(best_block)
            
            # Update uncovered
            for tri in itertools.combinations(best_block, t):
                tri_idx = triple_to_idx[tri]
                if tri_idx in uncovered:
                    uncovered.discard(tri_idx)
        
        return blocks

    def count_triples(blocks):
        """Count how many times each triple is covered."""
        counts = [0] * n_triples
        for b in blocks:
            for tri in itertools.combinations(b, t):
                counts[triple_to_idx[tri]] += 1
        return counts

    def remove_redundant(blocks):
        """Remove blocks that don't cover any unique triple."""
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            counts = count_triples(blocks)
            for i in range(len(blocks)):
                if time.time() > deadline - 2:
                    break
                b = blocks[i]
                can_remove = True
                for tri in itertools.combinations(b, t):
                    tri_idx = triple_to_idx[tri]
                    if counts[tri_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    def local_search(blocks):
        """Local search to reduce block count."""
        best_blocks = list(blocks)
        best_count = len(best_blocks)
        
        # Pre-compute triple indices for each block
        block_triples = []
        for b in best_blocks:
            tris = []
            for tri in itertools.combinations(b, t):
                tris.append(triple_to_idx[tri])
            block_triples.append(tris)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            counts = count_triples(best_blocks)
            
            # Recompute block_triples
            block_triples = []
            for b in best_blocks:
                tris = []
                for tri in itertools.combinations(b, t):
                    tris.append(triple_to_idx[tri])
                block_triples.append(tris)
            
            # Try removing each block
            for i in range(len(best_blocks)):
                if time.time() > deadline - 5:
                    break
                
                lost = [tri_idx for tri_idx in block_triples[i] if counts[tri_idx] == 1]
                
                if not lost:
                    # Block i is redundant
                    new_blocks = best_blocks[:i] + best_blocks[i+1:]
                    best_blocks = new_blocks
                    best_count = len(new_blocks)
                    improved = True
                    break
                elif len(lost) <= 5:
                    # Try to find a replacement block that covers all lost triples
                    # The replacement must contain the union of points in lost triples
                    U_points = set()
                    for tri_idx in lost:
                        U_points.update(all_triples[tri_idx])
                    
                    if len(U_points) <= k:
                        remaining_pool = [x for x in range(v) if x not in U_points]
                        num_to_pick = k - len(U_points)
                        
                        # Try a few random completions
                        for _ in range(20):
                            if not remaining_pool:
                                break
                            extra = random.sample(remaining_pool, min(num_to_pick, len(remaining_pool)))
                            candidate = list(U_points) + extra
                            candidate.sort()
                            
                            # Check if this candidate is already in the set
                            cand_tuple = tuple(candidate)
                            if cand_tuple not in [tuple(b) for b in best_blocks]:
                                # Swap
                                best_blocks[i] = candidate
                                improved = True
                                break
                        
                        if improved:
                            break
        
        return best_blocks

    # Main algorithm
    random.seed(42)
    
    # Step 1: Greedy construction
    blocks = greedy_construct()
    
    # Step 2: Remove redundant blocks
    blocks = remove_redundant(blocks)
    
    # Step 3: Local search
    if time.time() < deadline - 10:
        blocks = local_search(blocks)
    
    # Step 4: Final cleanup
    blocks = remove_redundant(blocks)
    
    return blocks
