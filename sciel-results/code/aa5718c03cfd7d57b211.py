"""Combined covering design program: fast greedy with precomputed incidence + local search reduction."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 35s margin before 150s kill
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets_set = set(all_tsubsets)
    num_tsubsets = len(all_tsubsets)
    
    # Precompute for each t-subset, which k-blocks contain it
    # This is expensive but done once. For v=24, k=7, t=3:
    # C(24,3) = 2024 triples, C(24,7) = 346104 blocks
    # Instead, for each triple, the blocks containing it are C(v-t, k-t) = C(21,4) = 5985
    # We'll build a mapping from triple -> list of block indices (represented as frozensets or sorted tuples)
    
    # Actually, let's use a different approach: for each triple, generate all blocks containing it
    # and store them. But we don't want to store 2024 * 5985 entries.
    
    # Better: use the approach from inspiration - for a given pivot triple, enumerate all blocks containing it
    # and compute gain on the fly.
    
    # Let's use a hybrid: precompute block triples for speed in local search
    # but for greedy, enumerate on the fly.
    
    def block_triples(block):
        """Generate all t-subsets of a block."""
        return list(itertools.combinations(block, t))
    
    def greedy_cover(seed):
        """Greedy covering with random pivot selection."""
        rng = random.Random(seed)
        uncovered = set(all_tsubsets_set)
        blocks = []
        
        while uncovered and time.time() < deadline:
            # Pick a random uncovered triple
            pivot_triple = rng.choice(tuple(uncovered))
            pivot_set = set(pivot_triple)
            other_points = [p for p in range(v) if p not in pivot_set]
            
            best_gain = -1
            best_block = None
            
            # Enumerate all k-subsets containing pivot_triple
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_set | set(combo)))
                
                # Calculate gain
                gain = 0
                for tri in itertools.combinations(block, t):
                    if tri in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is not None:
                blocks.append(list(best_block))
                for tri in itertools.combinations(best_block, t):
                    uncovered.discard(tri)
        
        # If we didn't finish in time, we still have a partial covering
        # But we need a valid covering, so let's complete it greedily
        while uncovered:
            pivot_triple = min(uncovered)
            pivot_set = set(pivot_triple)
            other_points = [p for p in range(v) if p not in pivot_set]
            
            best_gain = -1
            best_block = None
            
            for combo in itertools.combinations(other_points, k - t):
                block = tuple(sorted(pivot_set | set(combo)))
                gain = sum(1 for tri in itertools.combinations(block, t) if tri in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is not None:
                blocks.append(list(best_block))
                for tri in itertools.combinations(best_block, t):
                    uncovered.discard(tri)
        
        return blocks
    
    def remove_redundant(blocks, v, k, t):
        """Remove redundant blocks from the covering."""
        if not blocks:
            return blocks
        
        # Precompute triples for each block
        block_triples_list = [set(block_triples(b)) for b in blocks]
        
        # Compute coverage count for each triple
        triple_count = {}
        for bt in block_triples_list:
            for tri in bt:
                triple_count[tri] = triple_count.get(tri, 0) + 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                # A block is redundant if all its triples are covered by other blocks
                redundant = True
                for tri in block_triples_list[i]:
                    if triple_count.get(tri, 0) <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for tri in block_triples_list[i]:
                        triple_count[tri] -= 1
                    blocks.pop(i)
                    block_triples_list.pop(i)
                    changed = True
                    break  # Restart as indices changed
        
        return blocks
    
    def local_search(blocks, v, k, t, max_iterations=50):
        """Local search to improve the covering by removing redundant blocks after swaps."""
        if not blocks:
            return blocks
        
        # Precompute triples for each block
        block_triples_list = [set(block_triples(b)) for b in blocks]
        
        # Compute coverage count for each triple
        triple_count = {}
        for bt in block_triples_list:
            for tri in bt:
                triple_count[tri] = triple_count.get(tri, 0) + 1
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to find a redundant block
            redundant_idx = -1
            for i in range(len(blocks)):
                is_redundant = True
                for tri in block_triples_list[i]:
                    if triple_count.get(tri, 0) <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_idx = i
                    break
            
            if redundant_idx >= 0:
                # Remove the redundant block
                for tri in block_triples_list[redundant_idx]:
                    triple_count[tri] -= 1
                blocks.pop(redundant_idx)
                block_triples_list.pop(redundant_idx)
        
        return blocks
    
    # Try multiple greedy constructions with different seeds
    best_blocks = None
    best_len = float('inf')
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        if time.time() > deadline - 10:  # Leave 10s for local search
            break
        
        blocks = greedy_cover(seed)
        
        # Remove redundant blocks
        blocks = remove_redundant(blocks, v, k, t)
        
        # Local search
        blocks = local_search(blocks, v, k, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    # If no blocks found (shouldn't happen), do a final greedy
    if best_blocks is None:
        best_blocks = greedy_cover(42)
        best_blocks = remove_redundant(best_blocks, v, k, t)
    
    return best_blocks
