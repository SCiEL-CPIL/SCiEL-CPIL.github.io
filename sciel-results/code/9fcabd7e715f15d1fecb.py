import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(all_t_subsets)
    
    # Track which t-subsets are covered by which blocks
    # covered_count[i] = number of blocks covering t-subset i
    # subset_to_blocks[i] = set of block indices covering t-subset i
    
    def build_initial_greedy():
        """Construct initial covering using greedy with good block selection."""
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 5:
                break
            
            # Pick a random uncovered t-subset as seed (diversity)
            seed = random.choice(list(uncovered))
            block = set(seed)
            
            # Grow block to size k by adding point with max marginal coverage
            while len(block) < k:
                best_x = None
                best_gain = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets this x would cover
                    gain = 0
                    # For each (t-1)-subset of current block, check if adding x covers an uncovered t-subset
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x is not None:
                        # Tie-break: prefer smaller x for determinism
                        if x < best_x:
                            best_x = x
                
                if best_x is None:
                    break
                block.add(best_x)
            
            if len(block) == k:
                block = tuple(sorted(block))
                blocks.append(block)
                # Update uncovered
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
        
        return blocks
    
    def count_uncovered(blocks):
        """Return set of uncovered t-subsets."""
        covered = set()
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered.add(ts)
        return set(all_t_subsets) - covered
    
    def try_remove_block(blocks, idx):
        """Try removing block idx; return True if covering still valid."""
        remaining = blocks[:idx] + blocks[idx+1:]
        if not remaining:
            return False
        uncovered = count_uncovered(remaining)
        return len(uncovered) == 0
    
    def local_search(blocks, max_iter=200):
        """Try to remove redundant blocks via local search."""
        blocks = list(blocks)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter and time.time() < deadline - 10:
            improved = False
            iterations += 1
            
            # Try removing each block
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    return blocks
                
                # Check if block i is redundant
                remaining = blocks[:i] + blocks[i+1:]
                if not remaining:
                    continue
                
                # Check if all t-subsets still covered
                covered = set()
                for block in remaining:
                    for ts in itertools.combinations(block, t):
                        covered.add(ts)
                
                if len(covered) == total_t:
                    blocks = remaining
                    improved = True
                    break  # Restart after removal
        
        return blocks
    
    def pairwise_swap(blocks, max_iter=100):
        """Try replacing pairs of blocks with a single better block."""
        blocks = list(blocks)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter and time.time() < deadline - 10:
            improved = False
            iterations += 1
            
            for i in range(len(blocks)):
                if time.time() > deadline - 10:
                    return blocks
                
                for j in range(i+1, len(blocks)):
                    # Try removing both i and j, then see if we can add one block to cover what they covered
                    remaining = blocks[:i] + blocks[i+1:j] + blocks[j+1:]
                    
                    # Find what's uncovered without blocks i and j
                    covered = set()
                    for block in remaining:
                        for ts in itertools.combinations(block, t):
                            covered.add(ts)
                    
                    uncovered = set(all_t_subsets) - covered
                    if not uncovered:
                        # Both are redundant
                        blocks = remaining
                        improved = True
                        break
                    
                    # Try to find a single block that covers all uncovered
                    # This is hard; skip for now as it's expensive
                    # Instead, just check if removing one works
                    break
                
                if improved:
                    break
        
        return blocks
    
    # Phase 1: Initial greedy construction
    blocks = build_initial_greedy()
    
    if time.time() > deadline - 5:
        return [list(b) for b in blocks]
    
    # Phase 2: Local search to remove redundant blocks
    blocks = local_search(blocks, max_iter=300)
    
    if time.time() > deadline - 5:
        return [list(b) for b in blocks]
    
    # Phase 3: More aggressive local search with multiple passes
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        blocks = local_search(blocks, max_iter=200)
    
    # Final validation
    if time.time() < deadline - 2:
        uncovered = count_uncovered(blocks)
        if uncovered:
            # Emergency: add blocks to cover remaining
            while uncovered and time.time() < deadline - 1:
                seed = random.choice(list(uncovered))
                block = set(seed)
                while len(block) < k:
                    best_x = None
                    best_gain = -1
                    for x in range(v):
                        if x in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            cand = tuple(sorted(rest + (x,)))
                            if cand in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None:
                        break
                    block.add(best_x)
                if len(block) == k:
                    block = tuple(sorted(block))
                    blocks.append(block)
                    for ts in itertools.combinations(block, t):
                        uncovered.discard(ts)
    
    return [list(b) for b in blocks]
