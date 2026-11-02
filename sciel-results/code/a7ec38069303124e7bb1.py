"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    """
    Greedy set-cover with local search for covering design C(v,k,t).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to its index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Track coverage: for each t-subset, how many blocks cover it
    coverage = [0] * num_tsubsets
    # For each block, store the list of t-subset indices it covers
    block_tsubsets = []
    
    blocks = []
    
    # Helper: get list of t-subset indices covered by a block
    def block_to_tsubsets(block):
        return [tsubset_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    # Helper: check if a block is redundant (all its t-subsets are covered by other blocks)
    def is_redundant(block_idx):
        bts = block_tsubsets[block_idx]
        for idx in bts:
            if coverage[idx] <= 1:
                return False
        return True
    
    # Helper: remove a block
    def remove_block(block_idx):
        bts = block_tsubsets[block_idx]
        for idx in bts:
            coverage[idx] -= 1
        blocks.pop(block_idx)
        block_tsubsets.pop(block_idx)
    
    # Helper: add a block
    def add_block(block):
        bts = block_to_tsubsets(block)
        for idx in bts:
            coverage[idx] += 1
        blocks.append(block)
        block_tsubsets.append(bts)
    
    # Phase 1: Greedy set cover
    # Start with uncovered t-subsets
    uncovered_count = [1] * num_tsubsets  # 1 means uncovered, 0 means covered
    
    while any(uncovered_count):
        if time.time() > deadline - 5:
            break
        
        # Find the t-subset with minimum "potential" - actually just pick one uncovered
        # To make this more efficient, pick an uncovered t-subset and build a block around it
        uncovered_idx = next(i for i in range(num_tsubsets) if uncovered_count[i] > 0)
        seed_tsubset = all_tsubsets[uncovered_idx]
        
        # Build a block containing this t-subset, greedily adding points that cover most uncovered t-subsets
        block = list(seed_tsubset)
        
        while len(block) < k:
            if time.time() > deadline - 5:
                break
            
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many currently uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsubset_to_idx.get(ts)
                    if idx is not None and uncovered_count[idx] > 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None or best_gain <= 0:
                # No point improves coverage; just pick a random one
                candidates = [x for x in range(v) if x not in block]
                if candidates:
                    best_point = random.choice(candidates)
                else:
                    break
            
            block.append(best_point)
        
        if len(block) < k:
            # Pad with random points
            remaining = [x for x in range(v) if x not in block]
            random.shuffle(remaining)
            block.extend(remaining[:k - len(block)])
        
        block.sort()
        add_block(block)
        
        # Update uncovered_count
        for idx in block_tsubsets[-1]:
            uncovered_count[idx] = 0
    
    # Phase 2: Local search - try to remove redundant blocks
    if time.time() < deadline - 2:
        removed = True
        while removed and time.time() < deadline - 2:
            removed = False
            for i in range(len(blocks) - 1, -1, -1):
                if is_redundant(i):
                    remove_block(i)
                    removed = True
    
    # Phase 3: Local search - try to replace blocks with better ones
    # This is more complex; for now, just do a few random restarts if time permits
    if time.time() < deadline - 5:
        # Try to improve by swapping blocks
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                # Try removing block i and see if we can cover the newly uncovered t-subsets with fewer blocks
                bts = block_tsubsets[i]
                newly_uncovered = [idx for idx in bts if coverage[idx] <= 1]
                
                if not newly_uncovered:
                    # Block is redundant
                    remove_block(i)
                    improved = True
                    break
                else:
                    # Try to find a single block that covers all newly_uncovered
                    # This is expensive, so skip for now
                    pass
    
    return blocks
# EVOLVE-BLOCK-END
