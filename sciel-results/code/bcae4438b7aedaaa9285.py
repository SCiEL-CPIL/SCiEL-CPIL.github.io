"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import defaultdict


def _all_tsubsets(v, t):
    """Generate all t-subsets as sorted tuples."""
    return list(itertools.combinations(range(v), t))


def _tsubset_to_index(v, t):
    """Map a t-subset (tuple of sorted ints) to a linear index in 0..C(v,t)-1."""
    # Use combinatorial number system for mapping
    # For t=4, v=19, C(19,4)=3876
    # We'll use a dictionary for simplicity since C(19,4) is small
    pass


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each point, precompute which t-subsets contain it
    # point_tsubs[p] = list of indices of t-subsets containing point p
    point_tsubs = defaultdict(list)
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsubs[p].append(i)
    
    # For a block (set of k points), the t-subsets it covers are the union of
    # t-subsets fully contained in the block. We can compute this by checking
    # all t-subsets of the block.
    
    def block_covers_tsub_indices(block_set):
        """Return set of t-subset indices covered by a block (set of points)."""
        covered = set()
        for ts in itertools.combinations(sorted(block_set), t):
            idx = tsub_index[ts]
            covered.add(idx)
        return covered
    
    def greedy_construct(random_seed=0):
        """Greedy covering with randomization."""
        rng = np.random.default_rng(random_seed)
        
        # uncovered: set of t-subset indices not yet covered
        uncovered = set(range(n_tsubs))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            uncovered_list = list(uncovered)
            start_idx = uncovered_list[rng.integers(len(uncovered_list))]
            start_tsub = all_tsubs[start_idx]
            
            # Build block starting with points from start_tsub
            block = set(start_tsub)
            
            # Greedily add points that cover the most uncovered t-subsets
            while len(block) < k:
                best_point = None
                best_gain = -1
                best_tsubs = None
                
                for p in range(v):
                    if p in block:
                        continue
                    
                    # Compute gain: number of uncovered t-subsets that would be
                    # covered if we add point p to the block
                    # A t-subset is covered by block∪{p} if it's a t-subset of
                    # block∪{p} and contains p, OR if it's already a t-subset of block
                    # Actually, we want t-subsets that contain p and are subsets of block∪{p}
                    # i.e., t-subsets of (block ∪ {p}) that contain p
                    
                    # These are t-subsets where p is one element and the other t-1
                    # elements are from block
                    gain = 0
                    new_tsubs = []
                    for rest in itertools.combinations(sorted(block), t - 1):
                        ts = tuple(sorted(rest + (p,)))
                        idx = tsub_index.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1
                            new_tsubs.append(idx)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                        best_tsubs = new_tsubs
                
                if best_point is None or best_gain <= 0:
                    # No point adds coverage; pick a random point
                    candidates = [p for p in range(v) if p not in block]
                    if not candidates:
                        break
                    best_point = candidates[rng.integers(len(candidates))]
                    best_tsubs = []
                
                block.add(best_point)
            
            # Convert block to sorted list
            block_list = sorted(block)
            
            # Compute which t-subsets this block covers
            covered = block_covers_tsub_indices(set(block_list))
            uncovered -= covered
            
            blocks.append(block_list)
            
            # Check deadline
            if time.time() > deadline:
                break
        
        # Remove redundant blocks
        blocks = _remove_redundant(blocks, v, k, t, all_tsubs, tsub_index, deadline)
        
        return blocks
    
    def _remove_redundant(blocks, v, k, t, all_tsubs, tsub_index, deadline):
        """Remove blocks that don't contribute unique coverage."""
        if not blocks:
            return blocks
        
        # For each block, compute which t-subsets it uniquely covers
        # (i.e., t-subsets covered by this block but not by any other block)
        
        # First, compute coverage for each block
        block_covered = []
        for b in blocks:
            covered = block_covers_tsub_indices(set(b))
            block_covered.append(covered)
        
        # For each block, check if removing it leaves all t-subsets covered
        essential = [True] * len(blocks)
        
        for i, b in enumerate(blocks):
            if not essential[i]:
                continue
            if time.time() > deadline:
                break
            
            # Compute union of all other blocks' coverage
            other_covered = set()
            for j, covered_j in enumerate(block_covered):
                if j != i:
                    other_covered |= covered_j
            
            # Check if block i covers any t-subset not in other_covered
            unique = block_covered[i] - other_covered
            if not unique:
                essential[i] = False
        
        return [b for i, b in enumerate(blocks) if essential[i]]
    
    # Run multiple greedy constructions with different random seeds
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 20  # Adjust based on timing
    
    for seed in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        blocks = greedy_construct(random_seed=seed)
        
        if not blocks:
            continue
        
        # Verify coverage
        covered = set()
        for b in blocks:
            for ts in itertools.combinations(sorted(b), t):
                covered.add(ts)
        
        if len(covered) == n_tsubs:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        else:
            # Incomplete covering; try to complete it
            uncovered = set(all_tsubs) - covered
            while uncovered and time.time() < deadline - 2:
                start_tsub = next(iter(uncovered))
                block = list(start_tsub)
                while len(block) < k:
                    best_p = None
                    best_gain = -1
                    for p in range(v):
                        if p in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(sorted(block), t - 1):
                            ts = tuple(sorted(rest + (p,)))
                            if ts in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_p = p
                    if best_p is None:
                        candidates = [p for p in range(v) if p not in block]
                        if candidates:
                            best_p = candidates[0]
                        else:
                            break
                    block.append(best_p)
                block.sort()
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
            
            # Re-verify
            covered = set()
            for b in blocks:
                for ts in itertools.combinations(sorted(b), t):
                    covered.add(ts)
            
            if len(covered) == n_tsubs and len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
    
    # Fallback: if no valid covering found, use a simple greedy
    if best_blocks is None:
        uncovered = set(range(n_tsubs))
        blocks = []
        while uncovered and time.time() < deadline:
            start_idx = min(uncovered)
            start_tsub = all_tsubs[start_idx]
            block = list(start_tsub)
            while len(block) < k:
                best_p = None
                best_gain = -1
                for p in range(v):
                    if p in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(sorted(block), t - 1):
                        ts = tuple(sorted(rest + (p,)))
                        idx = tsub_index.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                if best_p is None:
                    candidates = [p for p in range(v) if p not in block]
                    if candidates:
                        best_p = candidates[0]
                    else:
                        break
                block.append(best_p)
            block.sort()
            blocks.append(block)
            covered = block_covers_tsub_indices(set(block))
            uncovered -= covered
        
        best_blocks = blocks
    
    return best_blocks
# EVOLVE-BLOCK-END
