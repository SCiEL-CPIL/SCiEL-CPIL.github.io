"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def _all_tsubs(v, t):
    """Precompute all t-subsets as frozensets for fast lookup."""
    return [frozenset(c) for c in itertools.combinations(range(v), t)]

def _block_covers(block, t):
    """Return set of t-subsets covered by a block."""
    return set(frozenset(c) for c in itertools.combinations(block, t))

def _greedy_construct(v, k, t, rng=None, deadline=None):
    """Greedy covering with optional deadline guard."""
    if rng is None:
        rng = random.Random()
    
    all_tsubs = _all_tsubs(v, t)
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if deadline and time.time() > deadline:
            break
        # Pick a random uncovered t-subset to start the block
        seed_sub = rng.choice(list(uncovered))
        block = list(seed_sub)
        
        while len(block) < k:
            if deadline and time.time() > deadline:
                break
            best_gain = -1
            best_point = None
            # Precompute t-1 subsets of current block for faster gain calc
            block_set = set(block)
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in t_minus_1_subsets:
                    candidate = frozenset(rest) | {x}
                    if candidate in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                # Fallback: add any unused point
                for x in range(v):
                    if x not in block_set:
                        best_point = x
                        break
            
            if best_point is not None:
                block.append(best_point)
            else:
                break
        
        if len(block) == k:
            block.sort()
            blocks.append(tuple(block))
            covered = _block_covers(block, t)
            uncovered.difference_update(covered)
    
    return [list(b) for b in blocks]

def _local_search(blocks, v, k, t, max_iters=50, deadline=None):
    """Try to remove or swap blocks to reduce total count."""
    if deadline and time.time() > deadline:
        return blocks
    
    all_tsubs = _all_tsubs(v, t)
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage for each block
    def get_coverage(block):
        return set(frozenset(c) for c in itertools.combinations(block, t))
    
    # Try removing each block and checking if coverage is maintained
    for i in range(len(blocks)):
        if deadline and time.time() > deadline:
            break
        remaining = blocks[:i] + blocks[i+1:]
        # Check if all t-subsets are still covered
        covered = set()
        for b in remaining:
            covered.update(get_coverage(b))
        if len(covered) == len(all_tsubs):
            blocks = remaining
            break
    
    return blocks

def construct(v, k, t):
    """Construct a covering design C(v,k,t) using greedy with restarts and local search."""
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline, 10s safety margin
    
    # Try multiple random restarts
    best_blocks = None
    best_len = float('inf')
    
    # First, run a deterministic greedy to get a baseline
    baseline = _greedy_construct(v, k, t, rng=random.Random(42), deadline=deadline)
    if baseline:
        best_blocks = baseline
        best_len = len(baseline)
    
    if time.time() > deadline:
        return best_blocks
    
    # Try random restarts
    restart_count = 0
    while time.time() < deadline - 30:  # Keep 30s for local search
        restart_count += 1
        rng = random.Random(restart_count * 12345 + 7)
        blocks = _greedy_construct(v, k, t, rng=rng, deadline=deadline)
        
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
    
    # Local search on the best solution found
    if best_blocks and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, v, k, t, max_iters=100, deadline=deadline)
    
    return best_blocks if best_blocks else []
# EVOLVE-BLOCK-END
