"""Covering design C(19,7,4) via randomized greedy with time-budgeted restarts."""
import time
import itertools
import random
from collections import defaultdict


def _greedy_cover(v, k, t, rng, max_blocks=None):
    """
    Randomized greedy set cover.
    Returns (blocks, covered_count) where covered_count is number of t-subsets covered.
    """
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # uncovered: set of indices of uncovered t-subsets
    uncovered = set(range(n_tsubs))
    blocks = []
    
    # For efficiency, precompute for each point x and each (t-1)-subset, 
    # which t-subsets contain x and that (t-1)-subset
    # Actually, let's use a different approach: for each candidate point x,
    # and current block B, the gain is |{S in uncovered : S \ B contains x and |S \ B| == 1}|
    # Wait, gain of adding x to block B: number of uncovered t-subsets S such that 
    # S ⊆ B ∪ {x} and x ∉ B, i.e., S \ {x} ⊆ B and |S \ {x}| = t-1.
    
    # Let's precompute: for each (t-1)-subset C and each point x not in C,
    # the t-subset is C ∪ {x}. We need to track which are uncovered.
    
    # Mapping from t-subset to its index
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    
    # For each (t-1)-subset, list of (point, tsub_idx) for each point not in it
    # Actually, for a (t-1)-subset C, the t-subsets containing C are C ∪ {x} for x not in C.
    
    while uncovered:
        if max_blocks is not None and len(blocks) >= max_blocks:
            break
        
        # Pick a random uncovered t-subset as the target
        target_idx = rng.choice(list(uncovered))
        target = all_tsubs[target_idx]
        block = list(target)
        rng.shuffle(block)
        
        while len(block) < k:
            # For each candidate x not in block, compute gain
            best_x = None
            best_gain = -1
            # Candidates: points not in block
            candidates = [x for x in range(v) if x not in block]
            rng.shuffle(candidates)
            
            for x in candidates:
                # Gain: number of uncovered t-subsets S such that S ⊆ block ∪ {x} and x ∈ S
                # S must contain x and S \ {x} ⊆ block, with |S \ {x}| = t-1
                # So S \ {x} is a (t-1)-subset of block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    s = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(s)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for s in itertools.combinations(block, t):
            idx = tsub_to_idx[s]
            uncovered.discard(idx)
    
    return blocks, n_tsubs - len(uncovered)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 110s to leave margin before 120s kill
    
    # For small cases, deterministic greedy might be fine, but let's use randomized
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized restarts
    rng = random.Random(42)
    
    # First, try a deterministic-ish greedy to get an initial bound
    # Then do randomized restarts
    
    # Estimate how many restarts we can afford
    # Each greedy run for v=19, k=7, t=4: number of blocks ~150, each block requires
    # up to 6 point additions, each with up to 19 candidates, each with C(6,3)=20 checks
    # So ~150 * 6 * 19 * 20 = 342000 operations per run. Should be fast.
    
    n_restarts = 200  # We'll check time and stop early
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:  # 5s safety margin
            break
        
        blocks, covered = _greedy_cover(v, k, t, rng, max_blocks=None)
        
        if covered == len(list(itertools.combinations(range(v), t))):
            # Complete covering
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]
        
        # If we've found a good solution, we can reduce restarts
        # But keep going to try to improve
    
    if best_blocks is not None:
        return best_blocks
    
    # Fallback: run one more greedy without time limit (shouldn't happen)
    blocks, covered = _greedy_cover(v, k, t, rng, max_blocks=None)
    return blocks
