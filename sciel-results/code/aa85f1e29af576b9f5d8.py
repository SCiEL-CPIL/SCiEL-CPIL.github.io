"""Improved covering design C(19,7,4) using randomized greedy with time budget."""
import itertools
import random
import time
import sys

# Precompute all 4-subsets of {0,...,18}
def _precompute():
    v = 19
    t = 4
    all_tsubsets = list(itertools.combinations(range(v), t))
    return all_tsubsets

_ALL_TSUBSETS = _precompute()
_NUM_TSUBSETS = len(_ALL_TSUBSETS)
# Map from t-subset to index for fast lookup
_TSUBSET_TO_IDX = {ts: i for i, ts in enumerate(_ALL_TSUBSETS)}


def _count_new_coverage(block, covered_mask):
    """Count how many t-subsets of block are not yet covered."""
    count = 0
    for ts in itertools.combinations(block, 4):
        idx = _TSUBSET_TO_IDX[ts]
        if not covered_mask[idx]:
            count += 1
    return count


def _greedy_block(uncovered_set, k, v, rng, temperature=1.0):
    """Build a k-element block greedily, optionally with randomization."""
    block = []
    # Start with the lexicographically smallest uncovered t-subset
    first = min(uncovered_set)
    block = list(first)
    
    while len(block) < k:
        # Compute gain for each candidate point
        gains = {}
        for x in range(v):
            if x in block:
                continue
            # Count how many uncovered t-subsets would be covered by adding x
            # A t-subset is covered by block if all its elements are in block
            # When we add x to block, a t-subset {a,b,c,d} is newly covered if:
            #   - x is in {a,b,c,d}
            #   - all other elements of {a,b,c,d} are in block
            #   - it was not already covered
            new_count = 0
            # Iterate over all (t-1)-subsets of current block
            for rest in itertools.combinations(block, 3):
                ts = tuple(sorted(rest + (x,)))
                if ts in uncovered_set:
                    new_count += 1
            gains[x] = new_count
        
        if not gains:
            break
        
        max_gain = max(gains.values())
        if max_gain == 0:
            # Pick a random point
            candidates = [x for x in range(v) if x not in block]
            x = rng.choice(candidates)
        else:
            # Weighted random selection: probability proportional to gain^temperature
            # Or use a softmax-like selection
            if temperature <= 0:
                # Deterministic: pick max gain (with random tie-breaking)
                max_candidates = [x for x, g in gains.items() if g == max_gain]
                x = rng.choice(max_candidates)
            else:
                # Randomized: weight = gain^temperature, but also allow some randomness
                # Use a simple approach: with probability 0.7 pick best, else random weighted
                if rng.random() < 0.7:
                    max_candidates = [x for x, g in gains.items() if g == max_gain]
                    x = rng.choice(max_candidates)
                else:
                    # Weighted random from all candidates
                    items = list(gains.items())
                    weights = [g ** temperature for g in (g for _, g in items)]
                    # Ensure at least one positive weight
                    if sum(weights) == 0:
                        x = rng.choice([x for x in range(v) if x not in block])
                    else:
                        x = rng.choices([it[0] for it in items], weights=weights, k=1)[0]
        
        block.append(x)
    
    return sorted(block)


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Try multiple randomized greedy runs
    max_runs = 50
    run = 0
    while run < max_runs:
        if time.time() > deadline - 10.0:
            break
        
        # Track coverage with a boolean array
        covered = [False] * num_tsubsets
        uncovered_count = num_tsubsets
        blocks = []
        
        while uncovered_count > 0:
            if time.time() > deadline - 5.0:
                break
            
            # Find uncovered t-subsets
            uncovered_set = set()
            for i, ts in enumerate(all_tsubsets):
                if not covered[i]:
                    uncovered_set.add(ts)
            
            if not uncovered_set:
                break
            
            # Build a new block
            block = _greedy_block(uncovered_set, k, v, rng, temperature=1.0)
            
            if not block or len(block) != k:
                # Fallback: just take a random k-subset
                block = sorted(rng.sample(range(v), k))
            
            # Check if this block is already in our list
            block_tuple = tuple(block)
            if block_tuple in [tuple(b) for b in blocks]:
                continue
            
            blocks.append(block)
            
            # Update coverage
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                if not covered[idx]:
                    covered[idx] = True
                    uncovered_count -= 1
        
        # Check if we have a complete covering
        if uncovered_count == 0:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]
        
        run += 1
    
    if best_blocks is None:
        # Fallback: use deterministic greedy
        covered = [False] * num_tsubsets
        uncovered_count = num_tsubsets
        blocks = []
        
        while uncovered_count > 0:
            uncovered_set = set()
            for i, ts in enumerate(all_tsubsets):
                if not covered[i]:
                    uncovered_set.add(ts)
            
            if not uncovered_set:
                break
            
            block = _greedy_block(uncovered_set, k, v, random.Random(0), temperature=0.0)
            blocks.append(block)
            
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                if not covered[idx]:
                    covered[idx] = True
                    uncovered_count -= 1
        
        best_blocks = blocks
        best_count = len(blocks)
    
    return best_blocks
