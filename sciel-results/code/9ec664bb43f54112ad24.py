import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Total number of t-subsets
    # We use bitmasks for subsets. Point i is represented by (1 << i).
    
    # Generate all t-subsets as bitmasks
    all_subsets = []
    for comb in _combinations(range(v), t):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_subsets.append(mask)
    
    uncovered = set(all_subsets)
    
    # Index: point -> list of uncovered subset masks containing that point
    # We initialize this. Note: lists are expensive to update (remove item).
    # Alternative: Just iterate all uncovered? 
    # For v=19, t=5, |uncovered| starts at 11628.
    # If we iterate all 11628 for each candidate point (19 candidates), that's ~220k ops per step.
    # Block construction takes k=8 steps. So ~1.7M ops per block.
    # We might need ~300 blocks. 300 * 1.7M = 510M ops. Too slow in Python?
    # Maybe. Let's try to optimize the gain calculation.
    
    # Better Gain Calc:
    # For a partial block B (mask mB), and candidate x (mask mx):
    # We want count of S in uncovered such that S has x and (S & ~mx) is subset of mB.
    # This is equivalent to: S is in uncovered, S & mx != 0, and (S & ~mx) & ~mB == 0.
    
    # Let's stick to the index method but optimize removal.
    # Instead of lists, we can just keep `uncovered` as a set and filter.
    # But filtering 11k items 19 times per step is heavy.
    
    # Let's try a different heuristic: Randomized Greedy with Restart.
    # The deterministic greedy might get stuck in local minima.
    # We will run the greedy, but add some randomness in tie-breaking or initial choice.
    
    blocks = []
    
    def get_gain(block_mask, candidate_mask, uncovered_set):
        # Count uncovered subsets S such that candidate is in S and S\{candidate} is in block
        count = 0
        # Iterate all uncovered sets? 
        # Optimization: If we had an index, it would be faster.
        # Let's use the index approach properly.
        pass

    # Let's implement the Index approach carefully.
    # index[p] is a set of masks.
    # Removing an item from a set is O(1) average.
    
    index = {i: set() for i in range(v)}
    for m in uncovered:
        for i in range(v):
            if m & (1 << i):
                index[i].add(m)
        
    def remove_from_index(mask):
        for i in range(v):
            if mask & (1 << i):
                if mask in index[i]:
                    index[i].discard(mask)
    
    # Greedy Loop
    while uncovered:
        if time.time() > deadline:
            break
            
        # Choose the first uncovered subset to start the block
        # Pick one arbitrarily (e.g., min) or random. Min is deterministic.
        start_mask = min(uncovered)
        
        # Initial block contains the points of start_mask
        block_points = [i for i in range(v) if start_mask & (1 << i)]
        block_mask = start_mask
        
        # We must cover start_mask. 
        # Actually, standard greedy picks a t-set, then extends.
        # The points in the t-set are forced into the block.
        
        while len(block_points) < k:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_candidate = None
            best_candidates = [] # For randomization/tie-breaking
            
            for x in range(v):
                if block_mask & (1 << x):
                    continue
                    
                # Calculate gain
                # S is in uncovered, S contains x, (S & ~(1<<x)) is subset of block_mask
                gain = 0
                # Iterate over uncovered sets containing x using index
                for s_mask in index[x]:
                    # Check if s_mask is still in uncovered (it should be, as we maintain it)
                    # Check if s_mask without x is subset of block_mask
                    # (s_mask & ~(1<<x)) & ~block_mask == 0
                    if (s_mask & ~(1 << x)) & ~block_mask == 0:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = x
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
                    
            if best_candidate is None:
                # Should not happen if v > k and uncovered is not empty
                # Pick a random point
                candidates = [x for x in range(v) if not (block_mask & (1 << x))]
                if not candidates:
                    break
                best_candidate = random.choice(candidates)
            else:
                # Random tie-breaking to escape local optima
                best_candidate = random.choice(best_candidates)
                
            # Add best_candidate to block
            block_mask |= (1 << best_candidate)
            block_points.append(best_candidate)
            
        # Block is complete (size k)
        # Find all t-subsets of this block and mark as covered
        block_set = set(block_points)
        to_remove = []
        for s_mask in list(uncovered):
            # Check if s_mask is a subset of block_set
            # s_mask has t bits. Check if all those bits are in block_mask
            if (s_mask & ~block_mask) == 0:
                to_remove.append(s_mask)
                
        for s_mask in to_remove:
            uncovered.discard(s_mask)
            remove_from_index(s_mask)
            
        blocks.append(sorted(block_points))
        
    return blocks

def _combinations(iterable, r):
    # Standard combinations generator
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i:] = range(indices[i] + 1, indices[i] + 1 + r - i)
        yield tuple(pool[i] for i in indices)
