import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for faster lookup? 
    # Or just use a set of tuples.
    # With v=19, t=5, there are 11628 subsets. A set of tuples is fine.
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy attempts
    # How many? Let's try to fit as many as possible.
    # One run takes ~0.1-0.5s? Let's test with a few and see.
    # We can just loop until deadline.
    
    while time.time() < deadline:
        uncovered = set(all_t_subs)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            # To make it faster, we can just pick min or random. 
            # Random helps diversity.
            # Converting set to list every time is expensive.
            # We can maintain a list of uncovered and shuffle? 
            # Or just pick a random element from the set.
            # random.choice(uncovered) is O(1) in CPython? 
            # Actually, random.choice on a set converts to list in some versions or uses internal structure.
            # It's better to keep a list of uncovered and manage it.
            
            # However, managing a list of uncovered while removing items is tricky.
            # Let's stick to set for simplicity, but pick min for determinism in greedy part?
            # Randomization is key.
            # Let's use a list of uncovered subsets to pick from.
            
            # Actually, let's re-structure.
            # We can just pick min(uncovered) for the base greedy, 
            # and randomize the tie-breaking in gain calculation.
            
            start_sub = min(uncovered)
            block = list(start_sub)
            
            while len(block) < k:
                # Calculate gain for each candidate
                gains = {}
                # Candidates are points not in block
                for x in range(v):
                    if x in block:
                        continue
                    # Count how many uncovered t-subsets are covered by adding x
                    # A t-subset is covered if it contains x and t-1 points from block
                    gain = 0
                    # Generate all (t-1)-subsets of block
                    for rest in itertools.combinations(block, t - 1):
                        # Check if (rest + x) is in uncovered
                        # rest is a tuple of length t-1
                        # We need to check if the set of rest + {x} is in uncovered
                        # Since uncovered contains tuples of length t, we need to form a sorted tuple
                        # But rest is already sorted (from combinations).
                        # We just need to insert x into rest in sorted order.
                        # Or simpler: create a tuple of rest + (x,) and sort it? 
                        # No, combinations yields sorted tuples.
                        # If we add x, we need to check if the set is in uncovered.
                        # We can create a set and check? No, too slow.
                        # We can create a tuple.
                        # Let's optimize: 
                        # rest is (a1, a2, ..., a_{t-1})
                        # We need to check if (a1, ..., a_{t-1}, x) sorted is in uncovered.
                        # We can just do:
                        # candidate = tuple(sorted(rest + (x,)))
                        # if candidate in uncovered: gain += 1
                        
                        # This is O(t log t) per check.
                        # t=5, so it's fast.
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    gains[x] = gain
                
                # Find max gain
                max_gain = max(gains.values())
                # Get all points with max gain
                candidates = [x for x, g in gains.items() if g == max_gain]
                
                # Randomly pick one
                if candidates:
                    block.append(random.choice(candidates))
                else:
                    # Should not happen if uncovered is not empty and v > k
                    # But if no gain, we pick a random one
                    available = [x for x in range(v) if x not in block]
                    if available:
                        block.append(random.choice(available))
                    else:
                        break # Should not happen

            block.sort()
            blocks.append(block)
            
            # Update uncovered
            # Remove all t-subsets of block
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        # Redundancy Removal
        # We have a list of blocks. Check if any can be removed.
        # We need to know for each t-subset, which blocks cover it.
        # Or simpler: for each block, check if all its t-subsets are covered by other blocks.
        
        # Build a map: t-subset -> count of blocks covering it
        # This is expensive to rebuild from scratch each time.
        # But v=19, k=8, blocks ~ 300.
        # 300 blocks * 56 subsets = 16800 entries.
        # We can just iterate.
        
        # Let's do a simple pass:
        # For each block i, check if removing it leaves all its t-subsets covered by others.
        # This is O(B * K * T_subs_in_block) if we do it naively.
        # B ~ 300, K ~ 300 (other blocks), T_subs ~ 56.
        # 300 * 300 * 56 = 5 million ops. Might be slow in Python.
        
        # Better:
        # Maintain a counter for each t-subset.
        # Counter[t_sub] = number of blocks covering it.
        
        # Rebuild counter for current blocks
        from collections import defaultdict
        counter = defaultdict(int)
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counter[sub] += 1
        
        # Now check for redundancy
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                # Check if all t-subsets of b have count > 1
                # If yes, b is redundant
                redundant = True
                for sub in itertools.combinations(b, t):
                    if counter[sub] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove b
                    changed = True
                    # Decrement counter for all sub in b
                    for sub in itertools.combinations(b, t):
                        counter[sub] -= 1
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
        # Check time
        if time.time() > deadline:
            break
            
    if best_blocks is None:
        # Fallback to a simple greedy if no time for randomized
        # Or return an empty list? No, must be valid.
        # The first run should have produced something.
        return []
        
    return best_blocks
