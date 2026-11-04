"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    # For speed, maintain a count of how many uncovered t-subsets each point can help cover
    # Actually, let's use a smarter greedy: for each candidate point x, count how many
    # uncovered t-subsets contain x and are compatible with the current partial block.
    
    while uncovered and time.time() < deadline:
        # Start block with the smallest uncovered t-subset
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            
            # For each candidate point not in block, compute how many uncovered t-subsets
            # it would help cover given the current block
            for x in range(v):
                if x in block:
                    continue
                
                # Count uncovered t-subsets that contain x and all (t-1)-subsets of block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    # Check if (rest + x) is an uncovered t-subset
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets
        for comb in itertools.combinations(block, t):
            uncovered.discard(comb)
    
    # If we ran out of time, we may have an incomplete covering.
    # The evaluator will check validity, so we must return a valid covering.
    # If we're incomplete, we need to finish it quickly or accept invalid result.
    # Since the task requires validity, let's try to finish if close to deadline.
    
    if uncovered and time.time() < deadline - 5:
        # Try to finish with a fast greedy
        while uncovered:
            first_uncovered = min(uncovered)
            block = list(first_uncovered)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for comb in itertools.combinations(block, t):
                uncovered.discard(comb)
    
    return blocks
# EVOLVE-BLOCK-END
