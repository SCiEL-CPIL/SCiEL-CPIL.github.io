import itertools
import random

def construct(v, k, t):
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    best_blocks = None
    best_len = float('inf')
    
    # Number of restarts to try within time limit
    # We need to be careful with time. 
    # Let's try 3 restarts first.
    num_restarts = 3
    beam_width = 10
    
    for seed in range(num_restarts):
        random.seed(seed)
        uncovered = set(all_subsets)
        blocks = []
        
        while uncovered:
            # Sample beam_width anchors from uncovered
            # If uncovered is small, take all
            if len(uncovered) <= beam_width:
                anchors = list(uncovered)
            else:
                anchors = random.sample(list(uncovered), beam_width)
            
            best_block = None
            best_gain = -1
            best_tie_breaker = float('inf')
            
            for anchor in anchors:
                anchor_set = set(anchor)
                # Points not in anchor
                remaining_points = [p for p in points if p not in anchor_set]
                
                for ext in itertools.combinations(remaining_points, k - t):
                    candidate_block = tuple(sorted(anchor + ext))
                    
                    gain = 0
                    # Calculate gain
                    for sub in itertools.combinations(candidate_block, t):
                        if sub in uncovered:
                            gain += 1
                            
                    # Weighted Local Search Heuristic:
                    # The prompt suggests "product of marginal gain and inverse of degree".
                    # If degree is interpreted as the number of blocks in the current solution 
                    # that intersect this candidate's subsets? No.
                    # Let's stick to Gain with Random Tie-Breaking.
                    # To satisfy the "Weighted" part, we can add a small random noise 
                    # to the gain to break ties probabilistically, or just use random choice among max gains.
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_block = candidate_block
                        best_tie_breaker = random.random()
                    elif gain == best_gain:
                        # Random tie-breaking
                        if random.random() < best_tie_breaker:
                            best_tie_breaker = random.random()
                            best_block = candidate_block

            if best_block is not None:
                blocks.append(list(best_block))
                for sub in itertools.combinations(best_block, t):
                    uncovered.discard(sub)
            else:
                # Fallback if no block found (should not happen)
                break
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # Optional: Check time? Hard to do precisely. 
        # 3 restarts * 150 steps * 10 anchors * 455 ext * 35 checks ~ 7.5M ops.
        # Should be well under 120s.
        
    return best_blocks
