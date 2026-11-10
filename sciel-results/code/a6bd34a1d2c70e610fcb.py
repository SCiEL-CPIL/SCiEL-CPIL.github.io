import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a randomized greedy algorithm 
    with a post-processing step to remove redundant blocks.
    """
    deadline = time.time() + 110  # Safety margin
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 20 # We can adjust this based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to seed the block? 
            # The seed program picked the min. Randomizing might help escape local minima.
            seed_subset = random.choice(list(uncovered))
            block = set(seed_subset)
            
            # Greedily add points to the block
            while len(block) < k:
                scores = [0] * v
                
                # For each uncovered t-subset S, determine which single point x 
                # added to the current block would cover S.
                # S is covered by block U {x} if S \ {x} is a subset of block.
                # This implies S \ block has size exactly 1, and that element is x.
                for s in uncovered:
                    # Calculate S \ block
                    # s is a tuple, block is a set
                    # We can iterate s and check membership
                    diff = []
                    for x in s:
                        if x not in block:
                            diff.append(x)
                    
                    if len(diff) == 1:
                        scores[diff[0]] += 1
                    # If len(diff) == 0, S is already covered by block (partially), 
                    # but since S is in 'uncovered', this shouldn't happen 
                    # unless 'uncovered' wasn't updated correctly. 
                    # Actually, if S is subset of block, it will be covered by the final block.
                    # But we only add S to 'uncovered' if it's not covered by *any* block yet.
                    # If S is subset of current partial block, it will be covered by the final block.
                    # So we don't need to score any x for it.
                    
                # Find best x not in block
                best_x = -1
                max_score = -1
                candidates = [x for x in all_points if x not in block]
                
                # Randomize tie-breaking
                random.shuffle(candidates)
                
                for x in candidates:
                    if scores[x] > max_score:
                        max_score = scores[x]
                        best_x = x
                
                if best_x == -1:
                    # Should not happen if uncovered is not empty
                    break
                    
                block.add(best_x)
                
            # Convert to sorted list for consistency
            block = sorted(block)
            blocks.append(block)
            
            # Update uncovered
            # Remove all t-subsets of the new block
            for s in itertools.combinations(block, t):
                uncovered.discard(s)
        
        # Post-processing: Remove redundant blocks
        # Count coverage of each t-subset
        coverage_count = {s: 0 for s in all_t_subsets}
        for b in blocks:
            for s in itertools.combinations(b, t):
                coverage_count[s] += 1
        
        # Try to remove blocks
        final_blocks = []
        for b in blocks:
            # Check if removing b leaves any t-subset uncovered
            redundant = True
            for s in itertools.combinations(b, t):
                if coverage_count[s] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove b
                for s in itertools.combinations(b, t):
                    coverage_count[s] -= 1
            else:
                final_blocks.append(b)
        
        if len(final_blocks) < best_count:
            best_count = len(final_blocks)
            best_blocks = final_blocks
            
        # Optional: Early stopping if we hit a very good number?
        # Lower bound is 264. If we get close, we might stop.
        # But we don't know the optimal.
        
    if best_blocks is None:
        # Fallback to simple greedy if something went wrong or time ran out before first completion
        # This should not happen with 110s limit and fast algorithm
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            seed_subset = min(uncovered)
            block = set(seed_subset)
            while len(block) < k:
                scores = [0] * v
                for s in uncovered:
                    diff = [x for x in s if x not in block]
                    if len(diff) == 1:
                        scores[diff[0]] += 1
                best_x = max((x for x in all_points if x not in block), key=lambda x: scores[x])
                block.add(best_x)
            block = sorted(block)
            blocks.append(block)
            for s in itertools.combinations(block, t):
                uncovered.discard(s)
        best_blocks = blocks

    return best_blocks
