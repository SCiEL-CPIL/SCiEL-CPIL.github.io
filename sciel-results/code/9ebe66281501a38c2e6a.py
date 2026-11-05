import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Greedy + Local Search with Multi-Start.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    # Helper to get all k-subsets containing a specific t-subset
    def get_candidates_for_subset(subset):
        # subset is a tuple of t points
        # We need to choose k-t points from the remaining v-t points
        remaining_points = [p for p in range(v) if p not in subset]
        candidates = []
        for extra in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(subset + extra))
            candidates.append(block)
        return candidates

    # Helper to get all t-subsets of a k-block
    def get_subsets_of_block(block):
        return list(itertools.combinations(block, t))

    best_blocks = None
    best_count = float('inf')

    # Multi-start loop
    while time.time() < deadline:
        # Initialize
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Greedy Phase
        while uncovered:
            # Pick a random uncovered subset
            # Using a random choice helps diversify the solution
            # We can optimize this by picking a subset with fewest candidates if we want, 
            # but random is good for multi-start diversity.
            S = random.choice(list(uncovered))
            
            # Get candidates containing S
            candidates = get_candidates_for_subset(S)
            
            # Score candidates: number of uncovered subsets they cover
            best_candidate = None
            max_score = -1
            
            # Optimization: We only need to check subsets in `uncovered`
            # For each candidate, count how many of its t-subsets are in `uncovered`
            for cand in candidates:
                score = 0
                for sub in get_subsets_of_block(cand):
                    if sub in uncovered:
                        score += 1
                        # Early termination? No, we need the exact count for max.
                        # But if score + (remaining subsets) < max_score, we can break?
                        # Max possible score is 128.
                if score > max_score:
                    max_score = score
                    best_candidate = cand
                    # If we found a candidate that covers all uncovered subsets? Unlikely.
                    # If max_score == len(uncovered) and len(uncovered) is small?
            
            if best_candidate is not None:
                blocks.append(best_candidate)
                # Update uncovered
                for sub in get_subsets_of_block(best_candidate):
                    uncovered.discard(sub)
            else:
                # Should not happen if v, k, t are valid
                break
        
        # Local Search: Deletion of redundant blocks
        # Count coverage
        coverage_count = defaultdict(int)
        for block in blocks:
            for sub in get_subsets_of_block(block):
                coverage_count[sub] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for block in blocks:
                is_redundant = True
                for sub in get_subsets_of_block(block):
                    if coverage_count[sub] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block
                    for sub in get_subsets_of_block(block):
                        coverage_count[sub] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks
        
        # Check if this solution is better
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # If we reached the lower bound (264), we can stop? 
            # The lower bound is 264. If we find 264, it's optimal.
            if best_count <= 264:
                break

    # Ensure we return a valid list of lists
    if best_blocks is None:
        # Fallback: return an empty list? No, must be valid.
        # If the loop didn't run or failed, we need a valid covering.
        # The greedy phase always produces a valid covering.
        # So best_blocks should not be None if at least one iteration ran.
        # If deadline was immediate, we might not have run.
        # Let's force one run if best_blocks is None.
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            S = random.choice(list(uncovered))
            candidates = get_candidates_for_subset(S)
            best_candidate = None
            max_score = -1
            for cand in candidates:
                score = sum(1 for sub in get_subsets_of_block(cand) if sub in uncovered)
                if score > max_score:
                    max_score = score
                    best_candidate = cand
            blocks.append(best_candidate)
            for sub in get_subsets_of_block(best_candidate):
                uncovered.discard(sub)
        best_blocks = blocks

    return [list(b) for b in best_blocks]
