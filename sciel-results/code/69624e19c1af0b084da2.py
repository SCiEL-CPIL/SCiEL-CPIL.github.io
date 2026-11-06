import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    all_blocks = list(combinations(points, 6))
    n_blocks = len(all_blocks)
    
    # Map 5-subsets to indices for fast lookup
    all_5subs = list(combinations(points, 5))
    n_5subs = len(all_5subs)
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    # For each block, store the list of 5-subset indices it covers
    block_sub_indices = []
    for b in all_blocks:
        indices = [sub_to_idx[s] for s in combinations(b, 5)]
        block_sub_indices.append(indices)
    
    # Precompute for each 5-subset, the list of blocks that contain it
    # This is useful for backtracking
    sub_to_blocks = [[] for _ in range(n_5subs)]
    for i, b in enumerate(all_blocks):
        for idx in block_sub_indices[i]:
            sub_to_blocks[idx].append(i)
    
    best_blocks = []
    best_count = 0
    
    # Backtracking search with symmetry breaking and time limit
    # We use a recursive backtracking approach
    
    def backtrack(covered, current_blocks, depth):
        nonlocal best_blocks, best_count
        
        if time.time() > deadline:
            return
        
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = list(current_blocks)
        
        # Find the most constrained uncovered 5-subset (minimum remaining values)
        # For efficiency, we just pick the first uncovered 5-subset
        # Actually, let's use a heuristic: find uncovered 5-subset with fewest available blocks
        
        # For speed, we'll use a simpler approach: just try blocks in order
        # but with some randomization
        
        # Find uncovered 5-subsets
        uncovered = [i for i in range(n_5subs) if i not in covered]
        
        if not uncovered:
            return
        
        # Pick the uncovered 5-subset with the fewest available blocks
        # For speed, we'll just pick the first one
        target_5sub = uncovered[0]
        
        # Get blocks that contain this 5-subset
        candidate_blocks = sub_to_blocks[target_5sub]
        
        # Shuffle for randomness
        random.shuffle(candidate_blocks)
        
        for b_idx in candidate_blocks:
            if time.time() > deadline:
                return
            
            # Check if this block conflicts with covered
            b_subs = block_sub_indices[b_idx]
            conflict = False
            for s in b_subs:
                if s in covered:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Add this block
            new_covered = covered | set(b_subs)
            current_blocks.append(all_blocks[b_idx])
            
            backtrack(new_covered, current_blocks, depth + 1)
            
            current_blocks.pop()
    
    # Start with empty covered set
    covered = set()
    current_blocks = []
    
    # Use symmetry breaking: fix the first block to contain point 0
    # Actually, for the Witt design, we can use a more structured approach
    # Let's try multiple random restarts with different orderings
    
    # First, let's try a greedy approach with multiple random orderings
    indices = list(range(n_blocks))
    
    # Try several random greedy attempts
    attempts = 0
    while time.time() < deadline - 5.0:
        random.shuffle(indices)
        covered = set()
        current_blocks = []
        
        for i in indices:
            b = all_blocks[i]
            subs = set(block_sub_indices[i])
            
            if not (covered & subs):
                current_blocks.append(b)
                covered |= subs
        
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = list(current_blocks)
        
        attempts += 1
        
        # If we've found 132, we're done
        if best_count >= 132:
            break
    
    # If we haven't found 132, try backtracking from a good starting point
    if best_count < 132 and time.time() < deadline - 10.0:
        # Use the best greedy result as a starting point
        if best_blocks:
            covered = set()
            for b in best_blocks:
                covered |= set(block_sub_indices[all_blocks.index(b)])
            
            # Try to extend with backtracking
            def extend_backtrack(covered, current_blocks):
                nonlocal best_blocks, best_count
                
                if time.time() > deadline:
                    return
                
                if len(current_blocks) > best_count:
                    best_count = len(current_blocks)
                    best_blocks = list(current_blocks)
                
                if best_count >= 132:
                    return
                
                # Find uncovered 5-subsets
                uncovered = [i for i in range(n_5subs) if i not in covered]
                
                if not uncovered:
                    return
                
                # Pick the uncovered 5-subset with fewest available blocks
                min_count = float('inf')
                target_5sub = -1
                for u in uncovered[:50]:  # Limit for speed
                    cnt = len(sub_to_blocks[u])
                    if cnt < min_count:
                        min_count = cnt
                        target_5sub = u
                
                if target_5sub == -1:
                    target_5sub = uncovered[0]
                
                candidate_blocks = sub_to_blocks[target_5sub]
                random.shuffle(candidate_blocks)
                
                for b_idx in candidate_blocks:
                    if time.time() > deadline:
                        return
                    
                    b_subs = set(block_sub_indices[b_idx])
                    if b_subs & covered:
                        continue
                    
                    new_covered = covered | b_subs
                    current_blocks.append(all_blocks[b_idx])
                    
                    extend_backtrack(new_covered, current_blocks)
                    
                    current_blocks.pop()
            
            extend_backtrack(covered, list(best_blocks))
    
    return [list(b) for b in best_blocks]
