import time
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 145.0
    
    points = range(12)
    all_5subsets = list(combinations(points, 5))
    num_5subsets = len(all_5subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    
    all_6subsets = list(combinations(points, 6))
    num_blocks = len(all_6subsets)
    
    # block_subs[b_idx] = list of 5-subset indices covered by block b_idx
    block_subs = []
    # candidates[s_idx] = list of block indices covering 5-subset s_idx
    candidates = [[] for _ in range(num_5subsets)]
    
    for b_idx, b in enumerate(all_6subsets):
        subs = []
        for s in combinations(b, 5):
            s_idx = subset_to_idx[s]
            subs.append(s_idx)
            candidates[s_idx].append(b_idx)
        block_subs.append(subs)
        
    # Pre-compute conflicts
    # conflicts[b_idx] = set of block indices that conflict with b_idx
    conflicts = [set() for _ in range(num_blocks)]
    for b_idx, subs in enumerate(block_subs):
        for s_idx in subs:
            for other_b in candidates[s_idx]:
                if other_b != b_idx:
                    conflicts[b_idx].add(other_b)
                    
    # Convert conflicts to lists for faster iteration
    conflicts = [list(s) for s in conflicts]
    
    valid = [True] * num_blocks
    best_solution = []
    
    def backtrack(uncovered, depth):
        nonlocal best_solution
        if time.time() - start_time > timeout:
            return False
            
        if not uncovered:
            # We need to reconstruct the solution from the path?
            # The current implementation passed current_blocks.
            # With the undo log, we can maintain a list of chosen blocks.
            # Let's add a `current_blocks` list to the recursion state or maintain it globally.
            # Maintaining it globally is easier with the undo log.
            pass # Placeholder
            
    # We need to track current_blocks.
    # Let's modify the backtrack function to take current_blocks list.
    # And use the global valid array.
    
    def backtrack(uncovered, current_blocks):
        nonlocal best_solution
        if time.time() - start_time > timeout:
            return False
            
        if not uncovered:
            best_solution = current_blocks[:]
            return True
            
        # Heuristic: Min remaining values
        min_subset = -1
        min_count = float('inf')
        
        # Iterate uncovered subsets
        for s_idx in uncovered:
            count = 0
            for b_idx in candidates[s_idx]:
                if valid[b_idx]:
                    count += 1
            if count < min_count:
                min_count = count
                min_subset = s_idx
            if min_count == 0:
                return False
            if min_count == 1:
                break
                
        # Try blocks
        for b_idx in candidates[min_subset]:
            if not valid[b_idx]:
                continue
                
            # Apply choice
            undo_stack = []
            valid[b_idx] = False
            undo_stack.append(b_idx)
            
            for c in conflicts[b_idx]:
                if valid[c]:
                    valid[c] = False
                    undo_stack.append(c)
                    
            new_uncovered = set(uncovered)
            for s_idx in block_subs[b_idx]:
                new_uncovered.discard(s_idx)
                
            if backtrack(new_uncovered, current_blocks + [b_idx]):
                # Undo not needed if we return True immediately?
                # But we should clean up to be safe if we wanted to continue searching.
                # Since we stop at first solution, we can just return.
                # However, if we want to be clean, we should undo.
                for idx in reversed(undo_stack):
                    valid[idx] = True
                return True
                
            # Undo
            for idx in reversed(undo_stack):
                valid[idx] = True
                
        return False
        
    initial_uncovered = set(range(num_5subsets))
    
    if backtrack(initial_uncovered, []):
        return [list(all_6subsets[i]) for i in best_solution]
    else:
        # Fallback
        covered = set()
        blocks = []
        for b in all_6subsets:
            subsets = tuple(combinations(b, 5))
            if not any(s in covered for s in subsets):
                blocks.append(list(b))
                covered.update(subsets)
        return blocks
