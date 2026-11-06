import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin before 120s limit
    
    # Precompute all 6-subsets and their 5-subsets
    all_blocks = list(itertools.combinations(range(12), 6))
    block_to_5sets = {}
    for b in all_blocks:
        # Convert to tuple for hashing
        b_tup = tuple(b)
        block_to_5sets[b_tup] = list(itertools.combinations(b_tup, 5))
    
    # We will use backtracking to find the 132 blocks
    # Symmetry breaking:
    # Block 0: (0, 1, 2, 3, 4, 5)
    # Block 1: (0, 1, 6, 7, 8, 9)
    
    initial_blocks = [tuple((0, 1, 2, 3, 4, 5)), tuple((0, 1, 6, 7, 8, 9))]
    
    covered_5sets = set()
    for b in initial_blocks:
        for s in block_to_5sets[b]:
            covered_5sets.add(s)
            
    blocks_found = list(initial_blocks)
    
    # Candidate blocks: all blocks not yet used
    # We need to check if a block conflicts with covered_5sets
    # A block conflicts if any of its 5-subsets is in covered_5sets
    
    # To speed up, we can maintain a list of valid candidates
    # But checking 924 blocks each step is O(924 * 6) which is fast.
    
    # Order candidates by heuristic: e.g., number of "new" 5-subsets they cover?
    # Or just lexicographic.
    
    # We need to find 132 blocks total.
    # We have 2. Need 130 more.
    
    # Backtracking state
    # To make it efficient, we can use a greedy selection first to get close,
    # then backtrack to fill gaps if needed.
    # But for S(5,6,12), greedy often fails to reach 132.
    # Backtracking with good pruning is better.
    
    # Pruning: If a 5-subset is not covered, it must be covered by some future block.
    # This is hard to check efficiently.
    
    # Alternative: Use the fact that the solution is unique (up to iso).
    # We can use a "fill" strategy:
    # 1. Start with the 2 fixed blocks.
    # 2. Greedily add blocks that cover the "most uncovered" 5-subsets.
    # 3. If stuck, backtrack.
    
    # Let's try a simple greedy first, then if it doesn't reach 132, do limited backtracking.
    # Given the time limit, a pure backtracker might be too slow if it explores wrong paths.
    # A hybrid:
    # 1. Greedy fill.
    # 2. If len(blocks) < 132, try to swap blocks to improve coverage.
    
    # However, the prompt suggests the Golay code method.
    # Let's try to generate the 132 blocks via the Golay code directly if possible.
    # But implementing the Golay code generator from scratch is error-prone.
    # The backtracking with symmetry breaking is safer.
    
    # Let's implement a recursive backtracker with a node limit.
    
    def is_valid_block(b_tup, covered):
        for s in block_to_5sets[b_tup]:
            if s in covered:
                return False
        return True

    def backtrack(current_blocks, covered, depth, max_depth=132):
        nonlocal start_time, deadline
        if time.time() > deadline:
            return False
        
        if len(current_blocks) == max_depth:
            return True
        
        # Heuristic: Sort candidates by number of uncovered 5-subsets they contain
        candidates = []
        for b in all_blocks:
            if b in current_blocks:
                continue
            if is_valid_block(b, covered):
                # Count how many new 5-subsets this block covers
                new_count = 0
                for s in block_to_5sets[b]:
                    if s not in covered:
                        new_count += 1
                candidates.append((new_count, b))
        
        # Sort by new_count descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Limit candidates to top 50 to keep branching factor manageable
        for _, b in candidates[:50]:
            new_covered = covered.copy()
            for s in block_to_5sets[b]:
                new_covered.add(s)
            
            current_blocks.append(b)
            if backtrack(current_blocks, new_covered, depth + 1, max_depth):
                return True
            current_blocks.pop()
        
        return False

    # Start backtracking
    # To avoid deep recursion issues, we can use an iterative approach or limit depth.
    # But 130 depth is fine for Python if we use sys.setrecursionlimit.
    import sys
    sys.setrecursionlimit(2000)
    
    # Try to find the full solution
    # If it fails, return the best greedy solution found so far?
    # The backtracker above returns True/False. We need the blocks.
    # Let's modify it to store the best solution found.
    
    best_solution = None
    
    def backtrack_with_best(current_blocks, covered, depth, max_depth=132):
        nonlocal start_time, deadline, best_solution
        if time.time() > deadline:
            return
        
        if len(current_blocks) > (len(best_solution) if best_solution else 0):
            best_solution = list(current_blocks)
        
        if len(current_blocks) == max_depth:
            return
        
        candidates = []
        for b in all_blocks:
            if b in current_blocks:
                continue
            if is_valid_block(b, covered):
                new_count = 0
                for s in block_to_5sets[b]:
                    if s not in covered:
                        new_count += 1
                candidates.append((new_count, b))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        for _, b in candidates[:20]: # Limit branching
            new_covered = covered.copy()
            for s in block_to_5sets[b]:
                new_covered.add(s)
            
            current_blocks.append(b)
            backtrack_with_best(current_blocks, new_covered, depth + 1, max_depth)
            current_blocks.pop()
            
            if time.time() > deadline:
                return

    backtrack_with_best(initial_blocks, covered_5sets, 2, 132)
    
    if best_solution and len(best_solution) == 132:
        return [list(b) for b in best_solution]
    elif best_solution:
        return [list(b) for b in best_solution]
    else:
        # Fallback to greedy if backtracking found nothing (unlikely)
        covered, blocks = set(), []
        for b in all_blocks:
            b_tup = tuple(b)
            if is_valid_block(b_tup, covered):
                blocks.append(list(b))
                for s in block_to_5sets[b_tup]:
                    covered.add(s)
        return blocks

# EVOLVE-BLOCK-END
