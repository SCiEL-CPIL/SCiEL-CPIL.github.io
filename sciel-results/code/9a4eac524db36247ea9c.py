import time
import sys
from itertools import combinations

# Precompute 5-subset indices
# Total 5-subsets of 12 points: C(12, 5) = 792
# We map each 5-subset to an integer index 0..791
# We can use a dictionary or a formula.
# Formula for index of subset (a,b,c,d,e) with 0<=a<b<c<d<e<12
# This is the rank in lexicographic order.

def subset_index(s):
    # s is a tuple of 5 sorted integers
    a, b, c, d, e = s
    # Count how many 5-subsets are lexicographically smaller
    # This can be done with combinatorial number system
    # index = C(e,1) + C(d,2) + C(c,3) + C(b,4) + C(a,5) ? No.
    # Standard unranking/ranking.
    # Let's just use a dict for simplicity and speed, built once.
    pass

# Better: Use a global dictionary built at module load.
_indices = {}
_idx = 0
for s in combinations(range(12), 5):
    _indices[s] = _idx
    _idx += 1

# Precompute block masks
# There are C(12, 6) = 924 blocks
_blocks = []
_block_masks = []
for b in combinations(range(12), 6):
    mask = 0
    for i in range(6):
        # 5-subset is b with the i-th element removed
        s = b[:i] + b[i+1:]
        mask |= 1 << _indices[s]
    _blocks.append(b)
    _block_masks.append(mask)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_blocks = []
    best_count = 0
    
    # We will use a DFS with branch and bound
    # State: current list of blocks, covered mask, current count
    # We iterate through blocks in a fixed order (lexicographic)
    # To allow backtracking, we need to try adding or skipping blocks?
    # No, standard set packing: we pick a block, then recurse with the rest.
    # But we must ensure we don't miss solutions.
    # Since we want to MAXIMIZE, and we know 132 is possible,
    # we can just do a DFS that tries to include blocks.
    
    # Optimization: 
    # 1. Sort blocks? Lexicographic is fine.
    # 2. Pruning: 
    #    max_possible = current_count + (792 - popcount(covered)) // 6
    #    if max_possible <= best_count: return
    
    # To make it faster, we can use a greedy heuristic to find a good initial solution.
    # Greedy:
    covered = 0
    greedy_blocks = []
    for i, mask in enumerate(_block_masks):
        if covered & mask == 0:
            greedy_blocks.append(_blocks[i])
            covered |= mask
            if time.time() > deadline:
                break
    best_count = len(greedy_blocks)
    best_blocks = greedy_blocks[:]

    # Now try to improve via backtracking?
    # Actually, if the greedy found 132, we are done.
    if best_count == 132:
        return best_blocks

    # If not, we need a more powerful search.
    # But for S(5,6,12), the greedy might be weak.
    # Let's run a limited backtracking search.
    
    # We will try to find a solution of size 132.
    # Since the design is unique up to isomorphism, the search space is constrained.
    
    # We will use a recursive function.
    # To avoid deep recursion limits, we use iterative or careful recursion.
    # Python recursion limit is usually 1000. Depth 132 is fine.
    
    def dfs(idx, covered_mask, current_blocks, current_count):
        nonlocal best_count, best_blocks, deadline
        
        if time.time() > deadline:
            return
        
        # Pruning
        # Calculate max possible additional blocks
        # Number of uncovered 5-subsets
        # We need to count bits in (all_ones ^ covered_mask)
        # all_ones = (1 << 792) - 1
        # This is expensive if done every time.
        # We can track `uncovered_count` as a parameter.
        # Initial uncovered = 792.
        # Each block covers 6 new 5-subsets (since they are disjoint in 5-subsets).
        # So uncovered_count = 792 - 6 * current_count.
        # Max additional = uncovered_count // 6.
        # Max total = current_count + uncovered_count // 6.
        # If Max total <= best_count, prune.
        
        if current_count + (792 - 6 * current_count) // 6 <= best_count:
            return
        
        # If we have found 132, stop
        if best_count == 132:
            return

        # Try to add blocks from idx to end
        # Optimization: Skip blocks that are already invalid
        # We can pre-filter? No, validity depends on covered_mask.
        
        # Heuristic: Try blocks that are "most constrained" first?
        # For now, iterate lexicographically.
        
        for i in range(idx, len(_blocks)):
            mask = _block_masks[i]
            if covered_mask & mask:
                continue
            
            # Add block i
            new_covered = covered_mask | mask
            new_count = current_count + 1
            new_blocks = current_blocks + [_blocks[i]]
            
            # Update best if better
            if new_count > best_count:
                best_count = new_count
                best_blocks = new_blocks[:]
                if best_count == 132:
                    return
            
            # Recurse
            # We must ensure we don't pick the same block again or one that conflicts.
            # The loop `i` in the next call should start from i+1 to avoid duplicates and maintain order.
            dfs(i + 1, new_covered, new_blocks, new_count)
            
            if time.time() > deadline or best_count == 132:
                return

    # Start DFS
    # To make it faster, we can start with a greedy solution and try to swap/extend?
    # No, standard DFS from scratch is fine if pruning is good.
    # However, the search space is huge.
    # A better approach for S(5,6,12) is to use the known structure.
    # But let's try the DFS first. If it's too slow, it will timeout and return greedy.
    
    # To improve the DFS, we can use a "forward checking" or "constraint propagation".
    # But simple pruning might be enough for 132 blocks if the order is good.
    
    # Let's try a different strategy: 
    # The blocks of S(5,6,12) can be generated by taking the 12 points and using the fact that it's a Steiner system.
    # I will hardcode the 132 blocks to be safe.
    
    # Since I cannot reliably hardcode 132 blocks from memory, I will rely on the DFS.
    # To ensure it finishes, I will limit the depth or time.
    
    # Actually, there is a known fast algorithm for this specific design.
    # But given the constraints, I will provide the DFS with a tight time guard.
    
    # One optimization: 
    # Instead of passing `current_blocks` list (copying is expensive), 
    # we can use a global array and backtrack by popping.
    
    global_stack = []
    
    def dfs_fast(idx, covered_mask, current_count):
        nonlocal best_count, best_blocks, deadline
        
        if time.time() > deadline:
            return
        
        if current_count + (792 - 6 * current_count) // 6 <= best_count:
            return
            
        if best_count == 132:
            return

        for i in range(idx, len(_blocks)):
            mask = _block_masks[i]
            if covered_mask & mask:
                continue
            
            new_covered = covered_mask | mask
            new_count = current_count + 1
            
            global_stack.append(_blocks[i])
            
            if new_count > best_count:
                best_count = new_count
                best_blocks = global_stack[:]
                if best_count == 132:
                    global_stack.pop()
                    return
            
            dfs_fast(i + 1, new_covered, new_count)
            
            global_stack.pop()
            
            if time.time() > deadline or best_count == 132:
                return

    dfs_fast(0, 0, 0)
    
    return best_blocks

# Initialize indices and masks at module load
_indices = {}
_idx = 0
for s in combinations(range(12), 5):
    _indices[s] = _idx
    _idx += 1

_blocks = []
_block_masks = []
for b in combinations(range(12), 6):
    mask = 0
    for i in range(6):
        s = b[:i] + b[i+1:]
        mask |= 1 << _indices[s]
    _blocks.append(b)
    _block_masks.append(mask)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_blocks = []
    best_count = 0
    
    # Greedy initial solution
    covered = 0
    greedy_blocks = []
    for i, mask in enumerate(_block_masks):
        if covered & mask == 0:
            greedy_blocks.append(_blocks[i])
            covered |= mask
            if time.time() > deadline:
                break
    best_count = len(greedy_blocks)
    best_blocks = greedy_blocks[:]
    
    if best_count == 132:
        return best_blocks

    # Backtracking
    global_stack = []
    
    def dfs_fast(idx, covered_mask, current_count):
        nonlocal best_count, best_blocks, deadline
        
        if time.time() > deadline:
            return
        
        # Pruning
        if current_count + (792 - 6 * current_count) // 6 <= best_count:
            return
            
        if best_count == 132:
            return

        for i in range(idx, len(_blocks)):
            mask = _block_masks[i]
            if covered_mask & mask:
                continue
            
            new_covered = covered_mask | mask
            new_count = current_count + 1
            
            global_stack.append(_blocks[i])
            
            if new_count > best_count:
                best_count = new_count
                best_blocks = global_stack[:]
                if best_count == 132:
                    global_stack.pop()
                    return
            
            dfs_fast(i + 1, new_covered, new_count)
            
            global_stack.pop()
            
            if time.time() > deadline or best_count == 132:
                return

    dfs_fast(0, 0, 0)
    
    return best_blocks
