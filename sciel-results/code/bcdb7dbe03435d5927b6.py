import itertools
import random
import time
import sys

# Precomputation for v=24, t=3
V = 24
T = 3
K = 7

# Map each t-subset to an index
def get_triple_index(triple):
    # triple is a sorted tuple of 3 ints
    # We can just use the tuple itself as a key in a dict, or map to int.
    # For speed, mapping to int is better for array lookups.
    # C(24,3) = 2024.
    pass

# We will use a dictionary mapping triple tuple to index
TRIPLE_TO_IDX = {}
IDX_TO_TRIPLE = []
def init_precomputation():
    global TRIPLE_TO_IDX, IDX_TO_TRIPLE
    idx = 0
    for comb in itertools.combinations(range(V), T):
        TRIPLE_TO_IDX[comb] = idx
        IDX_TO_TRIPLE.append(comb)
        idx += 1

init_precomputation()

def construct(v, k, t):
    # Assume v=24, k=7, t=3 based on task description
    # But handle generic case if possible? The task says "evaluator passes this task's parameters".
    # It implies v, k, t are fixed for this task instance.
    # The seed code uses generic v, k, t.
    # However, the precomputation is specific to v=24, t=3.
    # If v, k, t are different, this will fail.
    # The task description says "Covering design C(24,7,3)".
    # So we can assume v=24, k=7, t=3.
    # But to be safe, check if v != 24 or k != 7 or t != 3, fallback to simple greedy?
    # The prompt says "evaluator passes this task's parameters v, k and t".
    # It is highly likely they are 24, 7, 3.
    
    if v != 24 or k != 7 or t != 3:
        # Fallback to simple greedy for other params
        return _simple_greedy(v, k, t)

    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    best_blocks = None
    best_len = float('inf')
    
    # Set random seed for reproducibility in testing, but use different seeds for restarts
    # random.seed(42)
    
    while time.time() < deadline:
        blocks = _greedy_cover_v24()
        blocks = _refine_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
            # If we hit the known best, we can try to push for lower bound?
            # Best known is 76. Lower bound 69.
            # If we find 76, we can keep searching for 75?
            # It's unlikely to find < 76 quickly.
            
    return best_blocks

def _simple_greedy(v, k, t):
    """Fallback for non-standard parameters"""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        # Pick block covering most uncovered
        best_block = None
        max_gain = -1
        # This is O(C(v,k) * C(k,t)) which is slow for large v, but ok for small
        for block in itertools.combinations(range(v), k):
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = block
        if max_gain == 0:
            break
        blocks.append(list(best_block))
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    return blocks

def _greedy_cover_v24():
    """
    Optimized greedy for v=24, k=7, t=3.
    Strategy:
    1. Maintain set of uncovered triple indices.
    2. While uncovered:
       a. Pick a random uncovered triple T.
       b. Generate all blocks containing T.
       c. For each such block, count how many *other* uncovered triples it covers.
       d. Pick block with max count.
       e. Add block, update uncovered set.
    """
    all_triples_idx = set(range(2024))
    uncovered = all_triples_idx.copy()
    blocks = []
    
    while uncovered:
        # Pick a random uncovered triple
        # Using list(uncovered) is O(N). 
        # For better performance, we could maintain a list and swap-remove.
        # But set is fine for ~2000 elements.
        t_idx = random.choice(list(uncovered))
        t_tuple = IDX_TO_TRIPLE[t_idx]
        
        # Generate all blocks containing t_tuple
        # t_tuple has 3 elements. We need to choose 4 more from the remaining 21.
        # Points not in t_tuple
        points_in_t = set(t_tuple)
        available_points = [p for p in range(24) if p not in points_in_t]
        
        best_block = None
        max_gain = -1
        
        # Iterate all 4-subsets of available_points
        # There are C(21,4) = 5985 such subsets.
        for extra in itertools.combinations(available_points, 4):
            block = tuple(sorted(list(t_tuple) + list(extra)))
            
            # Calculate gain: number of triples in this block that are in uncovered
            gain = 0
            # The block has C(7,3)=35 triples.
            # One of them is t_tuple (which is in uncovered).
            # We can just iterate all 35.
            for sub in itertools.combinations(block, 3):
                idx = TRIPLE_TO_IDX[sub]
                if idx in uncovered:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_block = block
                
        # Add best_block
        if best_block is not None:
            blocks.append(list(best_block))
            # Update uncovered
            for sub in itertools.combinations(best_block, 3):
                idx = TRIPLE_TO_IDX[sub]
                uncovered.discard(idx)
                
    return blocks

def _refine_blocks(blocks):
    """
    Remove redundant blocks.
    A block is redundant if all its triples are covered by other blocks.
    """
    if not blocks:
        return blocks
    
    # Count how many times each triple is covered
    cover_count = [0] * 2024
    
    # Map block index to its triples indices
    block_triples = []
    for b in blocks:
        triples = []
        for sub in itertools.combinations(b, 3):
            idx = TRIPLE_TO_IDX[sub]
            triples.append(idx)
        block_triples.append(triples)
        
    for triples in block_triples:
        for idx in triples:
            cover_count[idx] += 1
            
    # Iteratively remove redundant blocks
    changed = True
    while changed:
        changed = False
        new_blocks = []
        new_block_triples = []
        
        # We need to re-evaluate redundancy based on the current set of blocks.
        # It's safer to just check each block against the current cover_count,
        # but removing one affects the others.
        # So we do one pass:
        # For each block, check if it is the *only* one covering any of its triples.
        # If yes, it's essential.
        # If no, it's redundant (for this pass).
        
        # However, removing a redundant block might make another block essential?
        # No, removing a block only *decreases* coverage.
        # If a block was redundant (all its triples covered >= 2 times), removing it makes them covered >= 1 time.
        # It doesn't make any other block "more essential" in a way that prevents removal of others.
        # Wait.
        # If Block A covers T1, T2. Block B covers T1, T2.
        # Both are redundant.
        # Remove A. T1, T2 covered by B (count 1).
        # Now B is essential.
        # So we can only remove blocks that are redundant *in the current set*.
        # If we remove all redundant blocks in one pass, we might remove A and B.
        # Then T1, T2 are uncovered!
        # So we must be careful.
        
        # Correct approach:
        # 1. Identify all blocks that are "non-essential" (no triple is covered only by this block).
        # 2. We can remove *at most* one such block? No.
        # 3. Actually, if a block is non-essential, removing it is safe *if* we don't remove others that depend on it?
        #    No, "non-essential" means every triple in it has count >= 2.
        #    If we remove it, those triples become count >= 1. They are still covered.
        #    So removing ANY non-essential block is safe.
        #    Can we remove multiple?
        #    If A and B are both non-essential.
        #    Remove A. A's triples still covered by others.
        #    Is B still non-essential?
        #    B's triples might have been covered by A.
        #    If B had a triple T covered only by A and B, count was 2.
        #    After removing A, count is 1. B is now essential.
        #    So we should remove blocks one by one, re-evaluating.
        
        # Efficient way:
        # Loop through blocks.
        # For each block, check if it is essential.
        # If not, remove it and update cover_counts.
        
        i = 0
        current_blocks = blocks
        current_block_triples = block_triples
        current_cover_count = cover_count
        
        while i < len(current_blocks):
            is_essential = False
            for idx in current_block_triples[i]:
                if current_cover_count[idx] == 1:
                    is_essential = True
                    break
            
            if is_essential:
                i += 1
            else:
                # Remove block i
                for idx in current_block_triples[i]:
                    current_cover_count[idx] -= 1
                current_blocks.pop(i)
                current_block_triples.pop(i)
                changed = True
                # Do not increment i, check next block at same index
                
    return current_blocks
