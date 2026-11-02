import itertools
import time
import random
import sys

def construct(v, k, t):
    # Parameters for the task: v=24, k=7, t=3
    # But we should use the passed arguments.
    
    # Precompute triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute 3D index lookup for speed: idx_3d[a][b][c] = index of triple (a,b,c)
    # -1 if not a valid triple (i.e., a==b etc, though we ensure distinct in logic)
    idx_3d = [[[-1]*v for _ in range(v)] for _ in range(v)]
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a == b or a == c or b == c:
                    continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a][b][c] = triple_to_idx[(t1, t2, t3)]
        
    def get_triple_idx(a, b, c):
        return idx_3d[a][b][c]
        
    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = bytearray([1] * num_triples)
        num_uncovered = num_triples
        blocks = []
        
        # List of points
        points = list(range(v))
        
        while num_uncovered > 0:
            # Find an uncovered triple
            # Randomly picking one might be better for diversity?
            # Or just the first one.
            # Let's pick a random index from uncovered ones to break symmetry
            # Finding a random 1 in bytearray is O(N). N=2024. Fine.
            # Or just iterate.
            
            # Pick first uncovered for determinism within a run?
            # No, let's pick a random one to help with local minima.
            # But we need to find one.
            # We can maintain a list of uncovered indices?
            # Removing from list is slow.
            # Bytearray scan is fast.
            
            # Let's just pick the first one found to keep it simple and fast.
            # Diversity comes from tie-breaking in point selection.
            
            # Actually, picking a random uncovered triple is better.
            # Let's shuffle indices and pick first 1?
            # Simpler: just pick index 0 if uncovered, else scan.
            # Let's stick to scanning for the first 1.
            
            start_idx = 0
            while start_idx < num_triples and uncovered[start_idx] == 0:
                start_idx += 1
                
            if start_idx >= num_triples:
                break # Should not happen if num_uncovered > 0
                
            # Get the triple
            target_triple = triples[start_idx]
            block = list(target_triple)
            
            # Mark this triple as covered?
            # It will be covered by the block we build.
            # We can mark it now to avoid re-considering it?
            # But we need to calculate gain for other points.
            # If we mark it now, it won't contribute to gain, which is correct 
            # because it's already "secured" by the fact that we started with it.
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in points if x not in block]
                
                # Shuffle candidates for random tie-breaking
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    # Count triples (x, u, v) where u, v in block
                    # u, v are distinct elements of block
                    # We can iterate pairs in block
                    # Block size is small (3 to 6)
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            p1 = block[i]
                            p2 = block[j]
                            # Check if triple (x, p1, p2) is uncovered
                            # Get index
                            idx = get_triple_idx(x, p1, p2)
                            if uncovered[idx]:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        # If gain is max possible, we can break?
                        # Max possible is C(len(block), 2)
                        # But we don't know max without calculating.
                    
                if best_point == -1:
                    # Should not happen if num_uncovered > 0 and v > k
                    break
                
                block.append(best_point)
                
                # Update uncovered status for all triples covered by the new point x
                # These are triples (x, p1, p2) for p1, p2 in block (excluding x)
                for i in range(len(block)):
                    if block[i] == best_point:
                        continue
                    for j in range(i+1, len(block)):
                        if block[j] == best_point:
                            continue
                        p1 = block[i]
                        p2 = block[j]
                        # Triple is (best_point, p1, p2)
                        idx = get_triple_idx(best_point, p1, p2)
                        if uncovered[idx]:
                            uncovered[idx] = 0
                            num_uncovered -= 1
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks

    def trim_blocks(blocks):
        # Count frequency of each triple
        counts = [0] * num_triples
        
        for block in blocks:
            # Iterate all triples in the block
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active_blocks = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i, block in enumerate(blocks):
                if not active_blocks[i]:
                    continue
                # Check if any triple in this block has count 1
                essential = False
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove block
                    active_blocks[i] = False
                    changed = True
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active_blocks[i]]

    def local_search(blocks, random_state, max_iters=1000, time_limit=10.0):
        """
        Local search to reduce the number of blocks.
        We try to remove a block if it is not essential.
        If no block can be removed, we try to swap a block with another that covers more unique triples?
        Or simply: try to remove blocks one by one. If removal breaks coverage, we must keep it.
        This is the 'trim' algorithm but applied iteratively in a random order.
        
        Better approach:
        1. Start with a valid covering.
        2. Randomly pick a block.
        3. Try to remove it.
        4. If coverage is broken, we need to add a new block that covers the missing triples.
           This is complex.
        
        Simpler local search:
        1. Start with trimmed blocks.
        2. While time allows:
           a. Pick a random block B.
           b. Remove B.
           c. Find all uncovered triples.
           d. If none, accept (block count reduced).
           e. If some, try to add a single block that covers as many of these uncovered triples as possible.
              If such a block exists (and is not already in the set), add it.
              This keeps the count same (removed 1, added 1) but might allow future removals.
              However, usually we want to strictly reduce count.
              
        Let's stick to a simpler heuristic:
        Iterate through blocks in random order.
        For each block, check if it can be removed (i.e., all its triples are covered by other blocks).
        If yes, remove it.
        Repeat until no more blocks can be removed.
        
        This is essentially what trim_blocks does, but trim_blocks uses a fixed order (index).
        Local search can try different orders or random restarts of the removal process.
        
        Actually, the current trim_blocks is already a form of local search (greedy deletion).
        To improve, we can try to find a *better* deletion order or allow "swaps".
        
        Let's implement a "swap" local search:
        1. Current set of blocks S.
        2. Pick a block B in S to remove.
        3. Identify uncovered triples U.
        4. Find a block B' (not in S) that covers the maximum number of triples in U.
        5. If |U| is small, and B' covers all of U, then we can swap B with B'.
           This doesn't reduce count, but might change the structure to allow further reductions.
        6. After a swap, run trim again.
        
        This is computationally expensive.
        
        Alternative:
        Just run the greedy construction many times with different seeds and keep the best.
        The current code does this.
        
        Let's enhance the greedy construction itself.
        Instead of just picking the best gain, we can use a "look-ahead" or "randomized gain".
        
        Also, we can try to construct the covering in a more structured way.
        For C(24,7,3), known constructions use group divisible designs or similar.
        But we don't have that knowledge hardcoded.
        
        Let's focus on speeding up the greedy and increasing the number of restarts.
        """
        pass # Placeholder, we will implement a better local search below if needed.

    def local_search_trim(blocks, random_state, time_limit=5.0):
        """
        Attempt to further reduce blocks by random reordering and deletion.
        """
        rng = random.Random(random_state)
        blocks = list(blocks)
        start_time = time.time()
        
        # Precompute triple counts for current blocks
        counts = [0] * num_triples
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                
        active = [True] * len(blocks)
        
        while time.time() - start_time < time_limit:
            # Shuffle active blocks indices
            active_indices = [i for i in range(len(blocks)) if active[i]]
            rng.shuffle(active_indices)
            
            found_removal = False
            for i in active_indices:
                if not active[i]:
                    continue
                # Check if block i is essential
                essential = False
                for triple in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove it
                    active[i] = False
                    found_removal = True
                    for triple in itertools.combinations(blocks[i], t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
            if not found_removal:
                break
                
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    seed_counter = 0
    while time.time() < deadline:
        # Run greedy
        blocks = greedy_cover(seed_counter)
        
        # Trim
        blocks = trim_blocks(blocks)
        
        # Local search trim
        blocks = local_search_trim(blocks, seed_counter, time_limit=2.0)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we hit the lower bound, stop?
        # Lower bound is 69.
        if best_len == 69:
            break
            
        seed_counter += 1
        
    # Convert tuples to lists for output
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback: return empty or a trivial covering?
        # Should not happen.
        return []
