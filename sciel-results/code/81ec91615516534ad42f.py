import itertools
import time
import random
import numpy as np

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
    idx_3d = np.full((v, v, v), -1, dtype=np.int32)
    for a in range(v):
        for b in range(v):
            for c in range(v):
                if a == b or a == c or b == c:
                    continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a, b, c] = triple_to_idx[(t1, t2, t3)]
    
    def get_triple_idx(a, b, c):
        return int(idx_3d[a, b, c])
        
    # Precompute for each pair (i, j), the list of triple indices for all third points x
    # pair_triples[i][j] = list of (x, triple_idx) for all x != i, j
    pair_triples = [[None] * v for _ in range(v)]
    for i in range(v):
        for j in range(v):
            if i == j:
                continue
            pair_triples[i][j] = []
            for x in range(v):
                if x == i or x == j:
                    continue
                idx = get_triple_idx(x, i, j)
                pair_triples[i][j].append((x, idx))
    
    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = np.ones(num_triples, dtype=np.bool_)
        num_uncovered = num_triples
        blocks = []
        
        # List of points
        points = list(range(v))
        
        while num_uncovered > 0:
            # Find an uncovered triple index
            # Use random choice to break symmetry
            # We can generate random indices until we find an uncovered one
            # Or just scan. Scanning is O(N) but N=2024.
            # Random is better for diversity.
            
            # Efficient way: get indices of True values? np.where is O(N).
            # Since N is small, let's just pick a random index and check.
            # But if coverage is high, this might take many tries.
            # Better: maintain a list of uncovered indices?
            # Or just scan from a random start point.
            
            start_idx = rng.randint(0, num_triples - 1)
            for offset in range(num_triples):
                idx = (start_idx + offset) % num_triples
                if uncovered[idx]:
                    break
            else:
                break # Should not happen
            
            target_triple = triples[idx]
            block = list(target_triple)
            
            # Mark this triple as covered
            uncovered[idx] = False
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                
                # Candidates are points not in block
                # For each candidate x, gain is number of pairs (u,v) in block such that (x,u,v) is uncovered
                # We can precompute the gain for all x at once?
                # Gain(x) = sum over pairs (u,v) in block of uncovered[get_triple_idx(x, u, v)]
                
                # Let's calculate gain for each x
                # Optimization: 
                # For a fixed block B, Gain(x) = sum_{u < v in B} I(uncovered[x, u, v])
                
                # We can iterate over all x not in B
                # B is small (size 3..6)
                # So we can just loop x
                
                # To speed up, we can use numpy if we vectorize over x?
                # But B changes each step.
                
                # Let's just do a simple loop, but optimized with local variables
                block_set = set(block)
                
                # We want to find x maximizing gain
                # We can also add a small random noise to break ties or explore?
                # The prompt suggests smooth surrogate or tie-breaking.
                # Let's stick to max gain, with random tie-breaking.
                
                gains = {}
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    # Iterate pairs in block
                    # len(block) is at most 6, so pairs are at most 15
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            u = block[i]
                            w = block[j]
                            idx = get_triple_idx(x, u, w)
                            if uncovered[idx]:
                                gain += 1
                    gains[x] = gain
                
                # Find max gain
                max_gain = max(gains.values()) if gains else 0
                # Get all x with max gain
                best_candidates = [x for x, g in gains.items() if g == max_gain]
                
                if not best_candidates:
                    break
                    
                best_point = rng.choice(best_candidates)
                block.append(best_point)
                
                # Update uncovered
                # New triples covered are (best_point, u, w) for u, w in block (excluding best_point)
                # Note: we already covered (target_triple) at start.
                # Now we add best_point to block.
                # The new triples involving best_point and pairs from the OLD block are covered.
                # Wait, the logic in the original code was:
                # "Update uncovered status for all triples covered by the new point x"
                # "These are triples (x, p1, p2) for p1, p2 in block (excluding x)"
                # This is correct because any triple containing x and two existing points is now covered.
                # We don't need to check triples with x and one existing point and one new point?
                # No, because we build the block sequentially.
                # When we add x, the new triples formed are (x, u, v) where u, v are already in the block.
                # Triples (x, u, w) where w is added later will be handled when w is added.
                
                # So we mark (best_point, u, v) for all u, v in block (excluding best_point)
                # Note: block currently includes best_point at the end.
                # So we iterate pairs from block[:-1]
                
                old_block = block[:-1]
                for i in range(len(old_block)):
                    for j in range(i+1, len(old_block)):
                        u = old_block[i]
                        w = old_block[j]
                        idx = get_triple_idx(best_point, u, w)
                        if uncovered[idx]:
                            uncovered[idx] = False
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

    def local_search(blocks, time_limit):
        # Try to swap points in blocks to improve coverage?
        # Or remove blocks and re-add?
        # A simple local search: 
        # 1. Try to remove a block. If coverage is lost, try to find a replacement block that covers the lost triples.
        # 2. Or, try to replace a point in a block with another point to increase coverage of other triples.
        
        # Let's try a simple "block replacement" search.
        # For each block B, try removing it.
        # Identify triples covered ONLY by B.
        # If none, remove B. (This is what trim_blocks does, but maybe we can do it iteratively with re-optimization?)
        
        # Better: 
        # While time allows:
        #   Pick a random block B.
        #   Try to replace one point x in B with y not in B.
        #   Check if new block B' is valid (distinct points).
        #   Check if B' covers at least the same number of *uncovered* triples as B?
        #   Actually, we want to minimize total blocks.
        #   
        #   Alternative: 
        #   Pick a random block B.
        #   Remove B.
        #   Find triples that are now uncovered (lost).
        #   Try to find a new block B' that covers ALL lost triples.
        #   If found, replace B with B'.
        #   This might reduce redundancy if B' covers more other triples.
        #   But it keeps the same number of blocks.
        
        # To reduce number of blocks, we need to remove a block and cover its unique triples with *existing* other blocks?
        # No, that's impossible if they were unique.
        # So we must modify other blocks to cover the unique triples, then remove B.
        
        # This is complex. Let's stick to a simpler improvement:
        # "Greedy Repair"
        # If we have a valid covering, try to remove blocks one by one (random order).
        # If removing a block breaks coverage, try to fix it by modifying other blocks?
        # Or just keep it.
        
        # Given the time constraint and complexity, maybe a simple random restart with better tie-breaking is enough.
        # The current greedy is the main driver.
        
        # Let's implement a simple "swap" local search to escape local minima in the greedy construction?
        # No, greedy is constructive.
        
        # Let's just return the blocks as is for now, relying on multiple restarts.
        # But we can try to "compress" the covering by checking if any block can be replaced by a subset of others?
        # No, blocks are size k.
        
        # Idea: Try to replace a block B with a different block B' that has the same "core" triples but covers more "peripheral" ones?
        # This is hard to define.
        
        # Let's try a simple heuristic:
        # For each block, try swapping one point with another to see if we can increase the "gain" (number of triples covered by this block that are covered by NO OTHER block).
        # This is like maximizing the "unique coverage" of each block.
        
        # Actually, the most effective local search for covering designs is often:
        # 1. Start with a covering.
        # 2. Randomly remove a block.
        # 3. If invalid, try to add a new block that covers the missing triples, preferring blocks that also cover other "rare" triples.
        # 4. If valid, keep the removal (reduces block count by 1).
        
        # Let's implement this "Remove and Repair" strategy.
        
        blocks = [list(b) for b in blocks]
        n_blocks = len(blocks)
        
        # Precompute coverage counts
        # triple_counts: count of blocks covering each triple
        # block_triples: list of triple indices for each block
        
        triple_counts = [0] * num_triples
        block_triples = []
        
        for block in blocks:
            bt = []
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                bt.append(idx)
                triple_counts[idx] += 1
            block_triples.append(bt)
        
        start_time = time.time()
        
        # Try to remove blocks
        # We iterate through blocks in random order
        order = list(range(n_blocks))
        rng = random.Random(42) # Fixed seed for local search? Or pass in?
        # Let's use a random seed based on current time or something?
        # Better to pass rng or use a global one.
        # Let's just use a local rng seeded with time.
        rng = random.Random(int(time.time()))
        
        changed = True
        while changed and time.time() - start_time < time_limit:
            changed = False
            rng.shuffle(order)
            for i in order:
                if i >= len(blocks):
                    continue
                    
                # Check if block i is essential
                # i.e., does it cover any triple with count 1?
                essential = False
                for idx in block_triples[i]:
                    if triple_counts[idx] == 1:
                        essential = True
                        break
                        
                if essential:
                    continue
                
                # Try to remove block i
                # It is not essential, so removal is safe?
                # Wait, if count > 1 for all triples, removal is safe.
                # So we can just remove it?
                # Yes!
                
                # Remove block i
                for idx in block_triples[i]:
                    triple_counts[idx] -= 1
                    
                blocks.pop(i)
                block_triples.pop(i)
                
                changed = True
                # Adjust order indices?
                # Since we are iterating over a snapshot of order, and removing items,
                # indices in order might be invalid.
                # It's safer to restart the loop or use a different structure.
                # Given the small size, let's just break and restart the while loop.
                break
        
        return blocks

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    seed_counter = 0
    while time.time() < deadline:
        blocks = greedy_cover(seed_counter)
        blocks = trim_blocks(blocks)
        
        # Try local search to further reduce
        # Give it 5 seconds per iteration? No, total time is limited.
        # Let's give local search a small fraction of the remaining time.
        remaining = deadline - time.time()
        if remaining > 2:
            ls_time = min(5, remaining - 1)
            blocks = local_search(blocks, ls_time)
            blocks = trim_blocks(blocks) # Trim again after local search
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len == 69:
            break
            
        seed_counter += 1
    
    # Convert tuples to lists for output
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
