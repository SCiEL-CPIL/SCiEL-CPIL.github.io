import itertools
import time
import random

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
    
    # For fast lookup of triples containing a specific pair and a third point?
    # We need to check if a specific triple is uncovered.
    
    # Precompute for each point x, and each pair (u,v) in the block, the triple index.
    # Actually, just calculating the index on the fly is fast enough for v=24.
    
    def get_triple_idx(a, b, c):
        return idx_3d[a][b][c]
        
    # To speed up gain calculation:
    # When adding x to block P, we cover triples (x, u, v) for u,v in P.
    # We can precompute a list of triple indices for each pair (u,v)?
    # No, the third point varies.
    
    # Let's use a bytearray for uncovered status
    # 1 means uncovered, 0 means covered
    
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

    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    # Run a few deterministic runs first?
    # Then random.
    
    seed_counter = 0
    while time.time() < deadline:
        blocks = greedy_cover(seed_counter)
        blocks = trim_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # Optional: If we hit the lower bound, stop?
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
