import itertools
import time
import random

def construct(v, k, t):
    # Precompute all t-subsets and map to indices
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # 3D lookup for triple index
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
    
    # Precompute for each point x, the list of triple indices involving x
    # triples_with[x] = list of (triple_idx, other1, other2)
    triples_with = [[] for _ in range(v)]
    for idx, (a, b, c) in enumerate(triples):
        triples_with[a].append((idx, b, c))
        triples_with[b].append((idx, a, c))
        triples_with[c].append((idx, a, b))
    
    # Precompute for each pair (i,j) with i<j, the list of triple indices where i and j are in the triple
    # pair_triples[i][j] = list of triple indices
    pair_triples = [[[] for _ in range(v)] for _ in range(v)]
    for idx, (a, b, c) in enumerate(triples):
        pair_triples[a][b].append(idx)
        pair_triples[a][c].append(idx)
        pair_triples[b][a].append(idx)
        pair_triples[b][c].append(idx)
        pair_triples[c][a].append(idx)
        pair_triples[c][b].append(idx)
    
    def greedy_cover(random_state, time_limit=None):
        rng = random.Random(random_state)
        uncovered = bytearray([1] * num_triples)
        num_uncovered = num_triples
        blocks = []
        points = list(range(v))
        
        while num_uncovered > 0:
            # Find an uncovered triple - use demand-based ordering
            # Pick the uncovered triple that appears in the most potential blocks
            # For simplicity, pick first uncovered, but we can be smarter
            
            # Actually, let's pick the uncovered triple with maximum "demand"
            # Demand = number of blocks that could cover it
            # A block containing triple (a,b,c) can have any k-t=4 other points
            # So demand is C(v-3, k-3) for all triples - same for all!
            # So demand-based ordering doesn't help here.
            
            # Instead, let's pick a random uncovered triple for diversity
            # Find all uncovered indices
            uncovered_indices = [i for i in range(num_triples) if uncovered[i]]
            if not uncovered_indices:
                break
            target_idx = rng.choice(uncovered_indices)
            target_triple = triples[target_idx]
            
            # Start with the target triple
            block = list(target_triple)
            uncovered[target_idx] = 0
            num_uncovered -= 1
            
            # Add remaining k-t points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in points if x not in block]
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    # Count triples (x, u, v) where u, v in block
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            p1 = block[i]
                            p2 = block[j]
                            idx = get_triple_idx(x, p1, p2)
                            if idx >= 0 and uncovered[idx]:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        if best_gain == len(block)*(len(block)-1)//2:
                            break  # Max possible gain achieved
                
                if best_point == -1:
                    break
                
                block.append(best_point)
                
                # Update uncovered status
                for i in range(len(block)):
                    if block[i] == best_point:
                        continue
                    for j in range(i+1, len(block)):
                        if block[j] == best_point:
                            continue
                        p1 = block[i]
                        p2 = block[j]
                        idx = get_triple_idx(best_point, p1, p2)
                        if idx >= 0 and uncovered[idx]:
                            uncovered[idx] = 0
                            num_uncovered -= 1
            
            blocks.append(tuple(sorted(block)))
            
            # Time check
            if time_limit and time.time() > time_limit:
                break
        
        return blocks
    
    def trim_blocks(blocks):
        # Remove redundant blocks
        counts = [0] * num_triples
        
        for block in blocks:
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
                essential = False
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active_blocks[i] = False
                    changed = True
                    for triple in itertools.combinations(block, t):
                        idx = triple_to_idx[triple]
                        counts[idx] -= 1
        
        return [blocks[i] for i in range(len(blocks)) if active_blocks[i]]
    
    def local_search(blocks, random_state, time_limit=None):
        """Try to improve the covering by removing and re-adding blocks."""
        rng = random.Random(random_state)
        blocks = list(blocks)
        
        # Check if valid
        def is_valid(block_list):
            covered = bytearray([0] * num_triples)
            for block in block_list:
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    covered[idx] = 1
            return all(covered)
        
        if not is_valid(blocks):
            return blocks
        
        improved = True
        while improved and (not time_limit or time.time() < time_limit):
            improved = False
            
            # Try removing each block
            for i in range(len(blocks)):
                if time_limit and time.time() > time_limit:
                    break
                
                candidate = blocks[:i] + blocks[i+1:]
                if is_valid(candidate):
                    blocks = candidate
                    improved = True
                    break
            
            # Try replacing a block with a better one
            # Find uncovered triples after removal? No, we just removed if valid.
            # Try to find a block that covers more uncovered triples
            # Actually, let's try a different local search:
            # For each block, try to find a replacement that covers the same or more triples
        
        return blocks
    
    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 145  # 5s buffer before 150s limit
    
    seed_counter = 0
    while time.time() < deadline:
        # Run greedy with different seeds
        blocks = greedy_cover(seed_counter)
        
        # Trim redundant blocks
        blocks = trim_blocks(blocks)
        
        # Local search
        remaining_time = deadline - time.time()
        if remaining_time > 2:
            blocks = local_search(blocks, seed_counter, time.time() + remaining_time - 1)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # Print for debugging (will be removed)
            # print(f"Seed {seed_counter}: {len(blocks)} blocks")
        
        # If we hit the lower bound, stop
        if best_len == 69:
            break
        
        seed_counter += 1
    
    # Convert tuples to lists for output
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback: should not happen
        return []
