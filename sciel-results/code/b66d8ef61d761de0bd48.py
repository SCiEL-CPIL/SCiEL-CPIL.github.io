<<<<<<< SEARCH
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
=======
    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = np.ones(num_triples, dtype=np.bool_)
        num_uncovered = num_triples
        blocks = []
        
        # Scarcity weight: number of uncovered triples containing each point
        point_uncovered = np.zeros(v, dtype=np.int32)
        for idx in range(num_triples):
            a, b, c = triples[idx]
            point_uncovered[a] += 1
            point_uncovered[b] += 1
            point_uncovered[c] += 1
        
        while num_uncovered > 0:
            # Find an uncovered triple with maximum scarcity weight
            # Weight = sum of point_uncovered for its three points
            start_idx = rng.randint(0, num_triples - 1)
            best_idx = -1
            best_weight = -1
            for offset in range(num_triples):
                idx = (start_idx + offset) % num_triples
                if uncovered[idx]:
                    a, b, c = triples[idx]
                    w = point_uncovered[a] + point_uncovered[b] + point_uncovered[c]
                    if w > best_weight:
                        best_weight = w
                        best_idx = idx
            if best_idx < 0:
                break
            
            target_triple = triples[best_idx]
            block = list(target_triple)
            
            # Mark this triple as covered
            uncovered[best_idx] = False
            num_uncovered -= 1
            for pt in target_triple:
                point_uncovered[pt] -= 1
            
            # Add remaining k - t points using scarcity-weighted gain
            while len(block) < k:
                block_set = set(block)
                best_score = -1
                best_point = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Gain: number of newly covered triples
                    gain = 0
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            u = block[i]
                            w = block[j]
                            idx = get_triple_idx(x, u, w)
                            if uncovered[idx]:
                                gain += 1
                    # Scarcity bonus: prefer points that are "hard" (few uncovered triples)
                    scarcity = point_uncovered[x]
                    # Combined score: primary is gain, secondary is scarcity (lower scarcity = harder)
                    # Use a weighted combination
                    score = gain * 1000 - scarcity
                    if score > best_score:
                        best_score = score
                        best_point = x
                        best_candidates = [x]
                    elif score == best_score:
                        best_candidates.append(x)
                
                if not best_candidates:
                    break
                    
                best_point = rng.choice(best_candidates)
                block.append(best_point)
                
                # Update uncovered and point_uncovered
                old_block = block[:-1]
                for i in range(len(old_block)):
                    for j in range(i+1, len(old_block)):
                        u = old_block[i]
                        w = old_block[j]
                        idx = get_triple_idx(best_point, u, w)
                        if uncovered[idx]:
                            uncovered[idx] = False
                            num_uncovered -= 1
                            a, b, c = triples[idx]
                            point_uncovered[a] -= 1
                            point_uncovered[b] -= 1
                            point_uncovered[c] -= 1
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks
>>>>>>> REPLACE

