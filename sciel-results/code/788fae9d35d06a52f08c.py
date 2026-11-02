<<<<<<< SEARCH
    # Run multiple greedy attempts with randomization
    # The number of attempts can be scaled by remaining time
    # But first, let's run one deterministic attempt to get a baseline
    
    def greedy_cover(randomize=False):
        uncovered = np.ones(num_triples, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            # Pick an uncovered triple to start the block
            # If randomize, pick a random uncovered triple, else the first
            uncovered_indices = np.where(uncovered)[0]
            if randomize:
                start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            else:
                start_idx = uncovered_indices[0]
            
            start_triple = triples[start_idx]
            block = list(start_triple)
            
            # Greedily add points
            while len(block) < k:
                best_gain = -1
                best_x = -1
                # Precompute pairs in current block for gain calculation
                # Pairs are (y, z) with y < z
                pairs = list(itertools.combinations(block, 2))
                
                for x in range(v):
                    if x in block:
                        continue
                    # Calculate gain
                    # Indices of triples (x, y, z) for y, z in pairs
                    # We can vectorize this lookup
                    # Create arrays for y and z
                    ys = np.array([p[0] for p in pairs])
                    zs = np.array([p[1] for p in pairs])
                    # Lookup indices
                    idxs = LUT[x, ys, zs]
                    # Filter out invalid indices (shouldn't happen if x not in block)
                    # Sum uncovered bits
                    gain = np.sum(uncovered[idxs])
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and randomize:
                        # Random tie-breaking
                        if random.random() < 0.5:
                            best_x = x
                
                if best_x == -1:
                    break # Should not happen
                block.append(best_x)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Mark triples covered
            for a, b, c in itertools.combinations(block, t):
                idx = triple_to_idx[(a, b, c)]
                uncovered[idx] = False
        
        return blocks
=======
    # Precompute all pairs for each possible block size
    # For a block of size m, we need pairs from that block
    # We'll compute gain on the fly but optimize the inner loop
    
    def greedy_cover(randomize=False):
        uncovered = np.ones(num_triples, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            # Pick an uncovered triple to start the block
            # If randomize, pick a random uncovered triple, else the first
            uncovered_indices = np.where(uncovered)[0]
            if randomize:
                start_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            else:
                start_idx = uncovered_indices[0]
            
            start_triple = triples[start_idx]
            block = list(start_triple)
            block_set = set(block)
            
            # Greedily add points
            while len(block) < k:
                best_gain = -1
                best_x = -1
                # Precompute pairs in current block for gain calculation
                pairs = list(itertools.combinations(block, 2))
                if not pairs:
                    break
                    
                ys = np.array([p[0] for p in pairs], dtype=np.int32)
                zs = np.array([p[1] for p in pairs], dtype=np.int32)
                
                # Vectorize: for each candidate x, compute gain
                # LUT[x, ys, zs] gives indices for all pairs
                # We need to handle this efficiently
                # For each x not in block, compute gain
                candidates = [x for x in range(v) if x not in block_set]
                
                if not candidates:
                    break
                    
                # Compute gains for all candidates at once
                # LUT is (v, v, v), ys and zs are (m,)
                # We want LUT[candidates, ys[:, None], zs[:, None]] -> (len(candidates), len(ys))
                # But that's expensive. Let's do it in a loop but with numpy
                gains = np.zeros(len(candidates), dtype=np.int32)
                
                for i, x in enumerate(candidates):
                    idxs = LUT[x, ys, zs]
                    # Filter valid indices
                    valid = idxs >= 0
                    if np.any(valid):
                        gains[i] = np.sum(uncovered[idxs[valid]])
                
                # Find best
                max_gain = np.max(gains)
                if max_gain <= 0:
                    break
                    
                best_idx = np.where(gains == max_gain)[0]
                if randomize and len(best_idx) > 1:
                    best_i = best_idx[random.randint(0, len(best_idx) - 1)]
                else:
                    best_i = best_idx[0]
                
                best_x = candidates[best_i]
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(tuple(block))
            
            # Mark triples covered
            for a, b, c in itertools.combinations(block, t):
                idx = triple_to_idx[(a, b, c)]
                uncovered[idx] = False
        
        return blocks
>>>>>>> REPLACE

