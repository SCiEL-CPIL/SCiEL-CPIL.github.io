<<<<<<< SEARCH
    def greedy_construct(v, k, t, seed=0, bias=0.0):
        """
        Greedy construction.
        bias: 0.0 means pure greedy (max gain). 1.0 means random.
        Intermediate values mix the two.
        """
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        # To speed up gain calculation, we can precompute pairs in the current block
        # and check membership in uncovered set.
        
        while uncovered:
            # Pick a starting triplet
            # Strategy: Pick a random uncovered triplet, or one that is "hard"
            # For now, random is good for diversity.
            start_triplet_id = rng.choice(list(uncovered))
            start_triplet = all_triplets[start_triplet_id]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                best_point = -1
                max_gain = -1
                second_best_gain = -1
                second_best_point = -1
                
                # Pairs in current block (for t=3)
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                # Get candidates
                # Optimization: Instead of iterating all v points, we can restrict
                # to points that appear in uncovered triplets containing pairs from block?
                # That's complex. v=24 is small, so iterating all is fine.
                
                gains = {}
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1
                    gains[p] = gain
                    if gain > max_gain:
                        second_best_gain = max_gain
                        second_best_point = best_point
                        max_gain = gain
                        best_point = p
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_point = p

                if max_gain == 0:
                    # No gain, pick random
                    candidates = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(candidates)
                else:
                    # Mix greedy and random based on bias
                    # If bias is 0, always pick best.
                    # If bias is 1, always pick random from top candidates?
                    # Let's use a softmax-like selection or simple probability.
                    # P(best) = 1 - bias, P(second) = bias * 0.5, P(random) = bias * 0.5
                    r = rng.random()
                    if r < (1.0 - bias):
                        pass # keep best_point
                    elif second_best_point != -1 and r < (1.0 - bias) + bias * 0.5:
                        best_point = second_best_point
                    else:
                        candidates = [p for p in range(v) if p not in block_set]
                        best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)
        
        return blocks
=======
    def greedy_construct_weighted(v, k, t, seed=0, bias=0.0, weight_decay=1.05, weight_init=1.0):
        """
        Weighted greedy construction.
        Score for candidate point p = gain(p) / (weight[p] * product of weights of pairs in block)
        This encourages balanced point utilization and breaks symmetry.
        
        bias: 0.0 means pure weighted greedy. 1.0 means random.
        weight_decay: factor by which weights increase after each block containing a point
        weight_init: initial weight for all points
        """
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        
        # Dynamic point weights to encourage balanced utilization
        point_weights = np.ones(v, dtype=np.float64) * weight_init
        
        while uncovered:
            # Pick a starting triplet - prefer triplets with high-weight points
            # to balance coverage
            uncovered_list = list(uncovered)
            if not uncovered_list:
                break
            
            # Weighted selection: prefer triplets containing high-weight points
            triplet_weights = []
            for tid in uncovered_list:
                triplet = all_triplets[tid]
                w = 1.0
                for p in triplet:
                    w *= point_weights[p]
                triplet_weights.append(w)
            
            # Normalize to probabilities
            total_w = sum(triplet_weights)
            if total_w > 0:
                probs = [w / total_w for w in triplet_weights]
                start_idx = rng.choices(range(len(uncovered_list)), weights=probs)[0]
            else:
                start_idx = rng.randrange(len(uncovered_list))
            
            start_triplet_id = uncovered_list[start_idx]
            start_triplet = all_triplets[start_triplet_id]
            
            block = list(start_triplet)
            block_set = set(block)
            
            while len(block) < k:
                # Calculate weighted gain for each candidate point
                best_point = -1
                max_score = -1.0
                second_best_score = -1.0
                second_best_point = -1
                
                # Pairs in current block (for t=3)
                pairs_in_block = list(itertools.combinations(block, t - 1))
                
                # Compute product of pair weights for normalization
                pair_weight_product = 1.0
                for pair in pairs_in_block:
                    pair_weight_product *= point_weights[pair[0]] * point_weights[pair[1]]
                
                # If no pairs yet (first point), use 1.0
                if not pairs_in_block:
                    pair_weight_product = 1.0
                
                scores = {}
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for pair in pairs_in_block:
                        triplet = tuple(sorted(pair + (p,)))
                        tid = get_id(triplet)
                        if tid in uncovered:
                            gain += 1
                    
                    # Weighted score: gain / (point_weight * pair_weight_product)
                    # Higher weight means less likely to be chosen (penalty for overuse)
                    if gain > 0:
                        score = gain / (point_weights[p] * pair_weight_product)
                    else:
                        score = 0.0
                    
                    scores[p] = score
                    if score > max_score:
                        second_best_score = max_score
                        second_best_point = best_point
                        max_score = score
                        best_point = p
                    elif score > second_best_score:
                        second_best_score = score
                        second_best_point = p

                if max_score <= 0:
                    # No gain, pick random
                    candidates = [p for p in range(v) if p not in block_set]
                    best_point = rng.choice(candidates)
                else:
                    # Mix weighted greedy and random based on bias
                    r = rng.random()
                    if r < (1.0 - bias):
                        pass # keep best_point
                    elif second_best_point != -1 and r < (1.0 - bias) + bias * 0.5:
                        best_point = second_best_point
                    else:
                        candidates = [p for p in range(v) if p not in block_set]
                        best_point = rng.choice(candidates)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for triplet in itertools.combinations(block, t):
                tid = get_id(triplet)
                uncovered.discard(tid)
            
            # Update point weights: increase weights for points in this block
            # to discourage future blocks from using them too much
            for p in block:
                point_weights[p] *= weight_decay
        
        return blocks
>>>>>>> REPLACE

