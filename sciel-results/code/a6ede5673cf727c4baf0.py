    def greedy_cover(random_state):
        rng = random.Random(random_state)
        uncovered = bytearray([1] * num_triples)
        num_uncovered = num_triples
        blocks = []
        points = list(range(v))
        
        while True:
            start_idx = uncovered.find(1)
            if start_idx == -1:
                break
                
            target_triple = triples[start_idx]
            block = list(target_triple)
            
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in points if x not in block]
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    # Optimization: Pre-compute pairs in block?
                    # Block size is small.
                    for i in range(len(block)):
                        u = block[i]
                        for j in range(i+1, len(block)):
                            v = block[j]
                            # Get idx
                            # Sort x, u, v
                            # We can do:
                            # a, b, c = sorted((x, u, v))
                            # idx = triple_to_idx[(a, b, c)]
                            # This creates a tuple and sorts.
                            
                            # Faster:
                            if x < u:
                                if x < v:
                                    # x < u, x < v
                                    if u < v:
                                        idx = triple_to_idx[(x, u, v)]
                                    else:
                                        idx = triple_to_idx[(x, v, u)]
                                else:
                                    # x < u, v <= x
                                    if v < u:
                                        idx = triple_to_idx[(v, x, u)]
                                    else:
                                        idx = triple_to_idx[(x, u, v)] # v==x? No, distinct. So v < x or v > x.
                                        # If v > x and x < u, then x < v < u or v > u?
                                        # We need sorted.
                                        # This manual sorting is error prone.
                                        # Let's use tuple(sorted((x,u,v)))
                                        pass
                            # Just use sorted()
                            idx = triple_to_idx[tuple(sorted((x, u, v)))]
                            
                            if uncovered[idx]:
                                gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    
                if best_point == -1:
                    break
                
                block.append(best_point)
                
                # Update uncovered
                for i in range(len(block)):
                    if block[i] == best_point:
                        continue
                    for j in range(i+1, len(block)):
                        if block[j] == best_point:
                            continue
                        u = block[i]
                        v = block[j]
                        idx = triple_to_idx[tuple(sorted((best_point, u, v)))]
                        if uncovered[idx]:
                            uncovered[idx] = 0
                            num_uncovered -= 1
                            
            blocks.append(tuple(sorted(block)))
            
        return blocks
