<<<<<<< SEARCH
    def greedy_initial(seed=0):
        """Fast greedy to get a starting solution."""
        rng = random.Random(seed)
        uncovered = set(range(n_tsubsets))
        blocks_used = []
        
        # Precompute gain: for each block, how many currently uncovered t-subsets it covers
        # Start with all uncovered
        gain = [len(block_tsubs[i]) for i in range(n_blocks)]
        
        while uncovered:
            if time.time() > deadline - 2:
                break
            
            # Find block with max gain
            max_gain = max(gain)
            if max_gain == 0:
                break
            
            # Pick randomly among max-gain blocks
            candidates = [i for i in range(n_blocks) if gain[i] == max_gain]
            chosen = rng.choice(candidates)
            blocks_used.append(chosen)
            
            # Update uncovered and gains
            for ts_idx in block_tsubs[chosen]:
                if ts_idx in uncovered:
                    uncovered.discard(ts_idx)
                    # Decrement gain for all blocks covering this t-subset
                    for bi in tsub_blocks[ts_idx]:
                        if gain[bi] > 0:
                            gain[bi] -= 1
            
            gain[chosen] = 0
        
        return blocks_used
=======
    def greedy_initial(seed=0):
        """Fast greedy to get a starting solution using a max-heap."""
        import heapq
        rng = random.Random(seed)
        uncovered = set(range(n_tsubsets))
        blocks_used = []
        
        # Heap elements: (-gain, block_index)
        # We need to handle stale entries in the heap
        gain = [len(block_tsubs[i]) for i in range(n_blocks)]
        heap = [(-gain[i], i) for i in range(n_blocks)]
        heapq.heapify(heap)
        
        while uncovered and heap:
            if time.time() > deadline - 2:
                break
            
            # Get the block with max gain
            neg_gain, chosen = heapq.heappop(heap)
            current_gain = -neg_gain
            
            # Check if this entry is stale (gain changed since push)
            if gain[chosen] != current_gain:
                continue
            
            if current_gain == 0:
                break
            
            blocks_used.append(chosen)
            
            # Update uncovered and gains
            for ts_idx in block_tsubs[chosen]:
                if ts_idx in uncovered:
                    uncovered.discard(ts_idx)
                    # Decrement gain for all blocks covering this t-subset
                    for bi in tsub_blocks[ts_idx]:
                        if gain[bi] > 0:
                            gain[bi] -= 1
                            # Push updated gain to heap
                            heapq.heappush(heap, (-gain[bi], bi))
            
            # Mark chosen block as used (gain 0)
            gain[chosen] = 0
        
        return blocks_used
>>>>>>> REPLACE

