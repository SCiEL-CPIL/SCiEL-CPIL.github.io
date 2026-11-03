    def run_greedy_lazy(seed_val=0):
        if seed_val:
            random.seed(seed_val)
            
        uncovered = all_mask
        selected = []
        used = [False] * n_blocks
        
        # Heap entries: (-gain, random_key, block_idx)
        # random_key helps break ties randomly
        heap = []
        for i in range(n_blocks):
            g = (masks[i] & uncovered).bit_count()
            if g > 0:
                heapq.heappush(heap, (-g, random.random(), i))
                
        while uncovered:
            best_gain = -1
            best_idx = -1
            
            while heap:
                neg_g, _, i = heapq.heappop(heap)
                if used[i]:
                    continue
                
                # Recalculate actual gain
                real_gain = (masks[i] & uncovered).bit_count()
                
                if real_gain != -neg_g:
                    # Stale entry. If real_gain > 0, push updated entry
                    if real_gain > 0:
                        heapq.heappush(heap, (-real_gain, random.random(), i))
                    continue
                
                # Valid entry
                best_gain = real_gain
                best_idx = i
                break
                
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            used[best_idx] = True
            uncovered &= ~masks[best_idx]
            
        return selected
