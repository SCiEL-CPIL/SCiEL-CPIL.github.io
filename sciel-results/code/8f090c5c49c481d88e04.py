    def greedy_cover(randomize_ties=False):
        uncovered = (1 << num_t_subsets) - 1
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_gain = -1
            candidates = []
            
            for i, mask in enumerate(block_masks):
                gain = (uncovered & mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
                    
            if not candidates:
                break
                
            if randomize_ties and len(candidates) > 1:
                best_block_idx = random.choice(candidates)
            else:
                best_block_idx = candidates[0]
                
            blocks.append(all_blocks[best_block_idx])
            uncovered &= ~block_masks[best_block_idx]
            
        return blocks
