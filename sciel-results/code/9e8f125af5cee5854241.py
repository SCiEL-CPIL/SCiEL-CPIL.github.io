    def run_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered_vec = np.ones(n_t_subsets, dtype=np.uint8)
        used_blocks = np.zeros(n_blocks, dtype=bool)
        blocks_selected = []
        uncovered_count = n_t_subsets
        
        while uncovered_count > 0:
            if time.time() > deadline:
                return None # Timeout
            
            gains = A_T.dot(uncovered_vec)
            gains[used_blocks] = -1
            
            max_gain = np.max(gains)
            if max_gain < 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_block_idx = candidates[random.randint(0, len(candidates)-1)]
            
            blocks_selected.append(all_blocks[best_block_idx])
            used_blocks[best_block_idx] = True
            
            for ts_idx in block_to_t_subsets[best_block_idx]:
                if uncovered_vec[ts_idx] == 1:
                    uncovered_vec[ts_idx] = 0
                    uncovered_count -= 1
                    
        return blocks_selected
