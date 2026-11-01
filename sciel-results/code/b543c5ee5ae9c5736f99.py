<<<<<<< SEARCH
    def local_search_swap(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
                
        # Build set of block indices currently in solution for O(1) lookup
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline - 1.0:
                return blocks, improved
            
            unique_ts = []
            for ts in itertools.combinations(blocks[i], t):
                idx = ts_index_map[ts]
                if cover_count[idx] == 1:
                    unique_ts.append(idx)
                    
            if not unique_ts:
                continue
            
            # Find candidate blocks that cover ALL unique_ts
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter out blocks already in solution
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            if not valid_candidates:
                continue
                
            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])
            
            old_block = blocks[i]
            old_block_idx = get_block_idx(old_block)
            
            blocks[i] = new_block
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)
            
            for ts in itertools.combinations(old_block, t):
                cover_count[ts_index_map[ts]] -= 1
            for ts in itertools.combinations(new_block, t):
                cover_count[ts_index_map[ts]] += 1
            
            improved = True
            
        return blocks, improved
=======
    def local_search_swap(blocks):
        if not blocks:
            return blocks, False
            
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        # Build cover count using precomputed t-subsets
        for b in blocks:
            b_idx = get_block_idx(b)
            if b_idx >= 0:
                for ts in block_t_subsets[b_idx]:
                    cover_count[ts_index_map[ts]] += 1
                
        # Build set of block indices currently in solution for O(1) lookup
        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)
                
        improved = False
        n = len(blocks)
        
        # Try multiple passes for better convergence
        for pass_num in range(3):
            if time.time() > deadline - 1.0:
                return blocks, improved
                
            for i in range(n):
                if time.time() > deadline - 1.0:
                    return blocks, improved
                    
                if i >= len(blocks):
                    break
                    
                block_idx_i = get_block_idx(blocks[i])
                if block_idx_i < 0:
                    continue
                    
                unique_ts = []
                for ts in block_t_subsets[block_idx_i]:
                    idx = ts_index_map[ts]
                    if cover_count[idx] == 1:
                        unique_ts.append(idx)
                        
                if not unique_ts:
                    continue
                
                # Find candidate blocks that cover ALL unique_ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Filter out blocks already in solution
                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                if not valid_candidates:
                    continue
                    
                # Prefer blocks that cover more already-covered t-subsets (for future flexibility)
                # Score candidates by how many unique t-subsets they cover beyond the required ones
                best_candidate = None
                best_score = -1
                for c in valid_candidates[:50]:  # Limit to 50 for speed
                    c_ts = block_to_ts[c]
                    # Count how many of the unique_ts this block covers
                    covered = sum(1 for ts in unique_ts if ts in set(c_ts))
                    if covered > best_score:
                        best_score = covered
                        best_candidate = c
                        
                if best_candidate is None:
                    best_candidate = valid_candidates[0]
                    
                new_block_idx = best_candidate
                new_block = list(all_k_blocks[new_block_idx])
                
                old_block = blocks[i]
                old_block_idx = get_block_idx(old_block)
                
                blocks[i] = new_block
                sol_block_indices.discard(old_block_idx)
                sol_block_indices.add(new_block_idx)
                
                for ts in block_t_subsets[old_block_idx]:
                    cover_count[ts_index_map[ts]] -= 1
                for ts in block_t_subsets[new_block_idx]:
                    cover_count[ts_index_map[ts]] += 1
                
                improved = True
                
            if not improved:
                break
                
        return blocks, improved
>>>>>>> REPLACE

