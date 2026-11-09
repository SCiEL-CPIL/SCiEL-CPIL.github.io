    def reduce_blocks(blocks, seed=None):
        if not blocks:
            return blocks
        if seed is not None:
            random.seed(seed)
            
        reduced = list(blocks)
        cover_count = np.zeros(N_t, dtype=np.int32)
        
        block_t_subsets = []
        for b in reduced:
            subs = [subset_to_id[comb] for comb in itertools.combinations(b, t)]
            block_t_subsets.append(subs)
            for s in subs:
                cover_count[s] += 1
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            indices = list(range(len(reduced)))
            random.shuffle(indices)
            
            to_remove = []
            for i in indices:
                if i >= len(reduced):
                    continue
                # Skip if already marked for removal? 
                # We need to be careful with indices if we remove.
                # Better to just check and mark, then remove all at end of pass.
                # But if we remove one, counts change.
                # Standard approach: try to remove, if fail, restore.
                # Or: collect candidates, remove all, then check if valid.
                # If valid, keep removed. If invalid, restore some.
                
                # Let's stick to the single removal but optimize the check.
                subs = block_t_subsets[i]
                
                # Decrement
                for s in subs:
                    cover_count[s] -= 1
                
                # Check min count in subs
                # Using numpy might be faster for large subs, but subs size is C(8,5)=56.
                # Python loop is fine.
                redundant = True
                for s in subs:
                    if cover_count[s] == 0:
                        redundant = False
                        break
                
                if redundant:
                    # Remove
                    reduced.pop(i)
                    block_t_subsets.pop(i)
                    changed = True
                    break # Restart loop to reshuffle indices? Or continue?
                    # Breaking and restarting is safer for correctness with indices.
                else:
                    for s in subs:
                        cover_count[s] += 1
            # End while
        return reduced
