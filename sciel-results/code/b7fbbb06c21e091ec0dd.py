<<<<<<< SEARCH
    while time.time() < deadline - 10:
        # 1. Greedy (fast)
        initial_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples)
        
        if not initial_blocks:
            seed += 1
            continue
        
        # 2. Local Search with shorter sub-deadline for more restarts
        remaining = deadline - time.time()
        # Shorter local search: 8s max, but at least 2s if we have time
        ls_time = min(8.0, max(2.0, remaining - 15.0))
        ls_deadline = time.time() + ls_time
        
        if ls_deadline > deadline - 12.0:
            ls_deadline = deadline - 12.0
        
        if ls_deadline <= time.time() + 1.0:
            break
            
        blocks = _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, ls_deadline, overlap_cache)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            print(f"New best (LS): {best_len} blocks at t={time.time()-start_time:.1f}s")
        
        # 3. WalkSAT refinement on the best solution found so far
        ws_remaining = deadline - time.time()
        if ws_remaining > 3.0:
            ws_deadline = time.time() + min(5.0, ws_remaining - 2.0)
            ws_init = best_blocks if best_blocks else blocks
            ws_assignment, ws_size = _walksat(v, k, t, ws_init, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed + 1000, ws_deadline)
            ws_blocks = [i for i in range(n_blocks) if ws_assignment[i]]
            
            if len(ws_blocks) < best_len:
                best_len = len(ws_blocks)
                best_blocks = ws_blocks
                print(f"New best (WS): {best_len} blocks at t={time.time()-start_time:.1f}s")
        
        seed += 1
=======
    while time.time() < deadline - 10:
        # Alternate between standard and weighted greedy for diversity
        if seed % 3 == 0:
            initial_blocks = _greedy_construct(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples)
        elif seed % 3 == 1:
            initial_blocks = _greedy_construct_weighted(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples, weight=0.3)
        else:
            initial_blocks = _greedy_construct_weighted(v, k, t, all_blocks, block_masks, full_mask, blocks_containing, n_blocks, seed, block_triples, weight=0.5)
        
        if not initial_blocks:
            seed += 1
            continue
        
        # 2. Local Search with shorter sub-deadline for more restarts
        remaining = deadline - time.time()
        # Shorter local search: 10s max, but at least 2s if we have time
        ls_time = min(10.0, max(2.0, remaining - 15.0))
        ls_deadline = time.time() + ls_time
        
        if ls_deadline > deadline - 12.0:
            ls_deadline = deadline - 12.0
        
        if ls_deadline <= time.time() + 1.0:
            break
            
        blocks = _local_search(v, k, t, initial_blocks, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed, ls_deadline, overlap_cache)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            print(f"New best (LS): {best_len} blocks at t={time.time()-start_time:.1f}s")
        
        # 3. WalkSAT refinement on the best solution found so far
        ws_remaining = deadline - time.time()
        if ws_remaining > 3.0:
            ws_deadline = time.time() + min(5.0, ws_remaining - 2.0)
            ws_init = best_blocks if best_blocks else blocks
            ws_assignment, ws_size = _walksat(v, k, t, ws_init, block_masks, full_mask, blocks_containing, n_blocks, n_triples, block_triples, seed + 1000, ws_deadline)
            ws_blocks = [i for i in range(n_blocks) if ws_assignment[i]]
            
            if len(ws_blocks) < best_len:
                best_len = len(ws_blocks)
                best_blocks = ws_blocks
                print(f"New best (WS): {best_len} blocks at t={time.time()-start_time:.1f}s")
        
        seed += 1
>>>>>>> REPLACE
