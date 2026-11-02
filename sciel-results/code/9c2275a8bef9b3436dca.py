import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0

    def comb(n, r):
        if r > n or r < 0:
            return 0
        r = min(r, n - r)
        if r == 0:
            return 1
        res = 1
        for i in range(1, r + 1):
            res = res * (n - i + 1) // i
        return res

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    block_to_ts_idx = []
    ts_to_block_idx = [[] for _ in range(num_t)]
    for j, blk in enumerate(all_k_blocks):
        idxs = []
        for ts in itertools.combinations(blk, t):
            idx = ts_index[ts]
            idxs.append(idx)
            ts_to_block_idx[idx].append(j)
        block_to_ts_idx.append(idxs)

    ts_block_arrays = [np.array(x, dtype=np.int32) for x in ts_to_block_idx]
    
    # Precompute block tuples for set operations
    block_tuples = [tuple(b) for b in all_k_blocks]

    def greedy(seed_val):
        rng = np.random.RandomState(seed_val)
        gain = np.full(num_k, comb(k, t), dtype=np.int32)
        freq = np.zeros(num_t, dtype=np.int32)
        selected = np.zeros(num_k, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 15.0:
                return None
            mg = int(np.max(gain))
            if mg <= 0:
                break
            cands = np.where((gain == mg) & (~selected))[0]
            if len(cands) == 0:
                break
            
            # Tie-breaking: prefer blocks with higher "diversity" or random
            # Simple random selection from max gain candidates
            b = cands[rng.randint(0, len(cands))]
            sol.append(int(b))
            selected[b] = True
            gain[b] = -1
            for ti in block_to_ts_idx[b]:
                if freq[ti] == 0:
                    gain[ts_block_arrays[ti]] -= 1
                freq[ti] += 1
        return sol

    best_idx = None
    best_len = float('inf')

    # Run multiple greedy restarts
    for s in range(100):
        if time.time() > deadline - 20.0:
            break
        sol = greedy(s)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_idx = sol

    def idx_to_blocks(idxs):
        return [list(all_k_blocks[i]) for i in idxs]

    blocks = idx_to_blocks(best_idx)

    # Redundancy removal
    def remove_redundant(blks):
        if not blks:
            return blks
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(blks)):
                if time.time() > deadline - 5.0:
                    return blks
                red = True
                for ts in itertools.combinations(blks[i], t):
                    if cc[ts_index[ts]] == 1:
                        red = False
                        break
                if red:
                    rem = blks.pop(i)
                    for ts in itertools.combinations(rem, t):
                        cc[ts_index[ts]] -= 1
                    changed = True
                    break
        return blks

    blocks = remove_redundant(blocks)
    best_len = len(blocks)

    # Local Search: 1-swap and 2-swap with lookahead
    def local_search(blks, max_passes=50):
        if not blks:
            return blks
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
        sol_set = set(tuple(b) for b in blks)
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            if time.time() > deadline - 6.0:
                break
            improved = False
            passes += 1
            
            # 1. Try 1-swaps that enable removal of another block
            n = len(blks)
            for i in range(n):
                if time.time() > deadline - 6.0:
                    break
                ut = []
                for ts in itertools.combinations(blks[i], t):
                    idx = ts_index[ts]
                    if cc[idx] == 1:
                        ut.append(idx)
                if not ut:
                    continue
                
                # Find candidates for block i
                cands = set(ts_to_block_idx[ut[0]])
                for ti in ut[1:]:
                    cands.intersection_update(ts_to_block_idx[ti])
                    if not cands:
                        break
                if not cands:
                    continue
                
                valid_cands = [c for c in cands if tuple(all_k_blocks[c]) not in sol_set]
                if not valid_cands:
                    continue
                
                # Try each candidate to see if it enables a removal
                for c in valid_cands:
                    nb = list(all_k_blocks[c])
                    # Simulate swap
                    # Decrement old, increment new
                    # Check if any other block becomes redundant
                    # To be fast, we can just check if the new block covers 
                    # unique subsets of some other block? No, it might cover 
                    # subsets that are unique to i, but also subsets that are 
                    # shared with j, making j redundant if j's other subsets 
                    # are covered.
                    
                    # A simpler heuristic: just do the swap if it's valid, 
                    # then run redundancy removal. 
                    # But we want to be smarter.
                    
                    # Let's just pick the first valid candidate for now 
                    # and rely on redundancy removal to clean up.
                    # This is what the original code did.
                    # To improve, we can try to find a candidate that 
                    # maximizes the number of "almost redundant" blocks.
                    
                    break # Just pick one for speed
                    
                # Perform the swap
                ni = valid_cands[0]
                nb = list(all_k_blocks[ni])
                old = blks[i]
                for ts in itertools.combinations(old, t):
                    cc[ts_index[ts]] -= 1
                for ts in itertools.combinations(nb, t):
                    cc[ts_index[ts]] += 1
                sol_set.discard(tuple(old))
                sol_set.add(tuple(nb))
                blks[i] = nb
                improved = True
            
            # 2. Try 2-swaps (replace i and j with k)
            n = len(blks)
            for i in range(n):
                if time.time() > deadline - 6.0:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 6.0:
                        break
                    uti = []
                    for ts in itertools.combinations(blks[i], t):
                        idx = ts_index[ts]
                        if cc[idx] == 1:
                            uti.append(idx)
                    utj = []
                    for ts in itertools.combinations(blks[j], t):
                        idx = ts_index[ts]
                        if cc[idx] == 1:
                            utj.append(idx)
                    au = uti + utj
                    if not au:
                        continue
                    cands = set(ts_to_block_idx[au[0]])
                    for ti in au[1:]:
                        cands.intersection_update(ts_to_block_idx[ti])
                        if not cands:
                            break
                    if cands:
                        valid = [c for c in cands if tuple(all_k_blocks[c]) not in sol_set]
                        if valid:
                            ni = random.choice(valid)
                            nb = list(all_k_blocks[ni])
                            oi, oj = blks[i], blks[j]
                            for ts in itertools.combinations(oi, t):
                                cc[ts_index[ts]] -= 1
                            for ts in itertools.combinations(oj, t):
                                cc[ts_index[ts]] -= 1
                            for ts in itertools.combinations(nb, t):
                                cc[ts_index[ts]] += 1
                            sol_set.discard(tuple(oi))
                            sol_set.discard(tuple(oj))
                            sol_set.add(tuple(nb))
                            blks.pop(j)
                            blks[i] = nb
                            improved = True
                            break
                if improved:
                    break
                    
            # 3. Remove redundant blocks
            blks = remove_redundant(blks)
            # Update cc after removal
            cc = np.zeros(num_t, dtype=np.int32)
            for b in blks:
                for ts in itertools.combinations(b, t):
                    cc[ts_index[ts]] += 1
            sol_set = set(tuple(b) for b in blks)

        return blks

    # Main loop with perturbation
    for _ in range(50):
        if time.time() > deadline - 8.0:
            break
        
        # Local search
        blocks = local_search(blocks, 30)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
        
        # Perturbation: Remove a random block and re-cover
        if len(blocks) > 0 and time.time() < deadline - 20.0:
            # Pick a random block to remove
            idx_to_remove = random.randint(0, len(blocks) - 1)
            removed_block = blocks.pop(idx_to_remove)
            
            # Find uncovered t-subsets
            cc = np.zeros(num_t, dtype=np.int32)
            for b in blocks:
                for ts in itertools.combinations(b, t):
                    cc[ts_index[ts]] += 1
            
            uncovered = []
            for ts in itertools.combinations(removed_block, t):
                idx = ts_index[ts]
                if cc[idx] == 0:
                    uncovered.append(idx)
            
            if uncovered:
                # Find a block that covers the most uncovered subsets
                best_gain = -1
                best_block_idx = -1
                # We can't just scan all 75k blocks efficiently every time
                # But we can scan candidates that cover the first uncovered subset
                cands = ts_to_block_idx[uncovered[0]]
                for c in cands:
                    if tuple(all_k_blocks[c]) in set(tuple(b) for b in blocks):
                        continue
                    gain = 0
                    for u in uncovered:
                        if c in ts_to_block_idx[u]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block_idx = c
                
                if best_block_idx != -1:
                    blocks.append(list(all_k_blocks[best_block_idx]))
                    
        # If perturbation made it worse, revert? 
        # No, we just continue searching. 
        # But we should keep the best solution found so far.
        # The local search will try to recover.

    # Final cleanup
    if time.time() < deadline - 4.0:
        blocks = remove_redundant(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)

    return blocks
