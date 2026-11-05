import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0

    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # block_tsets[bi] = list of tset indices covered by block bi
    block_tsets = []
    for b in all_blocks:
        ts_indices = []
        for ts in itertools.combinations(b, t):
            ts_indices.append(tset_index[ts])
        block_tsets.append(ts_indices)
        
    # tset_to_blocks[ti] = list of block indices covering tset ti
    tset_to_blocks = [[] for _ in range(n_tsub)]
    for bi, ts_list in enumerate(block_tsets):
        for ti in ts_list:
            tset_to_blocks[ti].append(bi)

    # 2. Greedy Set Cover
    uncovered = np.ones(n_tsub, dtype=bool)
    gain = np.zeros(n_blocks, dtype=np.int32)
    for bi in range(n_blocks):
        gain[bi] = len(block_tsets[bi])
        
    selected = []
    remaining = n_tsub
    
    while remaining > 0:
        if time.time() > deadline - 30.0:
            break
        bi = int(np.argmax(gain))
        if gain[bi] == 0:
            break
        selected.append(bi)
        gain[bi] = 0
        
        for ti in block_tsets[bi]:
            if uncovered[ti]:
                uncovered[ti] = False
                remaining -= 1
                for bj in tset_to_blocks[ti]:
                    if gain[bj] > 0:
                        gain[bj] -= 1
                    
    # Helper: Compute coverage counts
    def compute_cov(sel):
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            for ti in block_tsets[bi]:
                cov[ti] += 1
        return cov

    # Helper: Reduce solution (remove redundant blocks)
    def reduce_solution(sel):
        if not sel: return []
        cov = compute_cov(sel)
        
        new_sel = []
        for bi in sel:
            essential = False
            for ti in block_tsets[bi]:
                if cov[ti] == 1:
                    essential = True
                    break
            if not essential:
                for ti in block_tsets[bi]:
                    cov[ti] -= 1
            else:
                new_sel.append(bi)
                
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel

    selected = reduce_solution(selected)
    best_size = len(selected)
    
    # 3. Local Search: Add-Swap to reduce size
    # We look for b_new that covers unique t-sets of >= 2 blocks in selected.
    
    def find_improvement(sel):
        cov = compute_cov(sel)
        sel_set = set(sel)
        
        # Identify blocks with small unique sets
        # U_b = {t in block_tsets[b] | cov[t] == 1}
        small_unique_blocks = []
        for bi in sel:
            unique_ts = [ti for ti in block_tsets[bi] if cov[ti] == 1]
            if len(unique_ts) <= 3: # Threshold for "small"
                small_unique_blocks.append((bi, unique_ts))
        
        # Try to find b_new covering unique ts of 2 or more such blocks
        # To avoid O(N^2) over small_unique_blocks, we can iterate over pairs
        # But we need to find b_new first.
        # Alternative: Iterate over candidate b_new not in sel.
        # For each b_new, check how many blocks in sel it "covers" (makes redundant).
        
        # Optimization: Only consider b_new that intersects with unique ts of some blocks.
        # Let's just iterate all b_new not in sel, but break early if time is low.
        
        best_move = None
        best_gain = 0
        
        # Limit search to speed up
        # random.shuffle(all_blocks) ? No, we need to check all potentially.
        # But 75k blocks is a lot.
        
        # Heuristic: Prioritize b_new that covers many t-sets with cov==2?
        # Or just random sampling if time is tight.
        
        # Let's try iterating all, but with a fast check.
        # For a b_new, it makes block bi redundant if U_bi subset of block_tsets[b_new].
        # U_bi is small.
        
        # Group small_unique_blocks by their unique t-sets?
        
        # Simpler:
        # For each bi in small_unique_blocks:
        #   For each bj in small_unique_blocks where bj > bi:
        #     Target = U_bi union U_bj
        #     Find b_new containing all t in Target.
        #     If found and b_new not in sel:
        #        Check if b_new makes bi and bj redundant. (It does by construction if it covers U_bi and U_bj)
        #        Check if b_new makes any OTHER blocks redundant.
        #        Gain = number of blocks made redundant - 1.
        #        If Gain > best_gain: store.
        
        # This is O(|S|^2 * cost_find_block).
        # |S| is ~300. 300^2 = 90k pairs.
        # cost_find_block: Intersection of tset_to_blocks for t in Target.
        # Target size <= 6.
        # This is feasible.
        
        for i in range(len(small_unique_blocks)):
            bi, u_i = small_unique_blocks[i]
            for j in range(i+1, len(small_unique_blocks)):
                bj, u_j = small_unique_blocks[j]
                target_ts = set(u_i) | set(u_j)
                
                # Find blocks covering all t in target_ts
                # Start with blocks covering first t
                if not target_ts: continue
                
                first_t = next(iter(target_ts))
                candidates = set(tset_to_blocks[first_t])
                
                for t in list(target_ts)[1:]:
                    candidates.intersection_update(tset_to_blocks[t])
                    if not candidates: break
                
                if not candidates: continue
                
                # Filter out selected blocks
                candidates.difference_update(sel_set)
                
                if not candidates: continue
                
                # Pick one candidate (e.g., the one covering most other t-sets? or just first)
                # Let's just pick the first one to save time, or check a few.
                b_new = next(iter(candidates))
                
                # Verify: Does b_new make bi and bj redundant?
                # It covers U_bi and U_bj.
                # We need to check if bi has any OTHER unique t-sets? No, U_bi is ALL unique t-sets of bi.
                # So yes, bi becomes redundant. Same for bj.
                
                # Check if b_new makes any OTHER blocks redundant.
                # A block bk becomes redundant if U_bk subset of block_tsets[b_new].
                # We only checked bi, bj.
                # Let's count total redundant blocks if we add b_new.
                
                # This check is expensive if done for all sel.
                # But we can just assume gain is at least 2 (bi, bj) - 1 = 1.
                # If we find a move with gain 1, we take it.
                # To be safer, we can check if gain > 1.
                
                # For now, accept if we found such a pair.
                # We will remove bi and bj, add b_new.
                return (bi, bj, b_new)
                
        return None

    # Main Local Search Loop
    while time.time() < deadline - 5.0:
        # Reduce first
        selected = reduce_solution(selected)
        
        if len(selected) < best_size:
            best_size = len(selected)
            
        # Try to improve
        move = find_improvement(selected)
        
        if move:
            bi, bj, b_new = move
            # Apply move
            if bi in selected: selected.remove(bi)
            if bj in selected: selected.remove(bj)
            selected.append(b_new)
        else:
            # No improvement found with current heuristic.
            # Try random restart? Or break?
            # To avoid infinite loop, we can break or try a random swap.
            # Let's break for now, as the heuristic is strong.
            break
            
    # Final Reduce
    selected = reduce_solution(selected)
    
    return [list(all_blocks[bi]) for bi in selected]
