import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0
    
    def time_left():
        return deadline - time.time()
    
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
    
    # Convert block_tsets to numpy arrays for faster ops
    block_tsets_np = [np.array(ts, dtype=np.int32) for ts in block_tsets]
    
    # 2. Greedy Set Cover with optional randomization
    def run_greedy(randomize_ties=False, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        uncovered = np.ones(n_tsub, dtype=bool)
        gain = np.array([len(ts) for ts in block_tsets], dtype=np.int32)
        
        selected = []
        remaining = n_tsub
        
        while remaining > 0:
            if time_left() < 30.0:
                break
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            
            if randomize_ties:
                candidates = np.where(gain == max_gain)[0]
                bi = int(candidates[rng.randint(0, len(candidates) - 1)])
            else:
                bi = int(np.argmax(gain))
            
            selected.append(bi)
            gain[bi] = 0
            
            for ti in block_tsets[bi]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    remaining -= 1
                    for bj in tset_to_blocks[ti]:
                        gain[bj] -= 1
        
        return selected
    
    # 3. Fast reduction using numpy
    def reduce_solution(sel):
        if not sel:
            return []
        
        if time_left() < 5.0:
            return sel
        
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in sel:
            cov[block_tsets_np[bi]] += 1
        
        new_sel = []
        for bi in sel:
            bts = block_tsets_np[bi]
            if np.any(cov[bts] == 1):
                new_sel.append(bi)
            else:
                cov[bts] -= 1
        
        if len(new_sel) < len(sel):
            return reduce_solution(new_sel)
        return new_sel
    
    # 4. Local search: repeated reduction + swap passes
    def local_search(selected, max_rounds=100):
        if not selected or time_left() < 10.0:
            return selected
        
        cov = np.zeros(n_tsub, dtype=np.int32)
        for bi in selected:
            cov[block_tsets_np[bi]] += 1
        
        best_selected = list(selected)
        best_size = len(selected)
        
        for rnd in range(max_rounds):
            if time_left() < 10.0:
                break
            
            # Phase 1: Reduction
            selected = reduce_solution(selected)
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)
                
            if not selected:
                break
            
            # Rebuild coverage after reduction
            cov = np.zeros(n_tsub, dtype=np.int32)
            for bi in selected:
                cov[block_tsets_np[bi]] += 1
            
            # Phase 2: Swap search - try multiple random swaps per round
            improved = False
            n_swaps_to_try = min(len(selected), 50)
            
            for _ in range(n_swaps_to_try):
                if time_left() < 10.0:
                    break
                
                if not selected:
                    break
                
                idx_remove = random.randint(0, len(selected) - 1)
                bi_remove = selected[idx_remove]
                
                # Find tsets that would become uncovered
                bts_remove = block_tsets_np[bi_remove]
                newly_uncovered = bts_remove[cov[bts_remove] == 1]
                
                if len(newly_uncovered) == 0:
                    # Block is redundant, just remove it
                    cov[bts_remove] -= 1
                    selected.pop(idx_remove)
                    improved = True
                    continue
                
                # Find candidate blocks that cover ALL newly_uncovered tsets
                # Use intersection of covering sets
                if len(newly_uncovered) == 0:
                    continue
                    
                # Start with blocks covering the first newly-uncovered tset
                cand_set = set(tset_to_blocks[int(newly_uncovered[0])])
                for ti in newly_uncovered[1:]:
                    cand_set &= set(tset_to_blocks[int(ti)])
                
                cand_set.discard(bi_remove)
                cand_set -= set(selected)
                
                if not cand_set:
                    continue
                
                # Score candidates: how many total tsets they cover (bonus for covering more)
                cand_list = list(cand_set)
                
                # Pick randomly from candidates (or by score)
                # Use a simple heuristic: prefer blocks that cover more total tsets
                if len(cand_list) <= 5:
                    bi_add = random.choice(cand_list)
                else:
                    # Score by total coverage
                    scores = []
                    for bj in cand_list:
                        s = len(block_tsets[bj])
                        scores.append((s, bj))
                    scores.sort(key=lambda x: -x[0])
                    # Take from top 3
                    top_n = min(3, len(scores))
                    bi_add = random.choice([s[1] for s in scores[:top_n]])
                
                # Perform swap
                bts_add = block_tsets_np[bi_add]
                
                # Tentatively apply
                cov[bts_remove] -= 1
                cov[bts_add] += 1
                
                # Check validity
                valid = np.all(cov[bts_remove] >= 0)
                # Also need to ensure no tset went to 0 that wasn't in newly_uncovered
                # Actually, we only need to check that all tsets are covered
                # The critical check: no tset has cov == 0
                # But checking all n_tsub is expensive. Instead, check that
                # the tsets affected are still covered.
                
                # More precise check: for tsets in bts_remove, cov must be >= 1
                # For tsets in bts_add, cov increased so no issue there
                # For tsets in intersection, net change is 0
                # The risk is tsets in bts_remove but not in bts_add going to 0
                
                valid = np.all(cov[bts_remove] >= 1)
                
                if not valid:
                    # Revert
                    cov[bts_add] -= 1
                    cov[bts_remove] += 1
                    continue
                
                # Apply swap
                selected[idx_remove] = bi_add
                improved = True
                
                # After a successful swap, do a quick reduction
                selected = reduce_solution(selected)
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
                
                # Rebuild coverage
                cov = np.zeros(n_tsub, dtype=np.int32)
                for bi in selected:
                    cov[block_tsets_np[bi]] += 1
                
                if not selected:
                    break
            
            if not improved:
                break
        
        return best_selected
    
    # 5. Main: run multiple greedy starts + local search
    best_overall = None
    best_size_overall = float('inf')
    
    # Try deterministic greedy first
    if time_left() > 20.0:
        sel = run_greedy(randomize_ties=False)
        sel = reduce_solution(sel)
        sel = local_search(sel)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel
    
    # Try randomized greedy restarts
    n_restarts = 0
    while time_left() > 15.0 and n_restarts < 10:
        n_restarts += 1
        seed = n_restarts * 7919
        sel = run_greedy(randomize_ties=True, seed=seed)
        if not sel:
            continue
        sel = reduce_solution(sel)
        sel = local_search(sel)
        if sel and len(sel) < best_size_overall:
            best_size_overall = len(sel)
            best_overall = sel
    
    if best_overall is None:
        # Fallback: just use greedy without local search
        best_overall = run_greedy(randomize_ties=False)
        best_overall = reduce_solution(best_overall)
    
    # Final reduction
    best_overall = reduce_solution(best_overall)
    
    return [list(all_blocks[bi]) for bi in best_overall]
