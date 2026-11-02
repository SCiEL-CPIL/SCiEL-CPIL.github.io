import itertools
import random
import time

_DEADLINE = 145.0
_start_time = None

def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def construct(v, k, t):
    global _start_time
    _start_time = time.time()
    
    all_points = range(v)
    all_ts = list(itertools.combinations(all_points, t))
    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    n_ts = len(all_ts)
    
    all_blocks = list(itertools.combinations(all_points, k))
    block_index = {blk: i for i, blk in enumerate(all_blocks)}
    n_blocks = len(all_blocks)
    
    block_covered_ts = [None] * n_blocks
    for i, blk in enumerate(all_blocks):
        subs = [ts_index[ts] for ts in itertools.combinations(blk, t)]
        block_covered_ts[i] = set(subs)
        
    ts_to_blocks = [[] for _ in range(n_ts)]
    for bid, blk in enumerate(all_blocks):
        for s_idx in block_covered_ts[bid]:
            ts_to_blocks[s_idx].append(bid)

    def greedy_build(rng):
        coverage = [0] * n_ts
        unc_set = set(range(n_ts))
        current_blocks = []
        
        while unc_set and _remaining() > 5.0:
            anchor_ts_idx = rng.choice(list(unc_set))
            best_block_id = -1
            best_gain = -1
            
            for bid in ts_to_blocks[anchor_ts_idx]:
                gain = 0
                for s_idx in block_covered_ts[bid]:
                    if s_idx in unc_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_id = bid
            
            if best_block_id == -1:
                break
                
            current_blocks.append(all_blocks[best_block_id])
            for s_idx in block_covered_ts[best_block_id]:
                coverage[s_idx] += 1
                if coverage[s_idx] == 1:
                    unc_set.discard(s_idx)
            
        return current_blocks

    def local_search(blocks, rng):
        if not blocks:
            return blocks
            
        coverage = [0] * n_ts
        block_ids = [block_index[blk] for blk in blocks]
        for bid in block_ids:
            for s_idx in block_covered_ts[bid]:
                coverage[s_idx] += 1
        
        # Deletion
        improved = True
        while improved and _remaining() > 2.0:
            improved = False
            new_ids = []
            for bid in block_ids:
                redundant = True
                for s_idx in block_covered_ts[bid]:
                    if coverage[s_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for s_idx in block_covered_ts[bid]:
                        coverage[s_idx] -= 1
                    improved = True
                else:
                    new_ids.append(bid)
            block_ids = new_ids
            
        # Swap
        max_passes = 50
        for _ in range(max_passes):
            if _remaining() < 2.0:
                break
            improved = False
            rng.shuffle(block_ids)
            for idx, bid in enumerate(block_ids):
                if _remaining() < 1.0:
                    break
                blk = all_blocks[bid]
                blk_set = set(blk)
                for x in blk:
                    candidates = [y for y in range(v) if y not in blk_set]
                    rng.shuffle(candidates)
                    for y in candidates:
                        new_blk = tuple(sorted((blk_set - {x}) | {y}))
                        new_bid = block_index[new_blk]
                        
                        old_subs = block_covered_ts[bid]
                        new_subs = block_covered_ts[new_bid]
                        
                        gain = 0
                        for s_idx in new_subs:
                            if coverage[s_idx] == 0:
                                gain += 1
                                if gain > 0: # Just need to know if > 0? No, need exact for comparison with loss
                                    pass
                        # Actually, we need to compare gain vs loss.
                        # If gain == 0, skip.
                        if gain == 0:
                            continue
                            
                        loss = 0
                        for s_idx in old_subs:
                            if s_idx not in new_subs and coverage[s_idx] == 1:
                                loss += 1
                                
                        if gain > loss:
                            for s_idx in old_subs:
                                coverage[s_idx] -= 1
                            for s_idx in new_subs:
                                coverage[s_idx] += 1
                            block_ids[idx] = new_bid
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
                
        return [all_blocks[bid] for bid in block_ids]

    best_blocks = None
    best_count = float('inf')
    
    for i in range(30):
        if _remaining() < 5.0:
            break
        rng = random.Random(i)
        blocks = greedy_build(rng)
        blocks = local_search(blocks, rng)
        
        if not blocks:
            continue
            
        coverage = [0] * n_ts
        for blk in blocks:
            bid = block_index[blk]
            for s_idx in block_covered_ts[bid]:
                coverage[s_idx] += 1
        if all(c > 0 for c in coverage):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks

    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = greedy_build(rng)
        best_blocks = local_search(best_blocks, rng)
        
    return best_blocks
