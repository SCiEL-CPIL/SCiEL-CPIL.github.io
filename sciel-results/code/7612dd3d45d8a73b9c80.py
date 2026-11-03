import itertools
import random
import time
import sys
import math
from collections import defaultdict
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 145
    
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    def iter_set_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def greedy_cover(rng):
        uncovered_mask = (1 << n_tsubs) - 1
        blocks = []
        order = list(range(n_tsubs))
        rng.shuffle(order)
        
        while uncovered_mask:
            if time.time() > deadline - 20:
                return None
            
            seed_idx = -1
            for idx in order:
                if (uncovered_mask >> idx) & 1:
                    seed_idx = idx
                    break
            if seed_idx == -1:
                break
                
            seed = all_tsubs[seed_idx]
            block = list(seed)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_point = -1
                tie_points = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None:
                            if (uncovered_mask >> idx) & 1:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        tie_points = [x]
                    elif gain == best_gain and gain > -1:
                        tie_points.append(x)
                        
                if best_point >= 0:
                    if len(tie_points) > 1:
                        best_point = rng.choice(tie_points)
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_point = rng.choice(remaining)
                        block.append(best_point)
                        block_set.add(best_point)
                    else:
                        break
            
            block.sort()
            blocks.append(block)
            bi = block_to_idx[tuple(block)]
            block_mask = block_masks[bi]
            uncovered_mask &= ~block_mask
            
        if uncovered_mask:
            return None
        return blocks

    def remove_redundant(blocks, deadline):
        if not blocks:
            return blocks
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs
        for bi in block_indices:
            mask = block_masks[bi]
            for idx in iter_set_bits(mask):
                cover_count[idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                bi = block_indices[i]
                mask = block_masks[bi]
                redundant = True
                for idx in iter_set_bits(mask):
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in iter_set_bits(mask):
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_indices.pop(i)
                    changed = True
                    break
        return blocks

    def local_search_sa(blocks, deadline):
        block_indices = [block_to_idx[tuple(b)] for b in blocks]
        cover_count = [0] * n_tsubs
        for bi in block_indices:
            mask = block_masks[bi]
            for idx in iter_set_bits(mask):
                cover_count[idx] += 1
                
        uncovered_count = sum(1 for c in cover_count if c == 0)
        best_blocks = [list(b) for b in blocks]
        best_count = len(blocks)
        
        T = 10.0
        cooling = 0.999
        rng = random.Random(42)
        steps = 0
        max_steps = 1000000
        no_improve_steps = 0
        max_no_improve = 10000
        
        while time.time() < deadline - 5 and steps < max_steps and no_improve_steps < max_no_improve:
            steps += 1
            if not block_indices:
                break
                
            i = rng.randint(0, len(block_indices) - 1)
            bi = block_indices[i]
            block = blocks[i]
            
            p_idx = rng.randint(0, k - 1)
            p = block[p_idx]
            
            q = rng.randint(0, v - 1)
            if q in block:
                continue
                
            new_block = [x if x != p else q for x in block]
            new_block.sort()
            new_bi = block_to_idx.get(tuple(new_block))
            if new_bi is None or new_bi == bi:
                continue
                
            old_mask = block_masks[bi]
            new_mask = block_masks[new_bi]
            
            lost = old_mask & ~new_mask
            gained = new_mask & ~old_mask
            
            delta_uncovered = 0
            for idx in iter_set_bits(lost):
                if cover_count[idx] == 1:
                    delta_uncovered += 1
            for idx in iter_set_bits(gained):
                if cover_count[idx] == 0:
                    delta_uncovered -= 1
                    
            accepted = False
            if delta_uncovered <= 0:
                accepted = True
            else:
                if rng.random() < math.exp(-delta_uncovered / T):
                    accepted = True
                    
            if accepted:
                for idx in iter_set_bits(lost):
                    cover_count[idx] -= 1
                for idx in iter_set_bits(gained):
                    cover_count[idx] += 1
                    
                blocks[i] = new_block
                block_indices[i] = new_bi
                uncovered_count += delta_uncovered
                
                if uncovered_count == 0:
                    removed_any = True
                    while removed_any and time.time() < deadline - 5:
                        removed_any = False
                        if len(block_indices) == 0:
                            break
                        j = rng.randint(0, len(block_indices) - 1)
                        bj = block_indices[j]
                        mask_j = block_masks[bj]
                        redundant = True
                        for idx in iter_set_bits(mask_j):
                            if cover_count[idx] == 1:
                                redundant = False
                                break
                        if redundant:
                            for idx in iter_set_bits(mask_j):
                                cover_count[idx] -= 1
                            blocks.pop(j)
                            block_indices.pop(j)
                            removed_any = True
                            no_improve_steps = 0
                            if len(blocks) < best_count:
                                best_count = len(blocks)
                                best_blocks = [list(b) for b in blocks]
                    if not removed_any:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
                
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        return best_blocks

    rng = random.Random(42)
    best_blocks = None
    best_count = float('inf')
    
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        init_blocks = greedy_cover(rng)
        if init_blocks:
            reduced = remove_redundant(init_blocks, deadline)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = [list(b) for b in reduced]
                
    if best_blocks is None:
        return []
        
    if time.time() < deadline - 10:
        sa_blocks = local_search_sa(best_blocks, deadline)
        if sa_blocks and len(sa_blocks) < best_count:
            best_blocks = sa_blocks
            best_count = len(sa_blocks)
            
    best_blocks = remove_redundant(best_blocks, deadline)
    
    return best_blocks
