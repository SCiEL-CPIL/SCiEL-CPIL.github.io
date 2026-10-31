import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t_subs = len(all_t_subs)
    
    masks = [0] * n_t_subs
    for i, sub in enumerate(all_t_subs):
        m = 0
        for p in sub:
            m |= (1 << p)
        masks[i] = m
            
    best_blocks = None
    best_count = float('inf')
    
    while time.time() < deadline:
        uncovered_set = set(range(n_t_subs))
        blocks = []
        
        while uncovered_set:
            target_idx = random.choice(list(uncovered_set))
            target_sub = all_t_subs[target_idx]
            target_mask = masks[target_idx]
            target_points = set(target_sub)
            available_points = [p for p in range(v) if p not in target_points]
            ext_size = k - t
            
            # Build frequency map of extra masks
            # extra = mask(U) & ~target_mask
            extra_counts = {}
            for u_idx in uncovered_set:
                if u_idx == target_idx:
                    continue
                u_mask = masks[u_idx]
                extra = u_mask & ~target_mask
                # If extra has bits outside available_points? 
                # No, extra is subset of U, and U is subset of v.
                # But we only care if extra can be covered by ext.
                # ext is subset of available_points.
                # So if extra has bits not in available_points, it can never be covered.
                # So we can skip those.
                avail_mask = 0
                for p in available_points:
                    avail_mask |= (1 << p)
                if (extra & ~avail_mask) != 0:
                    continue
                extra_counts[extra] = extra_counts.get(extra, 0) + 1
                
            best_gain = 1 # target
            best_ext = None
            
            for ext in itertools.combinations(available_points, ext_size):
                ext_mask = 0
                for p in ext:
                    ext_mask |= (1 << p)
                
                gain = 1
                for extra, count in extra_counts.items():
                    if (extra & ~ext_mask) == 0:
                        gain += count
                        
                if gain > best_gain:
                    best_gain = gain
                    best_ext = ext
                    
            block_points = list(target_sub) + list(best_ext)
            block_points.sort()
            blocks.append(block_points)
            
            block_mask = 0
            for p in block_points:
                block_mask |= (1 << p)
                
            new_uncovered = set()
            for u_idx in uncovered_set:
                if (masks[u_idx] & ~block_mask) == 0:
                    pass
                else:
                    new_uncovered.add(u_idx)
            uncovered_set = new_uncovered
            
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        if time.time() > deadline - 5:
            break
            
    if best_blocks is None:
        return []
        
    return best_blocks
