import time
import itertools
import numpy as np

_SEED = 20240612

def _rng(seed):
    return np.random.RandomState(seed)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    rng = _rng(_SEED)
    
    # Precompute all t-subsets
    all_subs = list(itertools.combinations(range(v), t))
    n_all = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    count = np.zeros(n_all, dtype=np.int32)
    blocks = []

    def update_count(block, delta):
        for s in itertools.combinations(block, t):
            count[sub_to_idx[s]] += delta

    def is_covered():
        return not np.any(count == 0)

    # Phase 1: Greedy Construction
    # Start with a few random blocks
    for _ in range(5):
        b = tuple(sorted(rng.choice(v, size=k, replace=False)))
        blocks.append(b)
        update_count(b, 1)
        
    # Greedy Add
    max_greedy_steps = 300
    for _ in range(max_greedy_steps):
        if time.time() > deadline - 15:
            break
            
        if is_covered():
            break
            
        uncovered_mask = count == 0
        uncovered_indices = np.where(uncovered_mask)[0]
        
        # Sample uncovered triples to guide construction
        sample_size = min(10, len(uncovered_indices))
        sample_idx = rng.choice(uncovered_indices, size=sample_size, replace=False)
        
        best_block = None
        best_gain = -1
        
        for idx in sample_idx:
            start_sub = all_subs[idx]
            remaining_points = set(range(v))
            for p in start_sub:
                remaining_points.discard(p)
            
            block = list(start_sub)
            # Greedily fill
            for _ in range(k - len(block)):
                best_cand = None
                best_s = -1
                # Sample candidates
                cands = list(rng.choice(sorted(remaining_points), size=min(30, len(remaining_points)), replace=False))
                for cand in cands:
                    b = tuple(sorted(block + [cand]))
                    s = 0
                    for s_sub in itertools.combinations(b, t):
                        if count[sub_to_idx[s_sub]] == 0:
                            s += 1
                    if s > best_s:
                        best_s = s
                        best_cand = cand
                if best_cand is None:
                    best_cand = remaining_points.pop()
                block.append(best_cand)
                remaining_points.discard(best_cand)
            
            block = tuple(sorted(block))
            gain = 0
            for s_sub in itertools.combinations(block, t):
                if count[sub_to_idx[s_sub]] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is not None and best_gain > 0:
            blocks.append(best_block)
            update_count(best_block, 1)
        else:
            break

    # Phase 2: Local Optimization (Remove & Replace)
    # We iterate until no improvement is found or time runs out.
    
    # Precompute list of points in each block for faster union checks?
    # Not strictly necessary, but sets are good.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        
        for bi in indices:
            if time.time() > deadline - 5:
                break
            
            block_to_remove = blocks[bi]
            update_count(block_to_remove, -1)
            
            uncovered_mask = count == 0
            if not np.any(uncovered_mask):
                # Redundant block
                blocks.pop(bi)
                improved = True
                continue
            
            # Try to replace with a block that covers all uncovered triples
            uncovered_indices = np.where(uncovered_mask)[0]
            uncovered_triples = [all_subs[i] for i in uncovered_indices]
            
            union_points = set()
            for tr in uncovered_triples:
                union_points.update(tr)
            
            if len(union_points) <= k:
                # We can cover all uncovered triples with a single block containing union_points
                remaining_points = set(range(v))
                for p in union_points:
                    remaining_points.discard(p)
                
                # Try a few random fillings to minimize redundancy (optional, but good)
                best_new_block = None
                best_redundancy = float('inf')
                
                fill_needed = k - len(union_points)
                if fill_needed > 0:
                    # Try 3 random fillings
                    for _ in range(3):
                        fillers = list(rng.choice(sorted(remaining_points), size=fill_needed, replace=False))
                        new_block_list = list(union_points) + fillers
                        new_block = tuple(sorted(new_block_list))
                        
                        # Calculate redundancy: sum of counts for triples in this block
                        # Lower is better (less overlap with existing blocks)
                        redundancy = 0
                        for s_sub in itertools.combinations(new_block, t):
                            redundancy += count[sub_to_idx[s_sub]]
                        
                        if redundancy < best_redundancy:
                            best_redundancy = redundancy
                            best_new_block = new_block
                else:
                    best_new_block = tuple(sorted(union_points))
                
                if best_new_block is not None:
                    blocks[bi] = best_new_block
                    update_count(best_new_block, 1)
                    improved = True
            else:
                # Cannot replace, restore
                update_count(block_to_remove, 1)

    # Phase 3: Final Coverage Check
    # If we broke early due to time, ensure coverage
    while time.time() < deadline - 2:
        if is_covered():
            break
        
        uncovered_mask = count == 0
        uncovered_indices = np.where(uncovered_mask)[0]
        tr = all_subs[uncovered_indices[0]]
        
        remaining_points = set(range(v))
        for p in tr:
            remaining_points.discard(p)
        
        block = list(tr)
        fill_needed = k - len(block)
        fillers = list(rng.choice(sorted(remaining_points), size=fill_needed, replace=False))
        block.extend(fillers)
        block = tuple(sorted(block))
        
        blocks.append(block)
        update_count(block, 1)
        
        # Quick redundancy check after adding
        for bi in range(len(blocks)-1, -1, -1):
            update_count(blocks[bi], -1)
            if not np.any(count == 0):
                blocks.pop(bi)
            else:
                update_count(blocks[bi], 1)
                break

    return [list(b) for b in blocks]
