import time
import itertools
import random
import numpy as np

def _to_mask(lst):
    """Convert list of vertices to a 12-bit mask."""
    mask = 0
    for v in lst:
        mask |= (1 << v)
    return mask

def _from_mask(mask):
    """Convert 12-bit mask to list of vertices."""
    return [i for i in range(12) if (mask >> i) & 1]

def _is_comparable_mask(a_mask, b_mask):
    """Check if one mask is a strict subset of the other."""
    if a_mask == b_mask:
        return False
    return (a_mask & b_mask) == a_mask or (a_mask & b_mask) == b_mask

def _count_comparable_masks(masks):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(masks)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _is_comparable_mask(masks[i], masks[j]):
                count += 1
    return count

def _count_comparable_with(masks, candidate):
    """Count how many sets in masks are comparable with candidate."""
    cnt = 0
    for m in masks:
        if _is_comparable_mask(candidate, m):
            cnt += 1
    return cnt

def _local_search_1swap(masks, candidates, used_set, deadline, max_iter=200):
    """Local search using 1-swap moves (remove one, add one)."""
    current_score = _count_comparable_masks(masks)
    n = len(masks)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try all single swaps
        for i in range(n):
            if time.time() > deadline:
                break
            
            saved_mask = masks[i]
            masks.pop(i)
            used_set.discard(saved_mask)
            
            best_gain = -1
            best_mask = None
            
            for candidate_mask in candidates:
                if candidate_mask in used_set:
                    continue
                
                cnt = _count_comparable_with(masks, candidate_mask)
                if cnt > best_gain:
                    best_gain = cnt
                    best_mask = candidate_mask
            
            if best_mask is not None:
                masks.append(best_mask)
                used_set.add(best_mask)
                
                new_score = _count_comparable_masks(masks)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    masks.pop()
                    used_set.discard(best_mask)
                    masks.insert(i, saved_mask)
                    used_set.add(saved_mask)
            else:
                # Revert
                masks.insert(i, saved_mask)
                used_set.add(saved_mask)
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_score

def _local_search_2swap(masks, candidates, used_set, deadline, max_iter=100):
    """Local search using 2-swap moves (remove two, add two)."""
    current_score = _count_comparable_masks(masks)
    n = len(masks)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try all pairs to remove
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                mask_i = masks[i]
                mask_j = masks[j]
                
                # Temporarily remove both
                if i < j:
                    masks.pop(j)
                    masks.pop(i)
                else:
                    masks.pop(i)
                    masks.pop(j)
                
                used_set.discard(mask_i)
                used_set.discard(mask_j)
                
                # Find best pair to add
                best_gain = -1
                best_pair = None
                
                # Sample a subset of candidates for speed
                sample_size = min(len(candidates), 200)
                sample = random.sample(candidates, sample_size) if len(candidates) > sample_size else candidates
                
                for a_idx in range(len(sample)):
                    if time.time() > deadline:
                        break
                    for b_idx in range(a_idx + 1, len(sample)):
                        ca = sample[a_idx]
                        cb = sample[b_idx]
                        
                        if ca in used_set or cb in used_set:
                            continue
                        
                        cnt = _count_comparable_with(masks, ca) + _count_comparable_with(masks, cb)
                        # Add comparability between ca and cb
                        if _is_comparable_mask(ca, cb):
                            cnt += 1
                        
                        if cnt > best_gain:
                            best_gain = cnt
                            best_pair = (ca, cb)
                
                if best_pair is not None:
                    masks.append(best_pair[0])
                    masks.append(best_pair[1])
                    used_set.add(best_pair[0])
                    used_set.add(best_pair[1])
                    
                    new_score = _count_comparable_masks(masks)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        masks.pop()
                        used_set.discard(best_pair[1])
                        masks.pop()
                        used_set.discard(best_pair[0])
                        masks.append(mask_i)
                        masks.append(mask_j)
                        used_set.add(mask_i)
                        used_set.add(mask_j)
                else:
                    # Revert
                    masks.append(mask_i)
                    masks.append(mask_j)
                    used_set.add(mask_i)
                    used_set.add(mask_j)
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_score

def _greedy_construct(deadline):
    """Greedy construction starting from a good initial chain."""
    # Generate all subsets as masks
    all_masks = []
    for r in range(13):
        for combo in itertools.combinations(range(12), r):
            all_masks.append(_to_mask(list(combo)))
    
    random.seed(42)
    
    best_masks = []
    used_set = set()
    
    # Start with a full chain of length 13
    chain_masks = [_to_mask(list(range(i))) for i in range(13)]
    for m in chain_masks:
        best_masks.append(m)
        used_set.add(m)
    
    # Greedily add remaining sets
    remaining = 40 - len(best_masks)
    candidates = [m for m in all_masks if m not in used_set]
    random.shuffle(candidates)
    
    for _ in range(remaining):
        if time.time() > deadline:
            break
        
        best_gain = -1
        best_mask = None
        best_idx = None
        
        for idx, m in enumerate(candidates):
            cnt = _count_comparable_with(best_masks, m)
            if cnt > best_gain:
                best_gain = cnt
                best_mask = m
                best_idx = idx
        
        if best_mask is not None:
            best_masks.append(best_mask)
            used_set.add(best_mask)
            candidates.pop(best_idx)
    
    return best_masks, candidates, used_set

def _construct_base():
    """Construct a strong initial solution using greedy + multi-swap local search."""
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline for safety margin
    
    # Phase 1: Greedy construction
    best_masks, candidates, used_set = _greedy_construct(deadline)
    
    if time.time() > deadline:
        return [_from_mask(m) for m in best_masks[:40]]
    
    # Phase 2: 1-swap local search
    current_score = _local_search_1swap(best_masks, candidates, used_set, deadline, max_iter=150)
    
    if time.time() > deadline:
        return [_from_mask(m) for m in best_masks[:40]]
    
    # Phase 3: 2-swap local search for escaping local optima
    current_score = _local_search_2swap(best_masks, candidates, used_set, deadline, max_iter=50)
    
    if time.time() > deadline:
        return [_from_mask(m) for m in best_masks[:40]]
    
    # Phase 4: Final 1-swap polish
    current_score = _local_search_1swap(best_masks, candidates, used_set, deadline, max_iter=100)
    
    # Verify we have exactly 40 distinct sets
    if len(best_masks) != 40:
        # Fallback: fill with random sets
        extra_candidates = [m for m in candidates if m not in used_set]
        random.shuffle(extra_candidates)
        for m in extra_candidates:
            if len(best_masks) >= 40:
                break
            best_masks.append(m)
            used_set.add(m)
    
    assert len(best_masks) == 40
    assert len(set(best_masks)) == 40
    
    return [_from_mask(m) for m in best_masks]

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 82 second deadline for safety
    
    try:
        result = _construct_base()
        # Verify validity
        assert len(result) == 40
        seen = set()
        for s in result:
            assert isinstance(s, (list, tuple))
            for x in s:
                assert isinstance(x, int) and not isinstance(x, bool)
                assert 0 <= x <= 11
            fs = frozenset(s)
            assert fs not in seen
            seen.add(fs)
        return result
    except Exception:
        # Fallback: simple chain-based construction
        chain = [list(range(i)) for i in range(13)]
        extra = []
        for r in range(1, 12):
            for start in range(12 - r + 1):
                s = list(range(start, start + r))
                if frozenset(s) not in set(frozenset(c) for c in chain) and frozenset(s) not in set(frozenset(e) for e in extra):
                    extra.append(s)
                if len(chain) + len(extra) >= 40:
                    break
            if len(chain) + len(extra) >= 40:
                break
        result = chain + extra[:40 - len(chain)]
        while len(result) < 40:
            for r in range(13):
                for combo in itertools.combinations(range(12), r):
                    s = list(combo)
                    if frozenset(s) not in set(frozenset(x) for x in result):
                        result.append(s)
                        break
                if len(result) >= 40:
                    break
        result = result[:40]
        return [list(s) for s in result]
