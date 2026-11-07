import time
import itertools
import random

def _to_mask(lst):
    """Convert list of vertices to a 12-bit mask."""
    mask = 0
    for v in lst:
        mask |= (1 << v)
    return mask

def _from_mask(mask):
    """Convert 12-bit mask to list of vertices."""
    return [i for i in range(12) if (mask >> i) & 1]

def _is_subset(a_mask, b_mask):
    """Check if a_mask is a subset of b_mask (using bitwise AND)."""
    return (a_mask & b_mask) == a_mask

def _is_comparable_mask(a_mask, b_mask):
    """Check if one mask is a strict subset of the other."""
    return (a_mask & b_mask) == a_mask and a_mask != b_mask or \
           (a_mask & b_mask) == b_mask and a_mask != b_mask

def _count_comparable_masks(masks):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(masks)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _is_comparable_mask(masks[i], masks[j]):
                count += 1
    return count

def _local_search_masks(masks, all_masks, used_set, deadline, max_iter=1000):
    """Local search using bitmask operations for speed."""
    current_score = _count_comparable_masks(masks)
    improved = True
    iter_count = 0
    n = len(masks)
    
    while improved and iter_count < max_iter and time.time() < deadline:
        improved = False
        iter_count += 1
        
        # Try removing one set and adding a better one
        for i in range(n):
            if time.time() > deadline:
                break
            
            saved_mask = masks[i]
            masks.pop(i)
            used_set.discard(saved_mask)
            
            # Find best replacement
            best_gain = -1
            best_mask = None
            best_idx = None
            
            for idx, candidate_mask in enumerate(all_masks):
                if candidate_mask in used_set:
                    continue
                if time.time() > deadline:
                    break
                
                # Count comparable pairs with current sets
                cnt = 0
                for existing in masks:
                    if _is_comparable_mask(candidate_mask, existing):
                        cnt += 1
                
                if cnt > best_gain:
                    best_gain = cnt
                    best_mask = candidate_mask
                    best_idx = idx
            
            if best_mask is not None and best_idx is not None:
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
    
    return current_score

def _construct_base():
    """Construct a strong initial solution using chains and bitmask local search."""
    deadline = time.time() + 85  # 85 second deadline for safety
    
    # Generate all subsets as masks
    all_masks = []
    for r in range(13):
        for combo in itertools.combinations(range(12), r):
            all_masks.append(_to_mask(list(combo)))
    
    random.seed(42)
    random.shuffle(all_masks)
    
    best_masks = []
    used_set = set()
    
    # Chain 1: Standard chain {0}, {0,1}, {0,1,2}, ...
    chain1_masks = [_to_mask(list(range(i))) for i in range(13)]
    for m in chain1_masks:
        best_masks.append(m)
        used_set.add(m)
    
    # Chain 2: Shifted chain
    chain2_candidates = []
    for i in range(1, 12):
        s = [1] + list(range(2, 2 + i - 1))
        m = _to_mask(s)
        if m not in used_set:
            chain2_candidates.append(m)
    
    # Add up to 13 more sets from chain2
    for m in chain2_candidates[:13]:
        if len(best_masks) >= 40:
            break
        best_masks.append(m)
        used_set.add(m)
    
    # If we still need more, add greedily
    remaining = 40 - len(best_masks)
    if remaining > 0:
        candidates = [m for m in all_masks if m not in used_set]
        random.shuffle(candidates)
        
        for _ in range(remaining):
            if time.time() > deadline:
                break
            best_gain = -1
            best_mask = None
            best_idx = None
            
            for idx, m in enumerate(candidates):
                if time.time() > deadline:
                    break
                cnt = 0
                for existing in best_masks:
                    if _is_comparable_mask(m, existing):
                        cnt += 1
                if cnt > best_gain:
                    best_gain = cnt
                    best_mask = m
                    best_idx = idx
            
            if best_mask is not None and best_idx is not None:
                best_masks.append(best_mask)
                used_set.add(best_mask)
                candidates.pop(best_idx)
    
    # Local search to improve
    if time.time() < deadline:
        candidates = [m for m in all_masks if m not in used_set]
        random.shuffle(candidates)
        _local_search_masks(best_masks, candidates, used_set, deadline, max_iter=500)
    
    # Verify we have exactly 40 distinct sets
    if len(best_masks) != 40:
        # Fallback: fill with random sets
        candidates = [m for m in all_masks if m not in used_set]
        random.shuffle(candidates)
        for m in candidates:
            if len(best_masks) >= 40:
                break
            best_masks.append(m)
            used_set.add(m)
    
    assert len(best_masks) == 40
    assert len(set(best_masks)) == 40
    
    return [_from_mask(m) for m in best_masks]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline
    
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
        # Add remaining sets from a fixed pattern
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
            # Add more
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
