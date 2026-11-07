import time
import itertools
import random

def _count_comparable(sets):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(sets)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if len(sets[i]) < len(sets[j]):
                if all(x in sets[j] for x in sets[i]):
                    count += 1
            elif len(sets[j]) < len(sets[i]):
                if all(x in sets[i] for x in sets[j]):
                    count += 1
    return count

def _is_comparable(a, b):
    """Check if a is strict subset of b or vice versa."""
    if len(a) < len(b):
        return all(x in b for x in a)
    elif len(b) < len(a):
        return all(x in a for x in b)
    return False

def _greedy_add(sets, used, candidates, deadline):
    """Greedily add best candidate set to maximize comparable pairs."""
    best_score = -1
    best_set = None
    best_idx = None
    
    for idx, S in enumerate(candidates):
        if time.time() > deadline:
            break
        cnt = 0
        for existing in sets:
            if _is_comparable(S, existing):
                cnt += 1
        if cnt > best_score:
            best_score = cnt
            best_set = S
            best_idx = idx
    
    if best_set is not None and best_idx is not None:
        sets.append(best_set)
        used.add(frozenset(best_set))
        candidates.pop(best_idx)
        return True
    return False

def _local_search(sets, used, candidates, deadline, max_iter=500):
    """Local search: try swapping sets to improve comparable pair count."""
    current_score = _count_comparable(sets)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter and time.time() < deadline:
        improved = False
        iter_count += 1
        
        # Try removing one set and adding a better one
        for i in range(len(sets)):
            if time.time() > deadline:
                break
            
            saved = sets[i]
            saved_fs = frozenset(saved)
            sets.pop(i)
            used.discard(saved_fs)
            
            # Find best replacement
            best_score = -1
            best_set = None
            best_idx = None
            
            for idx, S in enumerate(candidates):
                if time.time() > deadline:
                    break
                cnt = 0
                for existing in sets:
                    if _is_comparable(S, existing):
                        cnt += 1
                if cnt > best_score:
                    best_score = cnt
                    best_set = S
                    best_idx = idx
            
            if best_set is not None and best_idx is not None:
                sets.append(best_set)
                used.add(frozenset(best_set))
                candidates.pop(best_idx)
                
                new_score = _count_comparable(sets)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    sets.pop()
                    used.discard(frozenset(best_set))
                    candidates.insert(best_idx, best_set)
                    sets.insert(i, saved)
                    used.add(saved_fs)
            else:
                # Revert
                sets.insert(i, saved)
                used.add(saved_fs)
    
    return current_score

def _construct_base():
    """Construct a strong initial solution using multiple chains and local search."""
    deadline = time.time() + 85  # 85 second deadline for safety
    
    # Generate all subsets
    all_subsets = []
    for r in range(13):
        for combo in itertools.combinations(range(12), r):
            all_subsets.append(list(combo))
    
    random.seed(42)
    random.shuffle(all_subsets)
    
    # Strategy 1: Multiple disjoint chains
    # Create several chains of different "offsets"
    best_sets = []
    used = set()
    
    # Chain 1: Standard chain {0}, {0,1}, {0,1,2}, ...
    chain1 = [list(range(i)) for i in range(13)]
    for s in chain1:
        best_sets.append(s)
        used.add(frozenset(s))
    
    # Chain 2: Shifted chain starting from different base
    # Use sets that contain {1} and build up
    chain2_candidates = []
    for i in range(1, 12):
        s = [1] + list(range(2, 2 + i - 1))
        if frozenset(s) not in used:
            chain2_candidates.append(s)
    
    # Add up to 13 more sets from chain2
    for s in chain2_candidates[:13]:
        if len(best_sets) >= 40:
            break
        best_sets.append(s)
        used.add(frozenset(s))
    
    # If we still need more, add greedily
    remaining = 40 - len(best_sets)
    if remaining > 0:
        candidates = [s for s in all_subsets if frozenset(s) not in used]
        random.shuffle(candidates)
        
        for _ in range(remaining):
            if time.time() > deadline:
                break
            if not _greedy_add(best_sets, used, candidates, deadline):
                break
    
    # Local search to improve
    if time.time() < deadline:
        candidates = [s for s in all_subsets if frozenset(s) not in used]
        random.shuffle(candidates)
        _local_search(best_sets, used, candidates, deadline, max_iter=200)
    
    # Verify we have exactly 40 distinct sets
    if len(best_sets) != 40:
        # Fallback: fill with random sets
        candidates = [s for s in all_subsets if frozenset(s) not in used]
        random.shuffle(candidates)
        for s in candidates:
            if len(best_sets) >= 40:
                break
            best_sets.append(s)
            used.add(frozenset(s))
    
    assert len(best_sets) == 40
    assert len(set(frozenset(s) for s in best_sets)) == 40
    
    return [list(s) for s in best_sets]

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
