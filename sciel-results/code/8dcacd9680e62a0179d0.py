import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    
    # Precompute all 2^12 subsets as bitmasks
    # Represent each subset as an integer bitmask
    
    def make_family_from_chains():
        """Create initial family with 3 chains sharing empty set."""
        sets = set()
        
        # Chain 1: add elements 0,1,2,...,11
        s = 0
        sets.add(s)
        for i in range(12):
            s |= (1 << i)
            sets.add(s)
        
        # Chain 2: add elements 1,2,...,11,0
        s = 0
        for v in [1,2,3,4,5,6,7,8,9,10,11,0]:
            s |= (1 << v)
            sets.add(s)
        
        # Chain 3: add elements 2,3,...,11,0,1
        s = 0
        for v in [2,3,4,5,6,7,8,9,10,11,0,1]:
            s |= (1 << v)
            sets.add(s)
        
        # We have 37 sets. Add 3 more.
        # Find sets that maximize comparabilities
        family = list(sets)
        seen = set(family)
        
        # Compute current score
        def score(fam):
            cnt = 0
            for i in range(len(fam)):
                for j in range(i+1, len(fam)):
                    a, b = fam[i], fam[j]
                    if a | b == a or a | b == b:
                        cnt += 1
            return cnt
        
        cur_score = score(family)
        
        # Add 3 more sets greedily
        all_masks = list(range(4096))
        random.shuffle(all_masks)
        
        for _ in range(3):
            best_add = None
            best_sc = cur_score
            for cand in all_masks:
                if cand in seen:
                    continue
                # Check if adding cand increases score
                # Count comparabilities of cand with current family
                deg = 0
                for s in family:
                    if cand | s == cand or cand | s == s:
                        deg += 1
                new_sc = cur_score + deg
                if new_sc > best_sc:
                    best_sc = new_sc
                    best_add = cand
            if best_add is not None:
                family.append(best_add)
                seen.add(best_add)
                cur_score = best_sc
            else:
                break
        
        return family
    
    family = make_family_from_chains()
    
    # Ensure exactly 40 distinct
    family = list(dict.fromkeys(family))
    if len(family) < 40:
        seen = set(family)
        for mask in range(4096):
            if len(family) >= 40:
                break
            if mask not in seen:
                family.append(mask)
                seen.add(mask)
    family = family[:40]
    
    # Local search with incremental updates
    def compute_score(fam):
        cnt = 0
        for i in range(len(fam)):
            for j in range(i+1, len(fam)):
                a, b = fam[i], fam[j]
                if a | b == a or a | b == b:
                    cnt += 1
        return cnt
    
    cur_score = compute_score(family)
    best_family = list(family)
    best_score = cur_score
    
    random.seed(42)
    
    # Precompute comparability for all pairs of masks? Too large (4096^2)
    # Instead, use incremental updates during search
    
    def get_degree(mask, fam, exclude_idx=None):
        """Count how many sets in fam are comparable with mask."""
        cnt = 0
        for i, s in enumerate(fam):
            if i == exclude_idx:
                continue
            if mask | s == mask or mask | s == s:
                cnt += 1
        return cnt
    
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(40):
            if time.time() > deadline - 5:
                break
            old = family[i]
            # Try swapping with random candidates
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                new = random.randint(0, 4095)
                if new == old or new in family:
                    continue
                # Remove old, add new
                # Score change = -deg(old) + deg(new in fam without old + new)
                deg_old = get_degree(old, family, i)
                # For new, compute degree in family with old replaced by new
                deg_new = 0
                for j, s in enumerate(family):
                    if j == i:
                        continue
                    if new | s == new or new | s == s:
                        deg_new += 1
                delta = deg_new - deg_old
                if delta > 0:
                    family[i] = new
                    cur_score += delta
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_family = list(family)
                    break  # restart from beginning
    
    # Final local search pass with more thorough exploration
    for _ in range(3):
        if time.time() > deadline - 3:
            break
        improved = True
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(40):
                if time.time() > deadline - 3:
                    break
                old = family[i]
                # Try all possible replacements (limited for speed)
                for new in random.sample(range(4096), min(200, 4096)):
                    if new == old or new in family:
                        continue
                    deg_old = get_degree(old, family, i)
                    deg_new = 0
                    for j, s in enumerate(family):
                        if j == i:
                            continue
                        if new | s == new or new | s == s:
                            deg_new += 1
                    delta = deg_new - deg_old
                    if delta > 0:
                        family[i] = new
                        cur_score += delta
                        improved = True
                        if cur_score > best_score:
                            best_score = cur_score
                            best_family = list(family)
                        break
                if improved:
                    break
    
    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(12) if (mask >> i) & 1]
        result.append(lst)
    
    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    # Count comparables
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
