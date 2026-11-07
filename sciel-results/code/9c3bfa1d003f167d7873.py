import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    full_mask = (1 << 12) - 1

    # Precompute subset and superset neighbor lists for fast degree computation
    # For a given mask, comparable sets are those that are subsets or supersets
    # Precompute for all 4096 masks: list of all subsets and all supersets
    subsets_of = [[] for _ in range(4096)]
    supersets_of = [[] for _ in range(4096)]
    for mask in range(4096):
        # Enumerate all subsets of mask
        sub = mask
        while True:
            subsets_of[mask].append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & mask
        # Enumerate all supersets of mask
        sup = mask
        remaining = full_mask ^ mask
        while True:
            supersets_of[mask].append(sup)
            if remaining == 0:
                break
            # Get lowest set bit of remaining
            low = remaining & (-remaining)
            remaining ^= low
            sup |= low
    # Note: supersets_of[mask] includes mask itself, and subsets_of[mask] includes mask itself
    # For comparability: A and B are comparable iff A is subset of B or B is subset of A
    # So for a candidate mask C, it's comparable with S iff S is in subsets_of[C] or S is in supersets_of[C]
    # But we need S != C for strict subset (well, the problem says "one member is a strict subset of the other")
    # Actually re-reading: "one member is a strict subset of the other" means A ⊂ B or B ⊂ A (strict)
    # So C and S are comparable iff (C ⊂ S) or (S ⊂ C), i.e., C != S and (C|S == C or C|S == S)
    # Which is: C != S and (S in subsets_of[C] or C in subsets_of[S])
    # Equivalently: C != S and (S in supersets_of[C] or S in subsets_of[C])
    # Let's build a set of comparable masks for each mask (excluding itself)
    comparable_set = [set() for _ in range(4096)]
    for mask in range(4096):
        # All subsets (including mask) and all supersets (including mask)
        for s in subsets_of[mask]:
            if s != mask:
                comparable_set[mask].add(s)
        for s in supersets_of[mask]:
            if s != mask:
                comparable_set[mask].add(s)
    # Convert to list for faster membership? No, set is fine for membership
    # But we need to count how many family members are in comparable_set[new_mask]
    # We can use: sum(1 for s in family if s in comparable_set[new_mask])
    # Or better: maintain a count array
    # Actually, let's precompute comparable_list as sorted lists for potential optimization
    # For now, use sets for O(1) membership

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
        
        family = list(sets)
        seen = set(family)
        
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
    
    # Fast degree computation using precomputed comparable sets
    def get_degree_fast(mask, fam, exclude_idx=None):
        """Count how many sets in fam are comparable with mask using precomputed sets."""
        comp = comparable_set[mask]
        cnt = 0
        for i, s in enumerate(fam):
            if i == exclude_idx:
                continue
            if s in comp:
                cnt += 1
        return cnt
    
    # Generate Hamming distance 1 and 2 neighbors
    def hamming_neighbors(mask, dist=2):
        """Generate masks at Hamming distance up to 'dist' from mask."""
        neighbors = set()
        bits = [i for i in range(12) if (mask >> i) & 1]
        nonbits = [i for i in range(12) if not ((mask >> i) & 1)]
        
        if dist >= 1:
            for b in bits:
                neighbors.add(mask ^ (1 << b))
            for b in nonbits:
                neighbors.add(mask | (1 << b))
        
        if dist >= 2:
            for i in range(len(bits)):
                for j in range(i+1, len(bits)):
                    neighbors.add(mask ^ (1 << bits[i]) ^ (1 << bits[j]))
            for i in range(len(nonbits)):
                for j in range(i+1, len(nonbits)):
                    neighbors.add(mask | (1 << nonbits[i]) | (1 << nonbits[j]))
            for b in bits:
                for nb in nonbits:
                    neighbors.add(mask ^ (1 << b) | (1 << nb))
        
        neighbors.discard(mask)
        return list(neighbors)
    
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(40):
            if time.time() > deadline - 5:
                break
            old = family[i]
            # Try structured neighbors (Hamming distance 1-2) first, then random
            candidates = hamming_neighbors(old, 2)
            random.shuffle(candidates)
            # Also add some random candidates
            for _ in range(20):
                r = random.randint(0, 4095)
                if r != old:
                    candidates.append(r)
            
            for new in candidates:
                if time.time() > deadline - 5:
                    break
                if new == old or new in family:
                    continue
                deg_old = get_degree_fast(old, family, i)
                deg_new = get_degree_fast(new, family, i)
                delta = deg_new - deg_old
                if delta > 0:
                    family[i] = new
                    cur_score += delta
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_family = list(family)
                    break  # restart from beginning
    
    # Second phase: more random exploration with structured neighbors
    for _ in range(5):
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
                candidates = hamming_neighbors(old, 3)
                random.shuffle(candidates)
                for _ in range(30):
                    r = random.randint(0, 4095)
                    if r != old:
                        candidates.append(r)
                
                for new in candidates:
                    if time.time() > deadline - 3:
                        break
                    if new == old or new in family:
                        continue
                    deg_old = get_degree_fast(old, family, i)
                    deg_new = get_degree_fast(new, family, i)
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
