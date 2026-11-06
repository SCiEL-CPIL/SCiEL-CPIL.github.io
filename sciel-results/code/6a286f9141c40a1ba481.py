"""Find a 12-element ambient set in [0,128] minimizing the largest Sidon subset."""
import time

def _is_sidon_bitmask(mask, vals):
    """Check if subset (bitmask over vals) has all pair sums i<=j unique."""
    seen = set()
    idxs = [i for i in range(len(vals)) if (mask >> i) & 1]
    for a_i in range(len(idxs)):
        va = vals[idxs[a_i]]
        for b_i in range(a_i, len(idxs)):
            vb = vals[idxs[b_i]]
            s = va + vb
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(vals):
    """Exhaustive: find largest subset of vals with all pair sums unique."""
    n = len(vals)
    best = 0
    # Check all subsets in descending size order
    for size in range(n, best, -1):
        # Enumerate combinations via bitmask
        for mask in range(1 << n):
            if bin(mask).count('1') != size:
                continue
            if _is_sidon_bitmask(mask, vals):
                best = size
                break
        if best == size:
            break
    return best

def _score(vals):
    return _max_sidon_size(vals)

def _search():
    """Search for a 12-set in [0,128] with max Sidon subset = 11."""
    deadline = time.time() + 75
    # Strategy: use residue classes mod 7.
    # If 4 elements share residue r mod 7, then among those 4,
    # a+a and b+b both ≡ 2r (mod 7), so collision within the 4.
    # This forces max Sidon subset <= 11.
    #
    # Build: pick 4 from residue 0, 4 from residue 1, 2 from residue 2, 2 from residue 3.
    # Or simpler: 4 from one residue, rest spread.
    #
    # Let's try: 4 multiples of 7, 4 from 7k+1, 2 from 7k+2, 2 from 7k+3.
    # Actually simplest: 4 from residue 0, 8 spread across other residues.
    #
    # Let me just try specific constructions and verify.
    
    # Construction 1: 4 elements ≡ 0 mod 7, rest spread
    # {0, 7, 14, 21} are 0 mod 7. Add 8 more from other residues.
    # To avoid creating large Sidon subsets, spread the rest.
    
    candidates = []
    
    # Try: 4 from residue 0 mod 7, 4 from residue 1, 2 from residue 2, 2 from residue 3
    # {0,7,14,21} + {1,8,15,22} + {2,9} + {3,10}
    c1 = [0, 7, 14, 21, 1, 8, 15, 22, 2, 9, 3, 10]
    candidates.append(c1)
    
    # Try: 4 from residue 0, 4 from residue 2, 2 from residue 1, 2 from residue 3
    c2 = [0, 7, 14, 21, 2, 9, 16, 23, 1, 8, 3, 10]
    candidates.append(c2)
    
    # Try: 4 from residue 0, 3 from residue 1, 3 from residue 2, 2 from residue 3
    c3 = [0, 7, 14, 21, 1, 8, 15, 2, 9, 16, 3, 10]
    candidates.append(c3)
    
    # Try: 5 from residue 0 (even stronger), 7 spread
    c4 = [0, 7, 14, 21, 28, 1, 8, 15, 22, 2, 9, 3]
    candidates.append(c4)
    
    # Try: 4 from residue 0, 4 from residue 3, 2 from residue 1, 2 from residue 2
    c5 = [0, 7, 14, 21, 3, 10, 17, 24, 1, 8, 2, 9]
    candidates.append(c5)
    
    # Verify each
    best_set = None
    best_score = 12
    for c in candidates:
        if len(set(c)) != 12:
            continue
        if min(c) < 0 or max(c) > 128:
            continue
        s = _score(c)
        if s < best_score:
            best_score = s
            best_set = c
        if best_score <= 10:
            break
    
    if best_set is not None:
        return best_set
    
    # If none worked, do a more systematic search
    # Try all combinations of 4 from {0,7,14,21,28,35} and 8 from remaining
    # This is too large; instead try a few more hand-crafted sets
    
    # Try with larger spread to avoid accidental Sidon structure
    c6 = [0, 7, 14, 21, 1, 9, 16, 24, 2, 10, 3, 11]
    candidates.append(c6)
    c7 = [0, 7, 14, 21, 1, 8, 16, 23, 2, 9, 3, 11]
    candidates.append(c7)
    c8 = [0, 7, 14, 21, 1, 9, 15, 22, 2, 10, 3, 11]
    candidates.append(c8)
    
    for c in candidates:
        if len(set(c)) != 12 or min(c) < 0 or max(c) > 128:
            continue
        s = _score(c)
        if s < best_score:
            best_score = s
            best_set = c
        if best_score <= 10:
            break
    
    if best_set is not None:
        return best_set
    
    # Fallback: random search
    import random
    random.seed(42)
    for _ in range(5000):
        if time.time() > deadline:
            break
        c = random.sample(range(129), 12)
        s = _score(c)
        if s < best_score:
            best_score = s
            best_set = c
            if best_score <= 10:
                break
    
    if best_set is not None:
        return best_set
    
    return list(range(12))  # fallback

# Pre-compute the best set at import time
_BEST_SET = _search()

def construct():
    return list(_BEST_SET)
