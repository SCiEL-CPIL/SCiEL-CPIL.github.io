import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ.
    
    We first try to find a 32-element set via local search. If that fails within
    the time budget, we fall back to the known valid 31-element AP [226..256].
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    def is_valid(s):
        """Check if set s satisfies the cardinality-separated subset sum property."""
        n = len(s)
        s_sorted = sorted(s)
        # Quick range check: if ranges are disjoint, it's valid
        # If ranges overlap, we need to check actual sums for small cardinalities
        min_sums = [0] * (n + 1)
        max_sums = [0] * (n + 1)
        for c in range(1, n + 1):
            min_sums[c] = sum(s_sorted[:c])
            max_sums[c] = sum(s_sorted[-c:])
        
        # If all ranges are disjoint, valid
        disjoint = True
        for c1 in range(n + 1):
            for c2 in range(c1 + 1, n + 1):
                if max_sums[c1] >= min_sums[c2]:
                    disjoint = False
                    break
            if not disjoint:
                break
        if disjoint:
            return True
        
        # Ranges overlap; check actual sums for overlapping cardinality pairs
        from itertools import combinations
        for c1 in range(n + 1):
            for c2 in range(c1 + 1, n + 1):
                if max_sums[c1] < min_sums[c2] or max_sums[c2] < min_sums[c1]:
                    continue  # ranges don't overlap
                # Ranges overlap; check for actual sum collision
                # Only feasible for small c1, c2
                if c1 > 6 or c2 > 6:
                    # Too expensive to enumerate; assume invalid (conservative)
                    return False
                sums_c1 = set()
                for combo in combinations(s, c1):
                    sums_c1.add(sum(combo))
                for combo in combinations(s, c2):
                    if sum(combo) in sums_c1:
                        return False
        return True
    
    # Try to find a 32-element set
    # Start with AP [225..256] and try local search
    base = list(range(225, 257))  # 32 elements
    if time.time() < deadline:
        # Try the base AP first
        if is_valid(base):
            return base
    
    # Try modifying the AP: replace some elements to create more spread
    # Strategy: use a superincreasing-like tail
    # Try sets of form [a, a+1, ..., a+m-1, b, b+1, ..., b+p-1] with gap
    # Or try [1, 2, 4, 8, 16, 32, 64, 128, ...] mixed with others
    
    # Try: [256, 255, ..., 225] with one element moved to create a gap
    # Actually, let's try a different structure: two clusters
    # e.g., [1..16] union [240..256] = 16+17=33 elements? No, 16+17=33 but need distinct
    
    # Let's try: small cluster + large cluster
    # [1,2,...,m] union [N-m+1, ..., 256] where these don't overlap
    # For m=16: [1..16] union [241..256] = 16+16=32 elements
    candidates = []
    
    # Candidate 1: [1..16] + [241..256]
    c1 = list(range(1, 17)) + list(range(241, 257))
    candidates.append(c1)
    
    # Candidate 2: [1..15] + [240..256] = 15+17=32
    c2 = list(range(1, 16)) + list(range(240, 257))
    candidates.append(c2)
    
    # Candidate 3: [1..14] + [239..256] = 14+18=32
    c3 = list(range(1, 15)) + list(range(239, 257))
    candidates.append(c3)
    
    # Candidate 4: [2..17] + [240..256] = 16+17=33? No, 16+17=33, too many
    # [2..16] + [240..256] = 15+17=32
    c4 = list(range(2, 17)) + list(range(240, 257))
    candidates.append(c4)
    
    # Candidate 5: [1..12] + [235..256] = 12+22=34? No, 12+22=34
    # [1..10] + [233..256] = 10+24=34
    # Let's be more careful: need exactly 32
    # [1..m] + [256-p+1..256] where m+p=32 and m < 256-p+1
    for m in range(1, 32):
        p = 32 - m
        start2 = 256 - p + 1
        if m < start2:  # no overlap
            cand = list(range(1, m + 1)) + list(range(start2, 257))
            candidates.append(cand)
    
    # Try each candidate
    for cand in candidates:
        if time.time() >= deadline:
            break
        if len(cand) == 32 and len(set(cand)) == 32 and all(1 <= x <= 256 for x in cand):
            if is_valid(cand):
                return cand
    
    # Try local search from AP [225..256]
    if time.time() < deadline:
        current = list(range(225, 257))
        best = current
        # Try swapping elements
        import random
        random.seed(42)
        for iteration in range(200):
            if time.time() >= deadline:
                break
            # Try replacing one element with a random other
            i = random.randint(0, 31)
            old_val = current[i]
            # Try values not in current
            available = [x for x in range(1, 257) if x not in set(current)]
            if not available:
                break
            new_val = random.choice(available)
            candidate = current[:]
            candidate[i] = new_val
            candidate.sort()
            if is_valid(candidate):
                current = candidate
                best = candidate[:]
                break
    
    # Fallback: known valid 31-element set
    result = list(range(226, 257))
    return result
# EVOLVE-BLOCK-END
