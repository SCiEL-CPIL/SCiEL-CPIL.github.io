import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    p = 11
    
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Generate all PSL(2,11) transformations
    # SL(2,11): matrices [[a,b],[c,d]] with ad-bc = 1 mod 11
    # PSL(2,11) = SL(2,11) / {±I}, so 660 elements
    # We'll generate all and apply to seed, deduplicating blocks
    
    def apply_trans(x, a, b, c, d):
        """Apply Möbius transformation x -> (ax+b)/(cx+d) mod p, with infinity handling"""
        if x == p:  # infinity
            if c == 0:
                return p  # infinity maps to infinity
            else:
                return inv[c] * a % p  # a/c mod p
        # x is 0..10
        denom = (c * x + d) % p
        if denom == 0:
            return p  # maps to infinity
        else:
            num = (a * x + b) % p
            return num * inv[denom] % p
    
    # Seed block: a known hexad in the Witt design via PSL(2,11)
    seed = [0, 1, 3, 4, 5, 9]
    
    blocks_set = set()
    
    # Generate all valid (a,b,c,d) with ad-bc = 1 mod 11
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # Need ad - bc = 1 => ad = 1 + bc
                if a == 0:
                    # -bc = 1 => bc = -1 = 10 mod 11
                    if c == 0:
                        continue
                    bc = (b * c) % p
                    if bc != 10:
                        continue
                    # a=0, so d is free? No: ad - bc = 0 - bc = -bc = 1 => bc = 10
                    # d doesn't appear, so any d works? No, we need the matrix to be in SL(2,11)
                    # det = ad - bc = 0*d - bc = -bc = 1, so bc = 10.
                    # d is determined? No, with a=0, det = -bc, independent of d.
                    # But we need unique matrices. Let's just iterate d too.
                    for d in range(p):
                        det = (a * d - b * c) % p
                        if det == 1:
                            # Apply to seed
                            new_block = tuple(sorted(apply_trans(x, a, b, c, d) for x in seed))
                            blocks_set.add(new_block)
                else:
                    # d = (1 + bc) * inv[a] mod p
                    bc = (b * c) % p
                    d = (1 + bc) * inv[a] % p
                    det = (a * d - b * c) % p
                    if det == 1:
                        new_block = tuple(sorted(apply_trans(x, a, b, c, d) for x in seed))
                        blocks_set.add(new_block)
    
    # Also need to consider the -I equivalent, but since we're deduplicating by set,
    # and -I gives the same orbit elements (just permuted), we should be fine.
    # Actually, PSL(2,11) identifies M and -M, so we only need one representative.
    # The above generates all of SL(2,11), which is 1320 matrices, giving 660 PSL elements.
    # But we're applying all 1320, so each block appears twice (from M and -M).
    # That's fine since we use a set.
    
    if len(blocks_set) != 132:
        # Fallback: use the backtracking approach if PSL generation fails
        points = list(range(1, 12))
        all_4_subsets = set(combinations(points, 4))
        candidates = list(combinations(points, 5))
        subset_to_cands = {s: [] for s in all_4_subsets}
        for i, cand in enumerate(candidates):
            for s in combinations(cand, 4):
                subset_to_cands[s].append(i)
        
        first_block = (1, 2, 3, 4, 5)
        first_idx = candidates.index(first_block)
        covered_4 = set(combinations(first_block, 4))
        blocks_4511 = [first_block]
        uncovered = all_4_subsets - covered_4
        
        def backtrack(covered, current_blocks, uncovered):
            if time.time() > deadline:
                return None
            if len(current_blocks) == 66:
                return current_blocks
            if not uncovered:
                return None
            
            min_count = float('inf')
            target_subset = None
            valid_cands = []
            
            for u in uncovered:
                vc = []
                for idx in subset_to_cands[u]:
                    cand = candidates[idx]
                    compatible = True
                    for s in combinations(cand, 4):
                        if s in covered:
                            compatible = False
                            break
                    if compatible:
                        vc.append(idx)
                if len(vc) == 0:
                    return None
                if len(vc) < min_count:
                    min_count = len(vc)
                    target_subset = u
                    valid_cands = vc
                    if min_count == 1:
                        break
            
            if target_subset is None:
                return None
                
            for idx in valid_cands:
                cand = candidates[idx]
                new_covered = set(covered)
                for s in combinations(cand, 4):
                    new_covered.add(s)
                new_uncovered = uncovered - new_covered
                result = backtrack(new_covered, current_blocks + [cand], new_uncovered)
                if result is not None:
                    return result
            return None
        
        sol_4511 = backtrack(covered_4, blocks_4511, uncovered)
        if sol_4511 is None:
            return []
        
        final_blocks = []
        for b in sol_4511:
            final_blocks.append([0] + list(b))
        for b in sol_4511:
            b_set = set(b)
            comp = [pt for pt in points if pt not in b_set]
            final_blocks.append(comp)
        return final_blocks
    
    # Convert to list of lists
    result = [list(b) for b in blocks_set]
    return result
