import time
import random
import itertools

def construct():
    n = 12
    m = 40
    total = 1 << n  # 4096
    
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    # Precompute: for each mask, its bitcount
    bitcount = [0] * total
    for i in range(1, total):
        bitcount[i] = bitcount[i >> 1] + (i & 1)
    
    # Precompute comparability: comp[a][b] = 1 if a and b are comparable (a!=b)
    # But 4096^2 = 16M entries, might be too much memory/time
    # Instead, use on-the-fly check: (a & b) == a or (a & b) == b
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    def count_pairs(family):
        """Count comparable pairs in family (list of masks)."""
        cnt = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if is_comparable(family[i], family[j]):
                    cnt += 1
        return cnt
    
    def score_of(family):
        return count_pairs(family)
    
    # Initial solution: try a structured approach
    # Take all subsets of a 6-element set (64 subsets), pick best 40
    # Or: take a chain and add branches
    
    # Strategy: start with all subsets of {0,1,2,3,4,5} (64 subsets)
    # and pick the 40 with highest "comparability density"
    # Actually, let's just do local search from a good start
    
    # Good start: take subsets of size 0,1,2,...,11 in a structured way
    # Or: take a "ball" around the middle
    
    # Let's try: all subsets of {0..5} (64 sets), randomly pick 40
    base_set = list(range(6))
    all_subsets_base = []
    for mask in range(1 << 6):
        all_subsets_base.append(mask)  # mask over 6 bits
    
    # For local search, we work with masks over 12 bits
    # Initial family: 40 random subsets from the 64 base subsets
    random.seed(42)
    
    best_family = None
    best_score = -1
    
    # Multiple restarts
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Initialize: pick 40 random distinct subsets from base (or all 4096)
        # Use base subsets for density
        if restart < 20:
            # From base 64
            pool = all_subsets_base[:]
            random.shuffle(pool)
            family = pool[:m]
        else:
            # From all 4096, but biased toward small sizes
            pool = list(range(total))
            random.shuffle(pool)
            family = pool[:m]
        
        fam_set = set(family)
        cur_score = score_of(family)
        
        if cur_score > best_score:
            best_score = cur_score
            best_family = family[:]
        
        # Local search: try swaps
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            # Shuffle order of positions to try
            positions = list(range(m))
            random.shuffle(positions)
            
            for i in positions:
                if time.time() > deadline - 3.0:
                    break
                
                # Try replacing family[i] with some other mask
                # Candidate: masks not in family
                # For efficiency, try a limited set of candidates
                
                # Compute current comparability count for family[i] with others
                # Then try candidates that improve it
                
                # Get current contribution of family[i]
                contrib_i = 0
                for j in range(m):
                    if j != i and is_comparable(family[i], family[j]):
                        contrib_i += 1
                
                # Try a few random candidates
                num_tries = 30
                for _ in range(num_tries):
                    new_mask = random.randint(0, total - 1)
                    if new_mask in fam_set:
                        continue
                    
                    # Compute new contribution
                    new_contrib = 0
                    for j in range(m):
                        if j != i and is_comparable(new_mask, family[j]):
                            new_contrib += 1
                    
                    if new_contrib > contrib_i:
                        # Accept
                        fam_set.remove(family[i])
                        family[i] = new_mask
                        fam_set.add(new_mask)
                        cur_score += (new_contrib - contrib_i)
                        improved = True
                        break
        
        if cur_score > best_score:
            best_score = cur_score
            best_family = family[:]
    
    # Final polish: try all pairs of swaps
    if best_family is not None and time.time() < deadline - 2.0:
        family = best_family[:]
        fam_set = set(family)
        cur_score = best_score
        
        # Try single swaps more exhaustively
        for i in range(m):
            if time.time() > deadline - 2.0:
                break
            contrib_i = 0
            for j in range(m):
                if j != i and is_comparable(family[i], family[j]):
                    contrib_i += 1
            
            # Try all masks (might be slow, limit)
            # Just try a few more random
            for _ in range(50):
                new_mask = random.randint(0, total - 1)
                if new_mask in fam_set:
                    continue
                new_contrib = 0
                for j in range(m):
                    if j != i and is_comparable(new_mask, family[j]):
                        new_contrib += 1
                if new_contrib > contrib_i:
                    fam_set.remove(family[i])
                    family[i] = new_mask
                    fam_set.add(new_mask)
                    cur_score += (new_contrib - contrib_i)
                    if cur_score > best_score:
                        best_score = cur_score
                        best_family = family[:]
    
    # Convert to list of lists
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
