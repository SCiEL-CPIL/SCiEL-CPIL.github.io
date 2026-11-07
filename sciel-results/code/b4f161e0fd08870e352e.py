import time
import random

def construct():
    n = 12
    m = 40
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s limit
    
    # Represent sets as bitmasks for fast comparability check
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    def count_comparable(fam):
        """Count comparable pairs in a family of masks"""
        c = 0
        for i in range(len(fam)):
            for j in range(i + 1, len(fam)):
                if is_comparable(fam[i], fam[j]):
                    c += 1
        return c
    
    def mask_to_list(mask):
        return [b for b in range(n) if (mask >> b) & 1]
    
    # Strategy: Build a family that's as "chain-like" as possible
    # Use greedy approach: start with empty set, keep adding comparable sets
    
    def greedy_build():
        """Build a family greedily, preferring sets comparable to existing ones"""
        family = [0]  # Start with empty set (mask 0)
        used = {0}
        
        # Generate candidate sets: all subsets, sorted by "chain potential"
        # Prefer sets that can be in a chain with existing family
        candidates = list(range(1, 1 << n))
        random.shuffle(candidates)
        
        while len(family) < m and candidates:
            best_idx = -1
            best_score = -1
            # Check a random sample of candidates for speed
            sample_size = min(len(candidates), 500)
            sample = random.sample(candidates, sample_size)
            for idx, cand in enumerate(sample):
                if cand in used:
                    continue
                # Score: number of existing family members comparable to cand
                score = sum(1 for f in family if is_comparable(cand, f))
                # Bonus for being in a "chain" (comparable to min and max)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx >= 0:
                cand = sample[best_idx]
                family.append(cand)
                used.add(cand)
                candidates.remove(cand)
            else:
                break
        
        # If we still don't have enough, add random unused sets
        if len(family) < m:
            remaining = [c for c in range(1 << n) if c not in used]
            random.shuffle(remaining)
            for c in remaining:
                if len(family) >= m:
                    break
                family.append(c)
                used.add(c)
        
        return family[:m]
    
    def local_search(family, max_iter=5000):
        """Improve family via swap-based local search"""
        family = list(family)
        used = set(family)
        best_score = count_comparable(family)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try swapping one element with an unused element
            for i in range(m):
                if time.time() > deadline:
                    break
                old = family[i]
                # Try replacing with a random unused set
                for _ in range(5):
                    new = random.randint(0, (1 << n) - 1)
                    if new in used:
                        continue
                    
                    # Calculate change in comparable pairs
                    # Remove old's contributions
                    old_contrib = 0
                    for j in range(m):
                        if j != i and is_comparable(old, family[j]):
                            old_contrib += 1
                    
                    # Add new's contributions
                    new_contrib = 0
                    for j in range(m):
                        if j != i and is_comparable(new, family[j]):
                            new_contrib += 1
                    
                    delta = new_contrib - old_contrib
                    if delta > 0:
                        used.remove(old)
                        used.add(new)
                        family[i] = new
                        best_score += delta
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try random restarts with perturbation
                if random.random() < 0.1:
                    # Perturb: randomly swap 2-3 elements
                    for _ in range(random.randint(2, 3)):
                        i = random.randint(0, m - 1)
                        j = random.randint(0, m - 1)
                        if i != j:
                            family[i], family[j] = family[j], family[i]
        
        return family, best_score
    
    # Multiple restarts
    best_family = None
    best_score = -1
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            # Pure greedy
            family = greedy_build()
        elif restart < 10:
            # Greedy with different random seeds
            random.seed(restart * 12345)
            family = greedy_build()
        else:
            # Random initialization then local search
            random.seed(restart * 67890)
            family = random.sample(range(1 << n), m)
        
        family, score = local_search(family, max_iter=2000)
        
        if score > best_score:
            best_score = score
            best_family = list(family)
    
    # Final polish: try more intensive local search on best
    if time.time() < deadline - 10:
        best_family, best_score = local_search(best_family, max_iter=10000)
    
    # Convert to list of lists
    result = [mask_to_list(mask) for mask in best_family]
    
    # Verify: exactly 40 distinct lists, each with distinct integers 0..11
    assert len(result) == m
    assert len(set(tuple(sorted(r)) for r in result)) == m
    for r in result:
        assert len(r) == len(set(r))
        assert all(isinstance(x, int) and 0 <= x <= n - 1 for x in r)
    
    return result
