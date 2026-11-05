"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
from itertools import combinations

def _is_sidon_fast(vals):
    """Fast check if vals has all unordered pair sums unique (including diagonals)."""
    n = len(vals)
    if n <= 1:
        return True
    seen = set()
    for i in range(n):
        vi = vals[i]
        for j in range(i, n):
            s = vi + vals[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_subset_size_fast(subset, deadline=None):
    """
    Find largest subset with all unordered pair sums unique (including diagonals).
    Uses a smarter approach: try to build the largest Sidon subset greedily
    and use backtracking for verification.
    """
    n = len(subset)
    if n <= 1:
        return n
    
    # For small n, use brute force but with early termination
    # Try from largest size down
    for size in range(n, 0, -1):
        if deadline and time.time() > deadline:
            return 0
        
        # For size >= 8, use a heuristic check first
        if size >= 8:
            # Quick check: if the full set is not Sidon, try removing elements
            # that cause the most collisions
            if not _is_sidon_fast(subset):
                # Try all subsets of this size - but limit to avoid explosion
                count = 0
                for combo in combinations(range(n), size):
                    if deadline and time.time() > deadline:
                        return 0
                    count += 1
                    if count > 5000:  # Limit brute force
                        break
                    vals = [subset[i] for i in combo]
                    if _is_sidon_fast(vals):
                        return size
                continue
            else:
                return size
        
        # For smaller sizes, do full brute force
        for combo in combinations(range(n), size):
            if deadline and time.time() > deadline:
                return 0
            vals = [subset[i] for i in combo]
            if _is_sidon_fast(vals):
                return size
    
    return 1

def _count_duplicate_sums_fast(subset):
    """Count how many unordered pair sums (including diagonals) are duplicated."""
    n = len(subset)
    sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
    c = Counter(sums)
    return sum(k * (k - 1) // 2 for k in c.values())

def _local_search_fast(candidate, deadline, max_iter=100):
    """
    Local search to maximize duplicate sums (minimize Sidon subset size).
    Uses incremental updates for speed.
    """
    candidate = list(candidate)
    n = len(candidate)
    
    def dup_count(arr):
        sums = [arr[i] + arr[j] for i in range(n) for j in range(i, n)]
        c = Counter(sums)
        return sum(k * (k - 1) // 2 for k in c.values())
    
    current_dups = dup_count(candidate)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try replacing each element with all possible values
        for i in range(n):
            old_val = candidate[i]
            # Try a subset of values for speed - focus on values that might create collisions
            # Sample strategically rather than trying all 129
            for new_val in range(0, 129, 1):
                if new_val == old_val:
                    continue
                # Check if new_val is already in candidate (excluding position i)
                if any(candidate[j] == new_val for j in range(n) if j != i):
                    continue
                
                candidate[i] = new_val
                new_dups = dup_count(candidate)
                
                if new_dups > current_dups:
                    current_dups = new_dups
                    improved = True
                    # Don't break - keep trying to find even better
                else:
                    candidate[i] = old_val
        
        if not improved:
            break
    
    return candidate, current_dups

def _generate_structured_candidate(seed_offset=0):
    """Generate a candidate with structured collisions."""
    random.seed(42 + seed_offset)
    
    # Strategy: create pairs/triples that sum to the same value
    # This maximizes duplicate sums and minimizes Sidon subset size
    
    candidates = []
    
    # Pattern 1: Arithmetic progression (many collisions)
    candidates.append(list(range(12)))
    
    # Pattern 2: Two interleaved arithmetic progressions
    candidates.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    
    # Pattern 3: Symmetric around a center
    candidates.append([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])  # Will be deduplicated
    candidates.append(sorted(set([0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0])))
    
    # Pattern 4: Pairs summing to same value
    # (0,12), (1,11), (2,10), (3,9), (4,8), (5,7) - all sum to 12
    candidates.append([0, 12, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7])
    
    # Pattern 5: Triples with same sum
    # (0,1,2)=3, (0,1,2) can't repeat... let's use different approach
    # Values where a+b = c+d for multiple pairs
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Pattern 6: Concentrated values
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Pattern 7: Mix of close and spread values
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20])
    
    # Pattern 8: Binary-like but with collisions
    candidates.append([0, 1, 2, 4, 8, 16, 32, 64, 63, 31, 15, 7])
    
    # Pattern 9: Modular structure
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Pattern 10: Random with bias toward collisions
    base = random.sample(range(0, 20), 12)
    candidates.append(base)
    
    # Pattern 11: Two clusters
    candidates.append([0, 1, 2, 3, 4, 5, 60, 61, 62, 63, 64, 65])
    
    # Pattern 12: Geometric with modifications
    candidates.append([1, 2, 4, 8, 16, 32, 64, 3, 6, 12, 24, 48])
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Evaluate structured candidates first
    best_subset = list(range(12))
    best_score = _max_sidon_subset_size_fast(best_subset, deadline)
    
    random.seed(42)
    
    # Try structured candidates
    structured_candidates = _generate_structured_candidate()
    
    for candidate in structured_candidates:
        if time.time() > deadline - 20:
            break
        # Ensure exactly 12 distinct values in [0, 128]
        candidate = list(dict.fromkeys(candidate))  # Remove duplicates preserving order
        while len(candidate) < 12:
            candidate.append(random.randint(0, 128))
        candidate = candidate[:12]
        if len(set(candidate)) != 12:
            continue
        
        score = _max_sidon_subset_size_fast(candidate, deadline)
        if score < best_score:
            best_score = score
            best_subset = candidate
            if best_score <= 3:
                break
    
    # Local search with multiple restarts
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Generate random candidate or perturb best
        if restart % 4 == 0:
            candidate = random.sample(range(129), 12)
        elif restart % 4 == 1:
            candidate = list(best_subset)
            # Perturb 1-2 elements
            for _ in range(random.randint(1, 2)):
                idx = random.randint(0, 11)
                candidate[idx] = random.randint(0, 128)
                attempts = 0
                while any(candidate[j] == candidate[idx] for j in range(12) if j != idx):
                    candidate[idx] = random.randint(0, 128)
                    attempts += 1
                    if attempts > 20:
                        break
        elif restart % 4 == 2:
            # Create candidate with intentional collisions
            # Pick a target sum and create pairs
            target = random.randint(10, 120)
            pairs = []
            used = set()
            for a in range(0, min(target, 129)):
                b = target - a
                if b >= 0 and b <= 128 and a != b and a not in used and b not in used:
                    pairs.append((a, b))
                    used.add(a)
                    used.add(b)
                    if len(pairs) >= 6:
                        break
            candidate = []
            for a, b in pairs:
                candidate.extend([a, b])
            while len(candidate) < 12:
                val = random.randint(0, 128)
                if val not in candidate:
                    candidate.append(val)
            candidate = candidate[:12]
        else:
            # Hybrid: start from best and add random perturbations
            candidate = list(best_subset)
            for _ in range(random.randint(2, 4)):
                idx = random.randint(0, 11)
                candidate[idx] = random.randint(0, 128)
                attempts = 0
                while any(candidate[j] == candidate[idx] for j in range(12) if j != idx):
                    candidate[idx] = random.randint(0, 128)
                    attempts += 1
                    if attempts > 20:
                        break
        
        # Local search
        improved_candidate, _ = _local_search_fast(candidate, deadline - 3)
        
        # Evaluate
        if time.time() > deadline - 2:
            break
        
        score = _max_sidon_subset_size_fast(improved_candidate, deadline)
        
        if score < best_score:
            best_score = score
            best_subset = improved_candidate
            if best_score <= 3:
                break
    
    return sorted(best_subset)
# EVOLVE-BLOCK-END
