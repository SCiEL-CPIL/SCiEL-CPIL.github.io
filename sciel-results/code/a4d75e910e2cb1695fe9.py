import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def _max_sidon_subset_size(arr):
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    def is_sidon(subset):
        m = len(subset)
        sums = set()
        for i in range(m):
            for j in range(i, m):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Try subsets in decreasing size order
    for size in range(n, 0, -1):
        for combo in itertools.combinations(range(n), size):
            subset = [arr[i] for i in combo]
            if is_sidon(subset):
                return size
    return 0

def _check_all_4_subsets_non_sidon(arr):
    """Check if ALL 4-element subsets are non-Sidon (i.e., max Sidon subset <= 3)."""
    n = len(arr)
    for combo in itertools.combinations(range(n), 4):
        subset = [arr[i] for i in combo]
        if _is_sidon_4(subset):
            return False
    return True

def _is_sidon_4(subset):
    """Check if a 4-element set is Sidon."""
    a, b, c, d = subset
    sums = set()
    pairs = [(a,a),(a,b),(a,c),(a,d),(b,b),(b,c),(b,d),(c,c),(c,d),(d,d)]
    for x, y in pairs:
        s = x + y
        if s in sums:
            return False
        sums.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Baseline: tight AP
    baseline = list(range(12))
    baseline_score = _max_sidon_subset_size(baseline)
    best_candidate = baseline
    best_score = baseline_score
    
    # Try to find a set with max Sidon subset = 3
    # Strategy: search over structured sets with heavy additive collisions
    
    # Approach 1: Try all APs (already known to give 4, but verify)
    # Approach 2: Try sets with more structure
    
    # Key insight: To get max Sidon subset = 3, we need every 4-subset to have a sum collision.
    # This is very restrictive. Let's try a targeted search.
    
    # Try sets of the form: union of two interleaved APs with small differences
    # Or: sets where elements are clustered
    
    # First, try to see if any 12-set can have max Sidon subset = 3
    # We'll do a heuristic search
    
    # Try various structured families
    candidates_to_try = []
    
    # Family 1: APs with different starts and diffs (known to give 4, but check)
    for diff in range(1, 12):
        for start in range(0, 129 - 11*diff):
            cand = [start + i*diff for i in range(12)]
            candidates_to_try.append(cand)
    
    # Family 2: Two interleaved APs
    for d1 in range(1, 8):
        for d2 in range(1, 8):
            for offset in range(0, 129 - 5*max(d1,d2) - 1):
                set1 = set(i*d1 for i in range(6))
                set2 = set(offset + i*d2 for i in range(6))
                combined = set1 | set2
                if len(combined) == 12 and max(combined) <= 128:
                    candidates_to_try.append(sorted(combined))
    
    # Family 3: Clustered sets - elements in two tight clusters
    for c1_start in range(0, 20):
        for c2_start in range(20, 120):
            # 6 elements in each cluster with small spread
            for spread1 in range(1, 4):
                for spread2 in range(1, 4):
                    cluster1 = set(c1_start + i*spread1 for i in range(6))
                    cluster2 = set(c2_start + i*spread2 for i in range(6))
                    combined = cluster1 | cluster2
                    if len(combined) == 12 and max(combined) <= 128:
                        candidates_to_try.append(sorted(combined))
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for cand in candidates_to_try:
        key = tuple(cand)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)
    
    # Evaluate candidates, prioritizing those likely to have lower Sidon subsets
    # First check if any has max Sidon subset = 3
    found_3 = False
    for cand in unique_candidates:
        if time.time() > deadline - 5.0:
            break
        # Quick check: if any 4-subset is Sidon, score >= 4
        # Only do full check if we haven't found a 3 yet
        if best_score > 3:
            if _check_all_4_subsets_non_sidon(cand):
                best_score = 3
                best_candidate = cand
                found_3 = True
                break
        else:
            # Already found 3, just continue checking for confirmation
            pass
    
    # If we found score 3, great. Otherwise, try to confirm score 4 is minimal
    # and return the best candidate with score 4.
    if not found_3:
        # Evaluate all candidates to find the best score-4 candidate
        for cand in unique_candidates:
            if time.time() > deadline - 3.0:
                break
            score = _max_sidon_subset_size(cand)
            if score < best_score:
                best_score = score
                best_candidate = cand
            elif score == best_score and score < 4:
                best_candidate = cand
    
    return best_candidate
# EVOLVE-BLOCK-END
