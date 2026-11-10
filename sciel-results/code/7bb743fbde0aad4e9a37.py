import time
import itertools
import random

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

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Baseline: tight AP {0, ..., 11}
    # Known to have max Sidon subset size 4.
    baseline = list(range(12))
    best_candidate = baseline
    best_score = _max_sidon_subset_size(baseline)
    
    # If we find a score of 3, we stop and return it.
    # If not, we return the best score 4 candidate (or baseline).
    
    candidates_to_try = []
    
    # 1. Generate Structured Candidates
    # Family 1: APs
    for diff in range(1, 12):
        for start in range(0, 129 - 11*diff):
            cand = [start + i*diff for i in range(12)]
            candidates_to_try.append(cand)
            
    # Family 2: Two interleaved APs (Union of two small APs)
    # Ensure they are disjoint
    for d1 in range(1, 6):
        for d2 in range(1, 6):
            for offset in range(0, 129 - 5*max(d1,d2) - 1):
                set1 = set(i*d1 for i in range(6))
                set2 = set(offset + i*d2 for i in range(6))
                combined = set1 | set2
                if len(combined) == 12 and max(combined) <= 128:
                    candidates_to_try.append(sorted(combined))
                    
    # Family 3: Symmetric sets around a center
    # {c-5d, c-4d, ..., c, c+d, ..., c+5d} is an AP.
    # Try non-uniform symmetric sets?
    # e.g., {c-5, c-4, c-2, c-1, c, c+1, c+2, c+4, c+5, ...}
    # This is hard to generate systematically.
    
    # Family 4: "Modular" patterns mapped to [0, 128]
    # x_i = (i * k) % M, then map to distinct values?
    # Simple approach: Take a base pattern and scale/shift.
    base_patterns = [
        list(range(12)),
        [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 127, 126], # Mix of powers and wrap
        [0, 1, 3, 7, 15, 31, 63, 127, 126, 125, 124, 123], # Powers of 2 - 1
    ]
    for pat in base_patterns:
        # Scale and shift
        for scale in range(1, 11):
            for shift in range(0, 129 - max(pat)*scale):
                cand = [p * scale + shift for p in pat]
                if len(set(cand)) == 12 and max(cand) <= 128:
                    candidates_to_try.append(cand)

    # Deduplicate
    seen = set()
    unique_candidates = []
    for cand in candidates_to_try:
        key = tuple(sorted(cand))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    # Evaluate structured candidates
    for cand in unique_candidates:
        if time.time() > deadline - 5.0:
            break
        score = _max_sidon_subset_size(cand)
        if score < best_score:
            best_score = score
            best_candidate = cand
            if best_score == 3:
                break # Can't get better than 3 (theoretical min for n=12 is likely 3 or 4)

    # If best_score is still 4 (or higher), do a random search for "dense" sets
    # Random sets are unlikely to have max Sidon < 4, but let's try.
    # Focus on "clustered" random sets.
    if best_score > 3 and time.time() < deadline - 2.0:
        attempts = 0
        while time.time() < deadline - 2.0 and attempts < 5000:
            attempts += 1
            # Generate a clustered set: pick 2-3 centers, place elements nearby
            num_clusters = random.randint(2, 3)
            centers = random.sample(range(10, 120), num_clusters)
            cand = set()
            for c in centers:
                # Add 4-6 elements around c
                count = 12 // num_clusters
                for _ in range(count):
                    # Small random offset
                    offset = random.randint(-5, 5)
                    val = c + offset
                    if 0 <= val <= 128 and val not in cand:
                        cand.add(val)
            
            # Fill remaining if needed
            if len(cand) < 12:
                # Add random elements
                while len(cand) < 12:
                    val = random.randint(0, 128)
                    cand.add(val)
            
            cand_list = sorted(list(cand))[:12]
            if len(set(cand_list)) == 12:
                score = _max_sidon_subset_size(cand_list)
                if score < best_score:
                    best_score = score
                    best_candidate = cand_list
                    if best_score == 3:
                        break

    return best_candidate
# EVOLVE-BLOCK-END
