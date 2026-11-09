import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    def is_sidon_subset(subset):
        seen = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                sm = subset[i] + subset[j]
                if sm in seen:
                    return False
                seen.add(sm)
        return True
    
    def max_sidon_size(ambient):
        n = len(ambient)
        for size in range(n, 2, -1):
            for subset in itertools.combinations(ambient, size):
                if is_sidon_subset(subset):
                    return size
        return 2
    
    best_set = list(range(12))
    best_ms = 4  # Known: range(12) has max Sidon size 4
    
    # Try to find sets with max Sidon size 3
    # Strategy: use dense clusters where sum collisions are frequent
    
    # Test specific candidates first
    test_candidates = [
        list(range(12)),  # baseline, ms=4
        [0,1,2,3,4,5,6,7,8,9,10,21],
        [0,1,2,3,4,5,10,11,12,13,14,15],
        [0,1,2,3,4,5,6,7,8,9,10,11],
        [0,2,4,6,8,10,12,14,16,18,20,22],
        [0,1,3,4,6,7,9,10,12,13,15,16],
    ]
    
    for cand in test_candidates:
        if max(cand) > 128 or len(set(cand)) != 12:
            continue
        ms = max_sidon_size(cand)
        if ms < best_ms:
            best_ms = ms
            best_set = cand
            if best_ms <= 3:
                return best_set
    
    # Systematic search
    # Try sets with many small numbers
    for max_val in range(12, 30):
        if time.time() > deadline:
            break
        # Try all combinations from 0 to max_val
        # This is C(max_val+1, 12) which can be huge
        # Instead, try structured subsets
        for start in range(0, max(1, max_val - 11)):
            if time.time() > deadline:
                break
            # Dense block
            cand = list(range(start, start + 12))
            if max(cand) <= 128:
                ms = max_sidon_size(cand)
                if ms < best_ms:
                    best_ms = ms
                    best_set = cand
                    if best_ms <= 3:
                        return best_set
    
    # Try two-cluster approach
    for c1 in range(3, 10):
        if time.time() > deadline:
            break
        c2 = 12 - c1
        if c2 < 3:
            continue
        for gap in range(10, 120, 2):
            if time.time() > deadline:
                break
            for s1 in range(0, 3):
                cl1 = list(range(s1, s1 + c1))
                for s2 in range(gap, min(gap + 3, 128 - c2 + 1)):
                    cl2 = list(range(s2, s2 + c2))
                    cand = cl1 + cl2
                    if max(cand) > 128 or len(set(cand)) != 12:
                        continue
                    ms = max_sidon_size(cand)
                    if ms < best_ms:
                        best_ms = ms
                        best_set = cand
                        if best_ms <= 3:
                            return best_set
    
    return best_set
