import itertools
import time
import random

def max_sidon_size(S):
    """
    Returns the largest k such that some k-subset of S has all unordered pair sums (with diag) unique.
    """
    best = 1
    n = len(S)
    # If n is small, we can check all subsets.
    # For n=12, checking all subsets of size k is C(12, k).
    # C(12, 4) = 495. C(12, 5) = 792.
    # Checking if a subset is Sidon takes O(k^2).
    # Total ops for k=4: 495 * 16 ~ 8000.
    # Total ops for k=5: 792 * 25 ~ 20000.
    # This is very fast.
    
    for k in range(1, n + 1):
        found = False
        for combo in itertools.combinations(range(n), k):
            sums = set()
            ok = True
            for i in range(k):
                for j in range(i, k):
                    s = S[combo[i]] + S[combo[j]]
                    if s in sums:
                        ok = False
                        break
                    sums.add(s)
                if not ok:
                    break
            if ok:
                found = True
                break
        if found:
            best = k
        else:
            break
    return best

def construct():
    deadline = time.time() + 50
    best_set = None
    best_val = 13
    
    # Candidate 1: Arithmetic Progression 0..11
    # Max Sidon size is 4.
    c1 = list(range(12))
    v1 = max_sidon_size(c1)
    if v1 < best_val:
        best_val = v1
        best_set = c1
        
    # Candidate 2: Try to find a set with max Sidon size 3 via random search
    # We want to minimize the max Sidon size.
    # If we find a set with max Sidon size 3, we return it.
    
    def random_search():
        for _ in range(2000):
            # Generate 12 distinct random integers in [0, 128]
            # Bias towards dense structures?
            # Try mixing small APs
            if random.random() < 0.5:
                # Two small APs
                start1 = random.randint(0, 50)
                start2 = random.randint(60, 120)
                step1 = random.randint(1, 3)
                step2 = random.randint(1, 3)
                cand = []
                for i in range(6):
                    cand.append(start1 + i * step1)
                    cand.append(start2 + i * step2)
            else:
                # Random set
                cand = random.sample(range(0, 129), 12)
            
            cand = sorted(set(cand))
            if len(cand) != 12:
                continue
            if any(x < 0 or x > 128 for x in cand):
                continue
            
            v = max_sidon_size(cand)
            if v < best_val:
                best_val = v
                best_set = cand
                if best_val == 3:
                    return best_set, best_val
            
            if time.time() > deadline:
                break
        return best_set, best_val

    best_set, best_val = random_search()
    
    # If we didn't find 3, we have at least the AP result (4)
    if best_set is None:
        best_set = c1
        best_val = v1
        
    return best_set
