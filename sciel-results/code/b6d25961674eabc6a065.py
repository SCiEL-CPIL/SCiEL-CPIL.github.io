import time
import random
import itertools
import numpy as np

def is_sidon(subset):
    """Check if a subset has all unique pair sums including diagonals."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    """Compute the maximum size of a Sidon subset in A."""
    n = len(A)
    for k in range(n, 0, -1):
        for combo in itertools.combinations(A, k):
            if is_sidon(combo):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute pair-sum collision structure
    # For a set A of 12 elements, we want every 4-subset to have a collision.
    # A 4-subset {a,b,c,d} has collision if:
    #   a+b == c+d, a+c == b+d, a+d == b+c (pair sum collision)
    #   or a+a == b+c, etc. (diagonal vs pair)
    #   or a+a == b+b (impossible for distinct)
    
    # Strategy: Build sets where many 4-subsets have a+b=c+d type collisions.
    # Use structured patterns: pairs that sum to same value.
    
    def count_4_sidon(A):
        """Count number of 4-element Sidon subsets."""
        cnt = 0
        for combo in itertools.combinations(A, 4):
            if is_sidon(combo):
                cnt += 1
        return cnt
    
    def has_4_sidon(A):
        """Check if any 4-element Sidon subset exists."""
        for combo in itertools.combinations(A, 4):
            if is_sidon(combo):
                return True
        return False
    
    # Initial best: start with a known structure
    # Try: {0, 1, 2, 4, 8, 16, 32, 64, 96, 120, 124, 128} - powers of 2 plus extras
    # But we want to MINIMIZE, so we want collisions.
    
    # Better initial: use arithmetic progression-like structure
    # {0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15} - many collisions
    
    candidates = []
    
    # Pattern 1: Two interleaved APs
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    # Pattern 2: Dense cluster
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12])
    # Pattern 3: Symmetric pairs
    candidates.append([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    # Pattern 4: Quadratic residues mod small prime
    candidates.append([0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121])
    # Pattern 5: Mixed
    candidates.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    
    best_A = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        # Check max sidon size
        ms = max_sidon_size(cand)
        if ms < best_score:
            best_score = ms
            best_A = list(cand)
    
    # If we haven't found size 4, use the best we have
    # Now do local search to try to reduce to 4 (or below)
    
    # Local search: modify one element at a time
    current_A = list(best_A)
    current_ms = best_score
    
    # If current_ms > 4, try hard to get to 4
    # If current_ms == 4, try to verify it's truly minimal
    
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        A_new = list(current_A)
        idx = random.randint(0, 11)
        
        # Generate new value: try to create collisions
        # Strategy: pick a value that creates a+b=c+d with existing elements
        if random.random() < 0.5:
            # Find a pair (i,j) and try to make new_val + A[k] = A[i] + A[j]
            # i.e., new_val = A[i] + A[j] - A[k]
            i, j, k = random.sample(range(12), 3)
            new_val = A_new[i] + A_new[j] - A_new[k]
        elif random.random() < 0.3:
            # Small perturbation
            new_val = A_new[idx] + random.randint(-5, 5)
        else:
            # Random in range
            new_val = random.randint(0, 128)
        
        # Ensure valid
        if new_val < 0 or new_val > 128:
            new_val = random.randint(0, 128)
        
        if new_val in A_new:
            # Try to find a valid replacement
            found = False
            for _ in range(20):
                new_val = random.randint(0, 128)
                if new_val not in A_new:
                    found = True
                    break
            if not found:
                continue
        
        A_new[idx] = new_val
        
        # Evaluate
        new_ms = max_sidon_size(A_new)
        
        if new_ms < current_ms:
            current_A = A_new
            current_ms = new_ms
            no_improve = 0
            
            if current_ms <= 4:
                # Try a few more iterations to see if we can go lower
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    A_try = list(current_A)
                    idx2 = random.randint(0, 11)
                    i2, j2, k2 = random.sample(range(12), 3)
                    nv = A_try[i2] + A_try[j2] - A_try[k2]
                    if 0 <= nv <= 128 and nv not in A_try:
                        A_try[idx2] = nv
                        try_ms = max_sidon_size(A_try)
                        if try_ms < current_ms:
                            current_A = A_try
                            current_ms = try_ms
                            no_improve = 0
        else:
            no_improve += 1
        
        # Periodic restart if stuck
        if no_improve > 200:
            # Restart from a random structured candidate
            restart_patterns = [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18],
                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24],
                [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66],
            ]
            current_A = list(random.choice(restart_patterns))
            # Perturb slightly
            for _ in range(3):
                idx_r = random.randint(0, 11)
                current_A[idx_r] = random.randint(0, 128)
                while current_A.count(current_A[idx_r]) > 1:
                    current_A[idx_r] = random.randint(0, 128)
            current_ms = max_sidon_size(current_A)
            no_improve = 0
    
    return current_A
