import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if a set is Sidon (all pair sums i<=j are unique)."""
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    """Find the maximum size of a Sidon subset of A."""
    # Check from largest to smallest
    for k in range(len(A), 2, -1):
        for comb in combinations(A, k):
            if is_sidon(list(comb)):
                return k
    return 2  # minimum Sidon size is 2 (any two distinct elements form a Sidon set)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    best_A = None
    best_score = 12  # worst case
    
    # Start with a dense set in small range
    # Consecutive integers create many collisions
    initial = list(range(12))
    
    # Try several initial candidates
    candidates = [
        list(range(12)),           # 0-11
        list(range(0, 24, 2)),     # 0,2,4,...,22
        [0,1,2,4,5,7,8,10,11,13,14,16],  # structured
    ]
    
    # Add random dense candidates
    for _ in range(30):
        # Sample from small range to encourage collisions
        cand = random.sample(range(0, 20), 12)
        candidates.append(cand)
    
    # Evaluate initial candidates
    for A in candidates:
        if len(set(A)) != 12:
            continue
        score = max_sidon_size(A)
        if score < best_score:
            best_score = score
            best_A = A[:]
            if best_score == 3:
                return best_A
    
    if best_A is None:
        best_A = list(range(12))
        best_score = max_sidon_size(best_A)
    
    # Local search: try to reduce max Sidon size
    A = best_A[:]
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a random element to change
        idx = random.randint(0, 11)
        old_val = A[idx]
        
        # Try replacing with a value that creates more collisions
        # Strategy: pick a value close to existing ones
        existing = [A[i] for i in range(12) if i != idx]
        
        # Try values near existing elements to create 3-term APs
        new_val = None
        for _ in range(20):
            # Pick a random existing element and offset
            base = random.choice(existing)
            offset = random.choice([-3, -2, -1, 1, 2, 3, -4, 4])
            candidate = base + offset
            if 0 <= candidate <= 128 and candidate not in existing:
                new_val = candidate
                break
        
        if new_val is None:
            new_val = random.randint(0, 25)
            if new_val in existing:
                continue
        
        A[idx] = new_val
        current_score = max_sidon_size(A)
        
        if current_score < best_score:
            best_score = current_score
            best_A = A[:]
            no_improve_count = 0
            if best_score == 3:
                return best_A
        else:
            # Revert
            A[idx] = old_val
            no_improve_count += 1
        
        # If stuck for too long, try a random restart
        if no_improve_count > 500:
            no_improve_count = 0
            # Try a fresh random dense set
            A = random.sample(range(0, 25), 12)
            current_score = max_sidon_size(A)
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
                if best_score == 3:
                    return best_A
            else:
                A = best_A[:]
    
    return best_A
