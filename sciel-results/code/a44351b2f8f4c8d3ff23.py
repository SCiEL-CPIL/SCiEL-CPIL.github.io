"""Approximate 3/2 ratio within a divisor-sum fiber."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    # Reduced MAX_N to ensure sieve completes quickly
    # 2 million allows for good search space while keeping time low
    MAX_N = 2_000_000
    
    # Compute sigma(n) for all n up to MAX_N using sieve
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Efficient sieve:
    # For each d, add d to sigma[d], sigma[2d], ...
    # Using NumPy slicing for speed
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
        
        # Periodic time check
        if d % 100_000 == 0:
            if time.time() > deadline - 15:
                break
    
    # Check if we completed the sieve
    if time.time() > deadline - 10:
        # Fallback: return a known valid pair if found, else default
        # We need a valid pair. Let's use a small hardcoded search or return 11,6 if valid?
        # sigma(11)=12, sigma(6)=12. 11/6 = 1.833, 3/2=1.5. Error=0.333.
        # Let's try to find a better one quickly if possible, but 11,6 is a safe bet.
        # Actually, let's just return the best found so far if any, else 11,6.
        # But we haven't searched yet.
        return [11, 6]
    
    # Build groups: sigma value -> list of n
    # Using a dictionary of lists
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        groups[sigma[n]].append(n)
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # For each group, find pairs (a, b) minimizing |a/b - 3/2|
    best_error = None
    best_pair = (11, 6)
    
    for sig_val, members in groups.items():
        if time.time() > deadline - 2:
            break
        
        if len(members) < 2:
            continue
        
        members.sort()
        n_members = len(members)
        
        # For each b in members, find a in members closest to 3b/2
        for i in range(n_members):
            b = members[i]
            target_a = 1.5 * b
            
            # Binary search for closest to target_a
            idx = np.searchsorted(members, target_a)
            
            candidates = []
            if idx < n_members:
                candidates.append(members[idx])
            if idx > 0:
                candidates.append(members[idx - 1])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute error: |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                den = 2 * b
                error = num / den
                
                if best_error is None or error < best_error:
                    best_error = error
                    best_pair = (a, b)
    
    if best_error is None:
        return [11, 6]
    
    a, b = best_pair
    a = int(a)
    b = int(b)
    
    # Validate types and constraints
    if not isinstance(a, int) or not isinstance(b, int):
        return [11, 6]
    if a <= 0 or b <= 0 or a > MAX_N or b > MAX_N:
        return [11, 6]
    
    # Double-check sigma(a) == sigma(b)
    if sigma[a] != sigma[b]:
        return [11, 6]
    
    return [a, b]

# EVOLVE-BLOCK-END
