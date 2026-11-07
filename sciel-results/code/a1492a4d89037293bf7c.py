"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    n = 256
    
    # Precompute prime factors for each number 1..256
    prime_factors = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if not prime_factors[i]:
            for j in range(i, n + 1, i):
                prime_factors[j].append(i)
    
    # Greedy initialization: pick largest numbers that are coprime with current set
    # 1 is always included
    selected = {1}
    # Sort numbers from 256 down to 2
    for num in range(n, 1, -1):
        if all(gcd(num, s) == 1 for s in selected):
            selected.add(num)
    
    current_sum = sum(selected)
    
    # Local search to improve
    # We try to add numbers (0-for-1) and swap numbers (1-for-1)
    # To make it faster, we precompute coprime checks or just use gcd
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # 1. Try to add any number not in selected
        for num in range(n, 1, -1):
            if num in selected:
                continue
            if all(gcd(num, s) == 1 for s in selected):
                selected.add(num)
                current_sum += num
                improved = True
                # Restart loop as set changed
                break
        
        if improved:
            continue
            
        # 2. Try 1-for-1 swaps
        # For each number in selected, try to replace it with a larger number not in selected
        for rem in list(selected):
            if rem == 1:
                continue
            # Try adding numbers larger than rem
            for add in range(n, rem, -1):
                if add in selected:
                    continue
                # Check if add is coprime with all other selected numbers
                ok = True
                for s in selected:
                    if s != rem and gcd(add, s) != 1:
                        ok = False
                        break
                if ok:
                    selected.remove(rem)
                    selected.add(add)
                    current_sum += (add - rem)
                    improved = True
                    break
            if improved:
                break
                
    # Convert to list of ints
    result = [int(x) for x in selected]
    return result
# EVOLVE-BLOCK-END
