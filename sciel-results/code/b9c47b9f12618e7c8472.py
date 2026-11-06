"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from math import gcd
from functools import reduce
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    # Weights scaled by L
    # w[i] = L / (i+1) for i in 0..29
    weights = [L // (i + 1) for i in range(30)]
    
    # Split into two halves of 15
    N_HALF = 15
    N_TOTAL = 30
    N_COMBOS = 3 ** N_HALF
    
    # 2. Generate sums for first half (indices 0..14)
    # Using itertools.product for simplicity, checking time
    # If too slow, switch to iterative base-3
    
    # We store (sum, index)
    # Actually, storing just sums and then sorting indices is better for memory?
    # List of 14M tuples (int, int) is heavy.
    # Let's store just sums, sort them, but we need indices to reconstruct.
    # So we must keep track of indices.
    
    # Optimization: 
    # Generate sumsA list.
    # Create list of indices [0, 1, ..., N_COMBOS-1]
    # Sort indices by sumsA[i]
    
    sumsA = [0] * N_COMBOS
    sumsB = [0] * N_COMBOS
    
    # Generate sumsA
    # weightsA = weights[0:15]
    # weightsB = weights[15:30]
    
    # To speed up generation, we can use the fact that product is lexicographic.
    # But 14M * 15 ops is ~200M ops. 
    # Let's try to be efficient.
    
    # Alternative: Recursion with accumulation
    def generate_sums(weights_part):
        n = len(weights_part)
        sums = [0] * (3 ** n)
        # Iterative base 3
        # coeffs are derived from index
        # This loop is 14M iterations.
        # Inside: 15 steps to compute sum? 
        # We can update incrementally.
        
        # Let's stick to product for now, it's C optimized for iteration.
        # Summing 15 ints is fast.
        vals = [-1, 0, 1]
        idx = 0
        for combo in itertools.product(vals, repeat=n):
            s = 0
            # Unroll loop? No, n is 15.
            for c, w in zip(combo, weights_part):
                if c != 0:
                    s += c * w
            sums[idx] = s
            idx += 1
            if time.time() > deadline - 30:
                # Fallback to smaller search if too slow? 
                # Or just break and use partial? 
                # Better to have a faster generator.
                pass
        return sums

    # The above might be slow. Let's use a faster incremental generator.
    def generate_sums_fast(weights_part):
        n = len(weights_part)
        N = 3 ** n
        sums = [0] * N
        # coeffs[i] is 0, 1, 2 representing -1, 0, 1
        coeffs = [0] * n
        # Start with all -1 (0 in base 3)
        # sum = sum (coeffs[i] - 1) * weights_part[i]
        # sum = sum coeffs[i]*w[i] - sum w[i]
        # Let's just compute directly.
        
        # coeffs = [0]*n -> all -1
        current_sum = 0
        for c, w in zip(coeffs, weights_part):
            current_sum += (c - 1) * w
        sums[0] = current_sum
        
        idx = 1
        # Increment coeffs in base 3
        # coeffs[0] is least significant? 
        # In product, last element changes fastest.
        # So coeffs[n-1] is least significant.
        # Let's align: 
        # product yields (c0, c1, ..., c14).
        # idx 0: (-1, -1, ..., -1)
        # idx 1: (-1, -1, ..., 0)
        # So c14 changes fastest.
        # Base 3: idx = c0*3^14 + ... + c14*3^0
        # So c14 = idx % 3.
        
        # We iterate idx from 1 to N-1.
        # We can update sum from previous idx.
        # Find which digit changed from idx-1 to idx.
        # This is expensive to find every time.
        
        # Alternative:
        # Just compute from scratch for each idx? 
        # 14M * 15 = 215M ops.
        # In Python, 215M ops is ~20-30s.
        # Plus sorting ~10s.
        # Total ~40s.
        # This is risky but likely fits in 90s.
        
        # Let's try to optimize the inner loop.
        # Precompute weights_part as a tuple.
        w_tuple = tuple(weights_part)
        
        for i in range(1, N):
            # Get coeffs for i
            # This is O(n) per iteration.
            # Total O(N*n).
            
            # Faster:
            # We can maintain the coeffs array.
            # But we need to reset it? No, we can increment it.
            pass
        
        # Let's use the product method but with a tighter loop.
        sums = [0] * N
        vals = (-1, 0, 1)
        idx = 0
        for combo in itertools.product(vals, repeat=n):
            # Use sum with generator? No, explicit loop is faster in Py3.
            s = 0
            # Unroll for n=15?
            # c0, c1, ..., c14
            # s = c0*w0 + c1*w1 + ...
            # We can use zip.
            for c, w in zip(combo, weights_part):
                if c:
                    s += c * w
            sums[idx] = s
            idx += 1
            # Time check
            if idx % 1000000 == 0 and time.time() > deadline - 20:
                # If we are running out of time, we might not finish.
                # But 14M is not that many.
                pass
        return sums

    sumsA = generate_sums_fast(weights[0:15])
    sumsB = generate_sums_fast(weights[15:30])
    
    if time.time() > deadline - 10:
        # Emergency fallback: return seed-like solution
        return [0]*28 + [1, -1]
        
    # Sort indices by sums
    # sorted_indices = sorted(range(N_COMBOS), key=sums.__getitem__)
    # This is fast.
    idxA = list(range(N_COMBOS))
    idxA.sort(key=sumsA.__getitem__)
    sorted_sumsA = [sumsA[i] for i in idxA]
    
    idxB = list(range(N_COMBOS))
    idxB.sort(key=sumsB.__getitem__)
    sorted_sumsB = [sumsB[i] for i in idxB]
    
    if time.time() > deadline - 5:
        return [0]*28 + [1, -1]
        
    # Two pointer search
    # i for A, j for B
    # We want min |sorted_sumsA[i] + sorted_sumsB[j]| != 0
    
    i = 0
    j = N_COMBOS - 1
    best_abs = float('inf')
    best_i = -1
    best_j = -1
    
    # Helper to check a pair
    def check_pair(i, j):
        nonlocal best_abs, best_i, best_j
        if i < 0 or i >= N_COMBOS or j < 0 or j >= N_COMBOS:
            return
        s = sorted_sumsA[i] + sorted_sumsB[j]
        if s == 0:
            return
        abs_s = abs(s)
        if abs_s < best_abs:
            best_abs = abs_s
            best_i = i
            best_j = j
            
    # Scan
    # To ensure we don't miss the next best if we hit 0, we check neighbors.
    # Also, we should check the current pair even if 0? No, we want non-zero.
    # But if we hit 0, the neighbors are the best candidates.
    
    while i < N_COMBOS and j >= 0:
        s = sorted_sumsA[i] + sorted_sumsB[j]
        
        if s == 0:
            # Check neighbors
            check_pair(i+1, j)
            check_pair(i, j-1)
            check_pair(i-1, j)
            check_pair(i, j+1)
            # Move pointers to continue search
            # We can move i up or j down.
            # Let's move i up.
            i += 1
        elif s > 0:
            check_pair(i, j)
            j -= 1
        else: # s < 0
            check_pair(i, j)
            i += 1
            
    if best_i == -1:
        # Fallback
        return [0]*28 + [1, -1]
        
    # Reconstruct coefficients
    idx_a = idxA[best_i]
    idx_b = idxB[best_j]
    
    coeffs = [0] * 30
    
    # Reconstruct A (first 15)
    d = idx_a
    for k in range(14, -1, -1):
        coeffs[k] = (d % 3) - 1
        d //= 3
        
    # Reconstruct B (last 15)
    d = idx_b
    for k in range(29, 14, -1):
        coeffs[k] = (d % 3) - 1
        d //= 3
        
    # Verify
    s_frac = Fraction(0)
    for c, i_val in zip(coeffs, range(1, 31)):
        if c != 0:
            s_frac += Fraction(c, i_val)
            
    if s_frac == 0:
        # Should not happen, but fallback
        return [0]*28 + [1, -1]
        
    return coeffs
# EVOLVE-BLOCK-END
