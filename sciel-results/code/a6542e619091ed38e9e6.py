import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s for safety

    N = 51975

    # 1. Generate all divisors of N greater than 1
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    
    valid_moduli = [m for m in divisors if m > 1]
    
    # 2. Greedy Construction
    # We want to select residues for these moduli.
    # Strategy: Add moduli one by one, choosing the residue that covers the most currently uncovered residues.
    # Order of adding: Sort by modulus ascending (smaller modulus = higher density = likely more coverage).
    # But we can also pick the "best" modulus at each step.
    # Given we must use all, the order matters for the greedy choice.
    # Smaller moduli cover more, so they should be placed first to define the "hard" constraints?
    # Actually, if we place a large modulus first, it covers a sparse set.
    # If we place a small modulus first, it covers a dense set.
    # Let's try sorting by modulus ascending.
    
    moduli_sorted = sorted(valid_moduli)
    
    covered = bytearray(N)
    classes = []
    
    # Greedy pass
    for m in moduli_sorted:
        best_r = 0
        max_gain = -1
        # Optimize residue r for modulus m
        # We want to maximize covered[r::m].count(0)
        # This is equivalent to minimizing covered[r::m].count(1)
        
        # To speed up, we can just iterate all r.
        # N/m is the length of the slice.
        # Summing a bytearray slice is fast in Python.
        for r in range(m):
            # Count zeros in covered[r:N:m]
            # total_len = (N - 1 - r) // m + 1
            # covered_count = sum(covered[r:N:m])
            # gain = total_len - covered_count
            
            # Optimization: if max_gain is already very high, we might break? No, just find max.
            # Using sum() on a bytearray slice creates a new object? No, it iterates.
            # Let's use a helper or just inline.
            
            # For speed, we can precompute indices? No, N is small.
            # Let's try to be efficient.
            cnt_1 = 0
            # Manual loop might be slower than C-implemented sum on slice?
            # sum(bytearray_slice) is implemented in C.
            slice_cov = covered[r:N:m]
            cnt_1 = sum(slice_cov)
            
            # Calculate total elements in this arithmetic progression
            # Number of terms = floor((N - 1 - r) / m) + 1
            total = (N - 1 - r) // m + 1
            
            gain = total - cnt_1
            
            if gain > max_gain:
                max_gain = gain
                best_r = r
                
        classes.append([m, best_r])
        # Update covered
        covered[best_r:N:m] = b'\x01' * ((N - 1 - best_r) // m + 1)
        
        # Time check
        if time.time() > deadline:
            break

    # Calculate initial score
    current_uncovered = sum(1 for c in covered if c == 0)
    best_uncovered = current_uncovered
    best_classes = [c[:] for c in classes]
    
    # 3. Local Search
    # Use cov_count to evaluate swaps efficiently.
    
    if time.time() < deadline - 5.0: # Reserve 5s for final calc
    
        # Build cov_count
        cov_count = [0] * N
        for m, r in classes:
            for x in range(r, N, m):
                cov_count[x] += 1
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Shuffle classes to break symmetry? Or just iterate.
            # Iterating in order is fine.
            for i in range(len(classes)):
                m, r = classes[i]
                # Get current set of residues
                old_residues = list(range(r, N, m))
                
                best_new_r = r
                best_delta = 0 # We want to minimize uncovered, so delta < 0 is good
                
                # Try all other residues r'
                for r_prime in range(m):
                    if r_prime == r:
                        continue
                    
                    new_residues = list(range(r_prime, N, m))
                    
                    # Calculate delta
                    # Delta = (newly uncovered) - (newly covered)
                    # Newly uncovered: x in old_residues where cov_count[x] == 1
                    # Newly covered: x in new_residues where cov_count[x] == 0
                    
                    delta = 0
                    for x in old_residues:
                        if cov_count[x] == 1:
                            delta += 1
                    for x in new_residues:
                        if cov_count[x] == 0:
                            delta -= 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_r = r_prime
                        
                if best_new_r != r:
                    # Update cov_count
                    for x in range(r, N, m):
                        cov_count[x] -= 1
                    for x in range(best_new_r, N, m):
                        cov_count[x] += 1
                    classes[i][1] = best_new_r
                    improved = True
        
        # Recalculate final score
        # We can compute from cov_count: uncovered = count of 0s
        final_uncovered = sum(1 for c in cov_count if c == 0)
        if final_uncovered < best_uncovered:
            best_uncovered = final_uncovered
            best_classes = [c[:] for c in classes]

    return best_classes
