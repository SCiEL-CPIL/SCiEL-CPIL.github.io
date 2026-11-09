import time
import sys

def _check_validity_bitset(A):
    """Check validity using bitsets for subset sums."""
    n = len(A)
    if n == 0:
        return True
    
    # B[k] is a bitset where bit s is 1 if sum s is achievable by a subset of size k
    B = [0] * (n + 1)
    B[0] = 1  # Sum 0 with 0 elements
    
    for a in A:
        # Update in reverse to avoid using 'a' multiple times in the same step
        for k in range(n - 1, -1, -1):
            if B[k]:
                B[k + 1] |= (B[k] << a)
        
    # Check for conflicts:
    # We need B[k] and B[j] to be disjoint for all k != j.
    # This is equivalent to:
    # 1. Compute Total = OR of all B[k]
    # 2. Compute Sum_Bits = Sum of all B[k] (as integers)
    # 3. If Total != Sum_Bits, there is a carry, meaning some bit was set in multiple B[k].
    # Wait, integer addition carries.
    # Example: B[0]=1 (bit 0), B[1]=2 (bit 1). Sum=3 (bits 0,1). Total=3 (bits 0,1). Match.
    # Example: B[0]=1 (bit 0), B[1]=1 (bit 0). Sum=2 (bit 1). Total=1 (bit 0). Mismatch.
    # So if Total != Sum_Bits, it's invalid.
    
    total = 0
    sum_bits = 0
    for k in range(n + 1):
        total |= B[k]
        sum_bits += B[k]
        
    return total == sum_bits

def construct():
    start_time = time.time()
    deadline = 55.0
    
    best_A = []
    
    # Helper to check time
    def time_left():
        return deadline - (time.time() - start_time)
        
    # Strategy 1: Greedy Ascending
    A = []
    for candidate in range(1, 257):
        if time_left() < 5:
            break
        new_A = A + [candidate]
        if _check_validity_bitset(new_A):
            A = new_A
    if len(A) > len(best_A):
        best_A = A[:]

    # Strategy 2: Greedy Descending
    if time_left() > 5:
        A = []
        for candidate in range(256, 0, -1):
            if time_left() < 5:
                break
            new_A = A + [candidate]
            if _check_validity_bitset(new_A):
                A = new_A
        if len(A) > len(best_A):
            best_A = A[:]

    # Strategy 3: Local Search (Add and Swap)
    if time_left() > 5:
        A = best_A[:]
        improved = True
        while improved and time_left() > 2:
            improved = False
            n = len(A)
            
            # Try to add one element
            added = False
            for candidate in range(1, 257):
                if time_left() < 2:
                    break
                if candidate in A:
                    continue
                new_A = A + [candidate]
                if _check_validity_bitset(new_A):
                    A = new_A
                    improved = True
                    added = True
                    break
            
            if added:
                continue
                
            # If we couldn't add, try swapping
            # Try to replace A[i] with candidate to potentially allow future additions?
            # Or just to see if a different set of same size allows adding more later?
            # This is complex. Let's just try random restarts or different orders.
            
            # Try a different greedy order?
            # Or try to find a "better" set of same size?
            # For now, break if no improvement.
            break
            
    # Strategy 4: Multiple Randomized Greedy Restarts
    # Use a few random orders to escape local optima
    import random
    random.seed(42)
    
    for _ in range(10):
        if time_left() < 10:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        A = []
        for candidate in candidates:
            if time_left() < 10:
                break
            new_A = A + [candidate]
            if _check_validity_bitset(new_A):
                A = new_A
        if len(A) > len(best_A):
            best_A = A[:]
            # Update deadline to allow more time if we found a better solution?
            # No, just keep going.

    return [int(x) for x in best_A]
