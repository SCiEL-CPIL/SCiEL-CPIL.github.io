import time
import sys
import random

def _check_validity_bitset(A):
    """Check validity using bitsets for subset sums."""
    n = len(A)
    if n == 0:
        return True
    
    # B[k] is a bitset where bit s is 1 if sum s is achievable by a subset of size k
    # Max sum is 256 * 60 = 15360. Bitset size ~ 15361 bits.
    # Python integers can handle this easily.
    B = [0] * (n + 1)
    B[0] = 1  # Sum 0 with 0 elements
    
    for a in A:
        # Update in reverse to avoid using 'a' multiple times in the same step
        for k in range(n - 1, -1, -1):
            if B[k]:
                B[k + 1] |= (B[k] << a)
        
    # Check for conflicts:
    # We need B[k] and B[j] to be disjoint for all k != j.
    total = 0
    sum_bits = 0
    for k in range(n + 1):
        total |= B[k]
        sum_bits += B[k]
        
    return total == sum_bits

def _sufficient_condition(A):
    """Check sufficient condition: min sum of (k+1) smallest > max sum of k largest for all k."""
    n = len(A)
    if n == 0:
        return True
    A_sorted = sorted(A)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + A_sorted[i]
    total = prefix[n]
    for k in range(n):
        min_sum_k1 = prefix[k + 1]
        max_sum_k = total - prefix[n - k]
        if min_sum_k1 <= max_sum_k:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    best_A = []
    
    # Helper to check time
    def time_left():
        return deadline - (time.time() - start_time)

    # Strategy 1: Greedy Ascending with Exact Check
    A = []
    for candidate in range(1, 257):
        if time_left() < 5:
            break
        new_A = A + [candidate]
        if _check_validity_bitset(new_A):
            A = new_A
    if len(A) > len(best_A):
        best_A = A[:]

    # Strategy 2: Greedy Descending with Exact Check
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

    # Strategy 3: Start with Powers of 2 and extend
    if time_left() > 5:
        A = [1, 2, 4, 8, 16, 32, 64, 128]
        # Try to add more
        for candidate in range(1, 257):
            if time_left() < 5:
                break
            if candidate in A:
                continue
            new_A = A + [candidate]
            if _check_validity_bitset(new_A):
                A = new_A
        if len(A) > len(best_A):
            best_A = A[:]
        
        # Try to replace elements with larger ones to allow more additions?
        # This is complex. Instead, try a randomized restart from powers of 2.
        
    # Strategy 4: Backtracking Search
    # Use the best greedy solution as a base
    if time_left() > 10:
        base_A = best_A[:]
        n = len(base_A)
        
        # Compute initial bitsets
        B = [0] * (n + 1)
        B[0] = 1
        for a in base_A:
            for k in range(n - 1, -1, -1):
                if B[k]:
                    B[k + 1] |= (B[k] << a)
        
        # Try to extend to various target sizes
        # We want to find if we can add 1, 2, ... more elements
        for target_size in range(n + 1, min(n + 5, 20)):
            if time_left() < 5:
                break
            
            # Candidate order: try elements not in base_A
            candidates = [x for x in range(1, 257) if x not in base_A]
            
            # Sort candidates by "separation" heuristic?
            # Prefer elements that are far from existing sums?
            # For now, try ascending and descending
            
            found = False
            for order_type in ['ascending', 'descending']:
                if time_left() < 3:
                    break
                
                if order_type == 'ascending':
                    ordered_candidates = candidates[:]
                else:
                    ordered_candidates = candidates[::-1]
                
                # Simple recursive search with forward checking
                def search(current_A, current_B, current_n, target, deadline, candidates):
                    if current_n >= target:
                        return current_A[:]
                    if time.time() > deadline:
                        return None
                    
                    for cand in candidates:
                        if time.time() > deadline:
                            return None
                        if cand in current_A:
                            continue
                        
                        new_B = current_B[:]
                        # Update bitsets
                        for k in range(current_n - 1, -1, -1):
                            if new_B[k]:
                                new_B[k + 1] |= (new_B[k] << cand)
                        
                        # Check validity
                        total = 0
                        sum_bits = 0
                        valid = True
                        for k in range(current_n + 2):
                            if k < len(new_B):
                                total |= new_B[k]
                                sum_bits += new_B[k]
                            else:
                                break
                        if total != sum_bits:
                            continue
                        
                        new_A = current_A + [cand]
                        result = search(new_A, new_B, current_n + 1, target, deadline, candidates)
                        if result is not None:
                            return result
                    return None

                result = search(base_A, B, n, target_size, time.time() + min(time_left() - 3, 10), ordered_candidates)
                if result is not None and len(result) > len(best_A):
                    best_A = result
                    found = True
                    break
            
            if found:
                continue  # Try next target size
        
    # Strategy 5: Local Search (Add)
    if time_left() > 3:
        A = best_A[:]
        improved = True
        while improved and time_left() > 1:
            improved = False
            n = len(A)
            
            # Try to add one element
            for candidate in range(1, 257):
                if time_left() < 1:
                    break
                if candidate in A:
                    continue
                new_A = A + [candidate]
                if _check_validity_bitset(new_A):
                    A = new_A
                    improved = True
                    break
            
            if len(A) > len(best_A):
                best_A = A[:]
            if not improved:
                break

    # Strategy 6: Multiple Randomized Greedy Restarts
    random.seed(42)
    for _ in range(10):
        if time_left() < 5:
            break
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        A = []
        for candidate in candidates:
            if time_left() < 5:
                break
            new_A = A + [candidate]
            if _check_validity_bitset(new_A):
                A = new_A
        if len(A) > len(best_A):
            best_A = A[:]

    return [int(x) for x in best_A]
