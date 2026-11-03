"""
Combined covering code search:
1. Precompute full cover matrix using NumPy.
2. Greedy set cover using matrix-vector dot products for gains.
3. Local search using cover_count array for O(1) amortized removal checks.
"""
import time
import numpy as np
import itertools

_DEADLINE_SECONDS = 140.0

def construct(q, n, radius):
    start = time.monotonic()
    deadline = start + _DEADLINE_SECONDS
    
    total = q ** n
    
    # Helper to convert index to tuple and vice versa
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return t[::-1]
    
    def tuple_to_idx(t):
        idx = 0
        for d in t:
            idx = idx * q + d
        return idx

    # Precompute powers of q for index calculation
    # idx = sum(t[i] * q^(n-1-i))
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q

    # Precompute cover matrix
    # M[i, j] = True if dist(i, j) <= radius
    # For radius 1: dist is 0 or 1.
    M = np.zeros((total, total), dtype=np.uint8)
    np.fill_diagonal(M, 1)
    
    if radius >= 1:
        # For each position, for each word, for each other value, set neighbor
        # Vectorized approach:
        # We can iterate over positions and use broadcasting or slicing.
        # However, explicit loop over positions is fast enough for N=2187.
        for pos in range(n):
            power = q_powers[pos]
            # For each word i, changing digit at pos by delta changes index by delta * power
            # We need to ensure we don't wrap around digits (e.g., 0 -> 2 is not dist 1 in base 3? 
            # Wait, Hamming distance is on the alphabet. 
            # If we have digit d, neighbors are d-1 and d+1 (if valid).
            # The index change is (new_val - old_val) * power.
            # This works as long as we handle the digit boundaries correctly.
            
            # Let's just iterate over all words and positions, it's N * n * (q-1) operations.
            # 2187 * 7 * 2 ~ 30k operations. Very fast.
            for i in range(total):
                w = idx_to_tuple(i)
                for pos in range(n):
                    orig = w[pos]
                    for val in range(q):
                        if val == orig:
                            continue
                        j = i + (val - orig) * power
                        M[i, j] = 1

    # Greedy Construction
    uncovered = np.ones(total, dtype=np.uint8)
    code = []
    M_uint8 = M # already uint8
    
    while np.any(uncovered):
        if time.monotonic() > deadline - 5.0:
            break
            
        # Compute gains: number of uncovered words covered by each candidate
        # gains = sum over j of M[i, j] * uncovered[j]
        # This is M @ uncovered
        gains = M_uint8.dot(uncovered)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Find all candidates with max gain
        candidates = np.where(gains == max_gain)[0]
        
        # Randomly pick one to break symmetry
        best_idx = int(candidates[np.random.randint(len(candidates))])
        
        code.append(best_idx)
        uncovered[M_uint8[best_idx]] = 0

    # Local Search: Removal Phase
    # cover_count[j] = number of codewords covering word j
    cover_count = np.zeros(total, dtype=np.int32)
    for idx in code:
        cover_count += M[idx]
        
    improved = True
    while improved and time.monotonic() < deadline - 2.0:
        improved = False
        # Shuffle to avoid bias
        np.random.shuffle(code)
        
        for i in range(len(code)):
            if time.monotonic() > deadline - 2.0:
                break
                
            idx = code[i]
            # Check if removable: all words covered by idx must be covered by > 1 codeword
            # i.e., min(cover_count[M[idx]]) > 1
            # M[idx] is a boolean mask (converted to uint8, so 0/1)
            # We need to check cover_count at positions where M[idx] == 1
            # Since M is uint8, M[idx] is a vector of 0s and 1s.
            # We can use indexing: cover_count[M[idx] == 1]
            # But boolean indexing creates a copy.
            # Alternative: np.min(cover_count * M[idx])? No, min will be 0 if any uncovered.
            # We want min of cover_count ONLY where M[idx] is 1.
            
            # Efficient check:
            # If there is any word w such that M[idx, w] == 1 and cover_count[w] == 1, it's not removable.
            # unique_words = (cover_count == 1) & (M[idx] == 1)
            # if np.any(unique_words): not removable
            
            # Vectorized:
            # counts_in_mask = cover_count[M[idx] == 1]
            # if np.min(counts_in_mask) > 1: removable
            
            # To avoid the boolean mask allocation overhead if possible:
            # We can precompute the list of neighbors for each codeword? 
            # No, M is dense enough.
            
            # Let's use the boolean mask approach, it's fast in NumPy.
            mask = M[idx] == 1
            if np.min(cover_count[mask]) > 1:
                # Remove
                cover_count -= M[idx]
                code.pop(i)
                improved = True
                break # Restart loop with new code

    # Final Verification and Conversion
    # Verify coverage
    final_cover = np.zeros(total, dtype=np.uint8)
    for idx in code:
        final_cover += M[idx]
    
    if not np.all(final_cover > 0):
        # Fallback if something went wrong (should not happen)
        # Use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        fallback_code = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                fallback_code.append(h + list(tail))
        return [list(cw) for cw in fallback_code]
        
    return [list(idx_to_tuple(idx)) for idx in code]
