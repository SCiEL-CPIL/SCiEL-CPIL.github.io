import random
import itertools
import time

# Fallback for popcount if not available
try:
    (1).bit_count
    def _popcount(x):
        return x.bit_count()
except AttributeError:
    def _popcount(x):
        return bin(x).count('1')

def _greedy_cover_specific(q, n, radius):
    """
    Solves the specific case q=3, n=6, radius=1 using randomized greedy set cover
    followed by local search (1-opt/2-opt) to minimize the number of codewords.
    """
    N = q ** n
    FullMask = (1 << N) - 1
    
    # Precompute masks for each word
    # Index i corresponds to the tuple of digits
    # Weight of digit at position pos (0-indexed from MSB) is q^(n-1-pos)
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    masks = [0] * N
    
    for i in range(N):
        # Reconstruct digits to find neighbors
        # Or just iterate all j? No, arithmetic is faster.
        # Digits of i in base q, MSB first
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        
        mask = 1 << i
        for pos in range(n):
            orig = digits[pos]
            weight = powers[n - 1 - pos]
            for new_d in range(q):
                if new_d == orig:
                    continue
                delta = (new_d - orig) * weight
                ni = i + delta
                mask |= (1 << ni)
        masks[i] = mask

    best_code = None
    best_size = float('inf')
    
    # Time limit check helper
    start_time = time.time()
    time_limit = 100.0  # Keep safe margin under 120s
    
    num_restarts = 100  # More restarts for better initial solutions
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit * 0.8:
            break
            
        # 1. Randomized Greedy
        uncovered = FullMask
        selected_indices = set()
        
        while uncovered:
            max_cov = 0
            candidates = []
            
            # Find word covering most uncovered bits
            for i in range(N):
                if i in selected_indices:
                    continue
                cov = _popcount(uncovered & masks[i])
                if cov > max_cov:
                    max_cov = cov
                    candidates = [i]
                elif cov == max_cov and cov > 0:
                    candidates.append(i)
            
            if not candidates:
                break # Should not happen
                
            chosen = random.choice(candidates)
            selected_indices.add(chosen)
            uncovered &= ~masks[chosen]
            
        # 2. Local Search
        # Represent S as a list for easy indexing, but use set for membership?
        # Let's use a list for S and a boolean array for in_S
        S = list(selected_indices)
        in_S = [False] * N
        for idx in S:
            in_S[idx] = True
            
        improved = True
        while improved:
            if time.time() - start_time > time_limit * 0.9:
                break
            improved = False
            
            # Try to remove an element
            for i_idx, c_in in enumerate(S):
                # Calculate union of S \ {c_in}
                # Optimization: We can precompute unions? No, S changes.
                # Just compute union.
                union_others = 0
                for c in S:
                    if c == c_in:
                        continue
                    union_others |= masks[c]
                
                gap = FullMask & ~union_others
                if gap == 0:
                    # We can remove c_in
                    S.pop(i_idx)
                    in_S[c_in] = False
                    improved = True
                    break # Restart while loop with new S
                
            if improved:
                continue
                
            # Try to swap an element
            # For each c_in in S, calculate gap if removed.
            # If gap != 0, look for c_out not in S that covers gap.
            for i_idx, c_in in enumerate(S):
                union_others = 0
                for c in S:
                    if c == c_in:
                        continue
                    union_others |= masks[c]
                
                gap = FullMask & ~union_others
                if gap == 0:
                    # Should have been caught above, but just in case
                    S.pop(i_idx)
                    in_S[c_in] = False
                    improved = True
                    break
                
                # Look for c_out
                # Optimization: Only check c_out if it covers the gap.
                # We can iterate all c_out not in S.
                found_swap = False
                # Randomize order of c_out to escape local minima?
                # For now, deterministic scan, but we can randomize S order in next iter?
                # Let's just scan.
                for c_out in range(N):
                    if in_S[c_out]:
                        continue
                    if (gap & ~masks[c_out]) == 0:
                        # c_out covers the gap
                        # Perform swap
                        S[i_idx] = c_out
                        in_S[c_in] = False
                        in_S[c_out] = True
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break # Restart while loop with new S
        
        if len(S) < best_size:
            best_size = len(S)
            best_code = S[:]
            # Optional: if we hit 73, we can stop early?
            # Lower bound is 71, best known 73.
            if best_size == 73:
                # We found the best known. Stop?
                # Maybe we can try to find 72? Unlikely.
                # Let's keep searching for a bit more to be safe, or break if time permits.
                pass

    # Convert indices back to list of lists
    if best_code is None:
        # Fallback to greedy result if local search failed completely (shouldn't happen)
        return []
        
    result = []
    for idx in best_code:
        word = []
        temp = idx
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        result.append(word)
        
    return result

def construct(q, n, radius):
    if q == 3 and n == 6 and radius == 1:
        return _greedy_cover_specific(3, 6, 1)
    else:
        # General fallback
        return _greedy_general(q, n, radius)

def _greedy_general(q, n, radius):
    # Simple greedy for general case
    N = q ** n
    if N > 10000:
        return []
        
    all_words = list(itertools.product(range(q), repeat=n))
    masks = [0] * N
    
    # Precompute powers for index calc
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
        
    for i in range(N):
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        
        mask = 0
        # Generate neighbors within radius
        # For radius 1, this is fast. For general radius, it's exponential.
        # Assuming small radius for this task.
        if radius == 1:
            mask |= (1 << i)
            for pos in range(n):
                orig = digits[pos]
                weight = powers[n - 1 - pos]
                for new_d in range(q):
                    if new_d == orig:
                        continue
                    delta = (new_d - orig) * weight
                    ni = i + delta
                    mask |= (1 << ni)
        else:
            # Brute force neighbors for general radius
            for neighbor in itertools.product(range(q), repeat=n):
                dist = sum(1 for a, b in zip(digits, neighbor) if a != b)
                if dist <= radius:
                    # Calculate neighbor index
                    ni = 0
                    for k, d in enumerate(neighbor):
                        ni += d * powers[n - 1 - k]
                    mask |= (1 << ni)
                    
        masks[i] = mask
        
    FullMask = (1 << N) - 1
    uncovered = FullMask
    selected_indices = []
    
    while uncovered:
        max_cov = 0
        candidates = []
        for i in range(N):
            if i in selected_indices:
                continue
            cov = _popcount(uncovered & masks[i])
            if cov > max_cov:
                max_cov = cov
                candidates = [i]
            elif cov == max_cov and cov > 0:
                candidates.append(i)
        
        if not candidates:
            break
            
        chosen = random.choice(candidates)
        selected_indices.append(chosen)
        uncovered &= ~masks[chosen]
        
    result = []
    for idx in selected_indices:
        word = []
        temp = idx
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        result.append(word)
        
    return result
