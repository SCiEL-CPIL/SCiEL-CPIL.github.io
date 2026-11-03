import numpy as np
import time

def construct(q, n, radius):
    total = q ** n
    # Precompute neighbors for all words
    # neighbors[i] is a list of indices within Hamming distance 1
    neighbors = [[] for _ in range(total)]
    powers = [q ** (n - 1 - pos) for pos in range(n)]
    
    for i in range(total):
        neighbors[i].append(i)
        digits = [0] * n
        tmp = i
        for pos in range(n - 1, -1, -1):
            digits[pos] = tmp % q
            tmp //= q
        
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_idx = i + (new_val - orig_val) * powers[pos]
                neighbors[i].append(new_idx)
                
    # Convert neighbors to numpy arrays for faster indexing?
    # Or just use lists.
    
    # 1. Initial Greedy to get a valid cover
    # We will use a fast randomized greedy
    in_code = np.zeros(total, dtype=bool)
    counts = np.zeros(total, dtype=np.int32)
    S = []
    uncovered_mask = np.ones(total, dtype=bool)
    
    # Fast Greedy:
    # To avoid O(N^2) complexity, we can pick a random uncovered word and a random neighbor.
    # This is very fast.
    rng = np.random.default_rng(42)
    
    while uncovered_mask.any():
        # Find an uncovered word
        # Using np.nonzero is fast
        uncov_indices = np.nonzero(uncovered_mask)[0]
        w = uncov_indices[rng.integers(len(uncov_indices))]
        
        # Pick a random neighbor of w
        nbs = neighbors[w]
        c = nbs[rng.integers(len(nbs))]
        
        if not in_code[c]:
            in_code[c] = True
            S.append(c)
            # Update counts
            counts[neighbors[c]] += 1
            # Update uncovered mask
            uncovered_mask[neighbors[c]] = False
            
    # 2. Minimization Phase
    start_time = time.time()
    time_limit = 110.0
    
    # Convert S to a list for iteration
    # We will maintain S as a list, but also use in_code for O(1) lookup
    
    while time.time() - start_time < time_limit:
        # Shuffle S to randomize removal order
        rng.shuffle(S)
        
        # Attempt to remove elements
        for c in S:
            if not in_code[c]:
                continue
                
            # Try to remove c
            # Decrement counts
            counts[neighbors[c]] -= 1
            
            # Check if any count is 0
            if np.min(counts) == 0:
                # Revert
                counts[neighbors[c]] += 1
            else:
                # Keep removed
                in_code[c] = False
                # Note: We don't remove from S immediately to keep iteration simple,
                # but we can compact S later or just check in_code.
                
        # Compact S occasionally? Or just let it grow with False entries?
        # It's better to compact to keep the shuffle fast.
        S = [c for c in S if in_code[c]]
        
        # Attempt Swaps (Local Search)
        # Try a few random swaps
        if len(S) > 0:
            for _ in range(1000): # Do some swaps per pass
                if time.time() - start_time > time_limit:
                    break
                if len(S) < 2:
                    break
                # Pick c in S
                c = S[rng.integers(len(S))]
                if not in_code[c]:
                    continue
                # Pick c' not in S
                # We need a list of candidates not in S.
                # Generating this list is O(N).
                # Instead, pick a random index c_prime and check if it's in S.
                # If it is, retry.
                # Since S is small (~400) and N=6561, probability of collision is ~6%.
                # So we can just pick random c_prime.
                
                c_prime = rng.integers(total)
                if in_code[c_prime]:
                    continue
                    
                # Try swap: remove c, add c_prime
                counts[neighbors[c]] -= 1
                counts[neighbors[c_prime]] += 1
                
                if np.min(counts) >= 1:
                    in_code[c] = False
                    in_code[c_prime] = True
                    # Update S list
                    # Remove c, add c_prime
                    # This is O(N) for list ops.
                    # We can defer compaction.
                    # But S is used for picking c.
                    # Let's just mark and compact after swaps.
                else:
                    counts[neighbors[c]] += 1
                    counts[neighbors[c_prime]] -= 1
                    
        # Compact S
        S = [c for c in S if in_code[c]]
        # Add any new ones?
        # in_code might have True for indices not in S (from swaps).
        # We need to ensure S contains ALL indices where in_code is True.
        # So:
        S = [i for i in range(total) if in_code[i]]
        
    # 3. Convert to output format
    codewords = []
    for idx in S:
        digits = [0] * n
        tmp = idx
        for pos in range(n - 1, -1, -1):
            digits[pos] = tmp % q
            tmp //= q
        codewords.append(digits)
        
    return codewords
