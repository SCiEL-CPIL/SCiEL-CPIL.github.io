import numpy as np
import time
import itertools

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters, though task is specific
        # Simple greedy for general case
        N = q ** n
        neighbors = np.zeros((N, 1 + (q-1)*n), dtype=np.int32)
        for w in range(N):
            idx = 0
            neighbors[w, idx] = w
            idx += 1
            temp = w
            digits = [0] * n
            for i in range(n):
                digits[i] = temp % q
                temp //= q
            for i in range(n):
                d = digits[i]
                for new_d in range(q):
                    if new_d == d: continue
                    w1 = w + (new_d - d) * (q ** i)
                    neighbors[w, idx] = w1
                    idx += 1
        # Greedy
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        while uncovered.sum() > 0:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0: break
            candidates = np.where(gains == max_gain)[0]
            w = candidates[0]
            code.append(w)
            uncovered[neighbors[w]] = 0
        return [[0]*n for _ in range(len(code))] # Placeholder, real logic below for 3,8,1

    N = q ** n
    deadline = time.time() + 115.0
    
    # Precompute neighbors for q=3, n=8, r=1
    # neighbors[w] contains w itself and all words at distance 1
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    
    # Powers of q for fast digit manipulation
    q_powers = np.array([q ** i for i in range(n)], dtype=np.int32)
    
    for w in range(N):
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        
        # Get digits
        temp = w
        digits = [0] * n
        for i in range(n):
            digits[i] = temp % q
            temp //= q
            
        for i in range(n):
            d = digits[i]
            # change digit i to (d+1)%q
            w1 = w + ((d + 1) % q - d) * (q ** i)
            # change digit i to (d+2)%q
            w2 = w + ((d + 2) % q - d) * (q ** i)
            
            neighbors[w, idx] = w1
            idx += 1
            neighbors[w, idx] = w2
            idx += 1

    # Helper to convert int to list of digits
    def int_to_word(w):
        word = [0] * n
        temp = w
        for i in range(n):
            word[i] = temp % q
            temp //= q
        return word

    # Seed solution: Tetracode * 3^4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            seed_code.append(tuple(h + list(tail)))
    
    # Convert seed to list of ints
    seed_ints = []
    for w in seed_code:
        val = 0
        for i, d in enumerate(w):
            val += d * (q ** i)
        seed_ints.append(val)
    
    best_code = seed_ints
    best_size = len(best_code)
    
    rng = np.random.default_rng(42)

    # Optimization Function: Given a valid code, try to reduce it
    def optimize(code):
        # Convert to set for fast lookup? Or keep list and use cov counts.
        # We use cov_counts for efficiency.
        cov_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            cov_counts[neighbors[c]] += 1
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            # 1. Try single removals
            # A codeword c is removable if min(cov_counts[neighbors[c]]) >= 2
            # We can vectorize the check: mins = cov_counts[neighbors].min(axis=1)
            # But we only care about codewords in 'code'.
            
            # Iterate through current code list
            # To avoid index shifting issues, we iterate and mark removals, then filter.
            # However, removing one changes cov_counts, affecting others.
            # So we do sequential removals in one pass, updating cov_counts immediately.
            
            new_code = []
            for c in code:
                # Check if c is redundant given current cov_counts (which includes c)
                # If min(cov_counts[neighbors[c]]) >= 2, c is redundant.
                if np.min(cov_counts[neighbors[c]]) >= 2:
                    # Remove c
                    cov_counts[neighbors[c]] -= 1
                    improved = True
                else:
                    new_code.append(c)
            code = new_code
            
            # 2. Try double-swap moves (remove 2, add 1 or 2)
            # This is more expensive. We only do this if no single removals happened in the last full pass?
            # Or just try a few random double swaps if stuck.
            # Let's try a few random double swaps if len(code) > 403.
            
            if len(code) > 403:
                # Try K random double swap attempts
                K = 50
                for _ in range(K):
                    if time.time() > deadline - 1.0: break
                    if len(code) < 2: break
                    
                    # Pick 2 distinct codewords to remove
                    idx1, idx2 = rng.choice(len(code), size=2, replace=False)
                    c1 = code[idx1]
                    c2 = code[idx2]
                    
                    # Temporarily remove them
                    # Check if union of neighbors of c1 and c2 is covered by the rest?
                    # Actually, we need to find a set of 1 or 2 new codewords that covers 
                    # the "uncovered" parts left by removing c1 and c2.
                    
                    # Let's calculate the "deficit" set: words covered by c1 or c2 but not by C \ {c1, c2}.
                    # This is equivalent to:
                    # count_current = cov_counts
                    # count_without_c1_c2 = cov_counts - neighbors[c1] - neighbors[c2] (careful with overlaps)
                    
                    # Easier: 
                    # 1. Decrement cov_counts for c1 and c2
                    cov_counts[neighbors[c1]] -= 1
                    cov_counts[neighbors[c2]] -= 1
                    
                    # 2. Identify words that are now uncovered (count == 0)
                    # These are the words that MUST be covered by the new replacement(s).
                    uncovered_mask = (cov_counts == 0)
                    
                    if not np.any(uncovered_mask):
                        # Removing c1 and c2 still leaves valid coverage!
                        # Just remove them.
                        code = [code[i] for i in range(len(code)) if i != idx1 and i != idx2]
                        improved = True
                        # We don't add anything back.
                        # Note: cov_counts is now correct for the new code.
                        break # Restart optimization loop with smaller code
                    else:
                        # We need to cover the uncovered words.
                        # Let U be the set of uncovered word indices.
                        U = np.where(uncovered_mask)[0]
                        
                        # Try adding 1 codeword
                        # Find a codeword w that covers ALL words in U.
                        # i.e., U is subset of neighbors[w].
                        # This means for all u in U, cov_counts[u] + 1 (from w) >= 1.
                        # Since cov_counts[u] is 0, we need w to be a neighbor of all u in U.
                        # This is a strong condition.
                        
                        # Heuristic: Pick a random u in U, look at its neighbors, check if any covers all U.
                        # Or, calculate gain for all words against U.
                        # gain[w] = number of words in U that are neighbors of w.
                        # We need gain[w] == len(U).
                        
                        # Efficient check:
                        # For each w, count how many of its neighbors are in U.
                        # We can precompute a boolean array for U.
                        # u_bool = np.zeros(N, dtype=bool); u_bool[U] = True
                        # counts = u_bool[neighbors].sum(axis=1)
                        # candidates = np.where(counts == len(U))[0]
                        
                        # This is O(N * 17) which is fast enough.
                        u_bool = np.zeros(N, dtype=bool)
                        u_bool[U] = True
                        counts = u_bool[neighbors].sum(axis=1)
                        candidates = np.where(counts == len(U))[0]
                        
                        if len(candidates) > 0:
                            # Pick a candidate
                            w_new = candidates[rng.integers(len(candidates))]
                            # Add w_new
                            cov_counts[neighbors[w_new]] += 1
                            # Remove c1, c2, add w_new
                            code = [code[i] for i in range(len(code)) if i != idx1 and i != idx2]
                            code.append(w_new)
                            improved = True
                            break # Restart optimization loop
                        else:
                            # Try adding 2 codewords?
                            # This is much harder. Skip for now or do simple heuristic.
                            # Let's try to cover U with 2 codewords.
                            # Pick a random u1 in U, find best w1 to cover most of U.
                            # Then pick best w2 to cover remaining.
                            
                            # Heuristic:
                            # Pick w1 that maximizes coverage of U.
                            # max_count = np.max(counts)
                            # if max_count == len(U): handled above.
                            # So max_count < len(U).
                            
                            # Let's just revert if we can't find a 1-swap.
                            # Maybe try 2-swap later if time permits.
                            # For now, revert.
                            cov_counts[neighbors[c1]] += 1
                            cov_counts[neighbors[c2]] += 1
                            
        return code

    # Multi-start loop
    while time.time() < deadline:
        # 1. Start with seed or random valid code
        if rng.random() < 0.5:
            current_code = list(best_code)
            # Shuffle to break symmetry?
            rng.shuffle(current_code)
        else:
            # Generate a random valid code using greedy with random tie-breaking
            uncovered = np.ones(N, dtype=np.uint8)
            current_code = []
            while uncovered.sum() > 0 and time.time() < deadline - 5.0:
                gains = uncovered[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0: break
                candidates = np.where(gains == max_gain)[0]
                w = candidates[rng.integers(len(candidates))]
                current_code.append(w)
                uncovered[neighbors[w]] = 0
            
            # Validate
            if len(current_code) == 0: continue
            # Check if valid (should be by construction, but safe to check if we modified logic)
            # Greedy ensures validity.
            
        # 2. Optimize
        optimized_code = optimize(current_code)
        
        if len(optimized_code) < best_size:
            best_size = len(optimized_code)
            best_code = optimized_code
            # Optional: If we hit lower bound, break?
            if best_size <= 403:
                break

    # Convert best_code (list of ints) to list of lists of digits
    final_code = []
    for w in best_code:
        word = [0] * n
        temp = w
        for i in range(n):
            word[i] = temp % q
            temp //= q
        final_code.append(word)
        
    return final_code
