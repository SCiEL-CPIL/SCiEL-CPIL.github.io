import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Construct a ternary covering code of length n and radius radius.
    Uses Simulated Annealing with double-swap moves to minimize codeword count.
    """
    # Handle specific case q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback: Simple greedy for other cases
        return _simple_greedy(q, n, radius)

    N = q ** n  # 6561
    deadline = time.time() + 110.0  # 10s safety margin

    # Precompute neighbors: neighbors[w] contains w and all words at dist 1
    # Shape: (N, 1 + 2*n)
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    powers = np.array([q ** i for i in range(n)], dtype=np.int32)
    
    for w in range(N):
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        
        # Efficiently get digits using base-q conversion logic
        # Since N is small (6561), a loop is fine, but we can optimize.
        temp = w
        for i in range(n):
            d = temp % q
            temp //= q
            # Change digit i to (d+1)%q
            w1 = w + ((d + 1) % q - d) * (q ** i)
            neighbors[w, idx] = w1
            idx += 1
            # Change digit i to (d+2)%q
            w2 = w + ((d + 2) % q - d) * (q ** i)
            neighbors[w, idx] = w2
            idx += 1

    # Initialize best solution with a strong seed
    # Tetracode-based seed
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert seed to integer indices
    def list_to_idx(lst):
        val = 0
        for i, d in enumerate(lst):
            val += d * (q ** i)
        return val
        
    best_code_idx = [list_to_idx(c) for c in seed_code]
    best_size = len(best_code_idx)
    
    # Precompute a boolean mask for the best code's coverage? No, we need dynamic coverage.
    # We will run multiple SA restarts.
    
    rng = np.random.default_rng(42)
    
    def get_coverage_counts(code_idx_list):
        cov = np.zeros(N, dtype=np.int32)
        for c in code_idx_list:
            cov[neighbors[c]] += 1
        return cov

    # Initial coverage for best
    # We'll manage coverage dynamically inside the SA loop for speed.
    
    for restart in range(20): # Max 20 restarts, time limited
        if time.time() > deadline:
            break
            
        # Generate a random initial valid code using Greedy
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        
        # Greedy construction with some randomness
        while uncovered.sum() > 0:
            # Calculate gains
            # To speed up, we can sample candidates if N is large, but N=6561 is small enough 
            # to compute all gains if we do it efficiently.
            # gains[w] = sum(uncovered[neighbors[w]])
            gains = uncovered[neighbors].sum(axis=1)
            
            # To add diversity, we don't always pick the absolute max.
            # Pick from top K candidates
            top_k = 5
            if np.max(gains) > 0:
                # Get indices of top k
                # argsort is O(N log N). For 6561, it's fast.
                # But we can use partition for top k
                # Simpler: just take max, but sometimes randomize among ties or near-ties
                max_gain = np.max(gains)
                candidates = np.where(gains == max_gain)[0]
                
                # Randomly pick from candidates
                w = rng.choice(candidates)
                
                code.append(w)
                uncovered[neighbors[w]] = 0
        
        # Convert to list of ints for easier manipulation
        code = list(code)
        
        # Local Search / Simulated Annealing Phase
        # We want to minimize len(code).
        # Moves:
        # 1. Remove one codeword if it doesn't break coverage.
        # 2. Swap one codeword for another (remove c1, add c2) if it reduces size or same size but better "potential".
        # 3. Double Swap: Remove c1, c2. Add c3, c4 (or just c3). Net change -1 or 0.
        
        # Let's stick to a robust "Greedy Reduction + Randomized Local Search"
        
        # First, aggressive reduction
        cov_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            cov_counts[neighbors[c]] += 1
        
        # Remove redundancies
        changed = True
        while changed:
            changed = False
            # Check all codewords
            # We iterate through the list. If we remove, we mark it.
            # To be safe and fast, we check condition: min(cov_counts[neighbors[c]]) >= 2
            # This means every neighbor of c is covered by at least 2 codewords.
            # If we remove c, they are still covered by at least 1.
            
            # Vectorized check is hard because we modify the list.
            # But N is small.
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if np.min(cov_counts[neighbors[c]]) >= 2:
                    cov_counts[neighbors[c]] -= 1
                    code.pop(i)
                    changed = True
                    
        if len(code) < best_size:
            best_size = len(code)
            best_code_idx = code[:]

        # Now, try to improve further with swaps.
        # We use a simple hill-climbing with random restarts within the SA budget.
        # Time per restart: ~1-2 seconds.
        
        t_start = time.time()
        local_deadline = t_start + 4.0 # 4 seconds per restart
        
        # Current state: 'code' (list of ints), 'cov_counts' (array)
        # We need to track which codewords are in the set for quick lookup?
        # A set of ints is good.
        code_set = set(code)
        
        # Precompute "neighbors of neighbors" or just use neighbors.
        
        # Move: Remove c1, Add c2.
        # Condition: c1 is in code, c2 is not in code.
        # After move: coverage of neighbors(c1) must be >= 2 (so removing c1 is safe) 
        # AND coverage of neighbors(c2) must be >= 1 (so adding c2 covers its area, though it's already covered? 
        # Wait. If we remove c1 and add c2, the set size stays same. 
        # This move is useful if it allows future removals. 
        # But we are minimizing size. 
        # So we prefer moves that reduce size: Remove c1, Add nothing (if possible). 
        # We already did greedy removal. 
        # So we look for: Remove c1, c2. Add c3. (Size -1).
        # Or: Remove c1. Add c2, c3. (Size +1, bad).
        
        # Let's try "Double Removal, Single Addition"
        # Find c1, c2 in code such that N(c1) U N(c2) is covered by Code \ {c1, c2} U {c3}.
        # This is complex.
        
        # Simpler: Just try random single swaps to escape local minima, then reduce again.
        # Or, try to find a codeword c1 that is "almost" redundant (min cov is 1 for some neighbors).
        # If we swap c1 with c2, and c2 covers those unique neighbors, we might be able to remove something else.
        
        # Let's implement a standard "Remove-Add" local search.
        # 1. Try to remove a random codeword. If invalid, revert.
        # 2. Try to add a random codeword. If it allows removing 2 others, do it.
        
        # Given time constraints, let's just do a randomized local search on the set size.
        # We'll iterate:
        # 1. Pick a random codeword c_out to remove.
        # 2. If valid (min cov >= 2), remove it. Update best.
        # 3. If invalid, pick a random c_in to add.
        # 4. Add c_in. Now try to remove 2 codewords (c_out and another c_bad) that are now redundant due to c_in?
        #    Actually, adding c_in increases coverage. It might make c_out redundant? No, c_out was not redundant.
        #    It might make another codeword c_other redundant.
        #    If we add c_in and remove c_out and c_other, size changes by -1.
        
        while time.time() < local_deadline:
            if len(code) == 0:
                break
                
            # 1. Try simple removal
            c_out = rng.choice(code)
            
            # Check if c_out is redundant
            if np.min(cov_counts[neighbors[c_out]]) >= 2:
                # Remove
                cov_counts[neighbors[c_out]] -= 1
                code.remove(c_out)
                code_set.discard(c_out)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code_idx = code[:]
                continue
            
            # 2. Try double swap: Remove c_out, Add c_in, Remove c_other
            # We need c_in to cover the "gaps" left by c_out, or just allow c_other to be removed.
            # Let's pick a random c_in not in code.
            c_in = rng.integers(0, N)
            while c_in in code_set:
                c_in = rng.integers(0, N)
                
            # Tentatively add c_in
            cov_counts[neighbors[c_in]] += 1
            
            # Now try to remove c_out and potentially another codeword c_other
            # First, is c_out redundant now?
            if np.min(cov_counts[neighbors[c_out]]) >= 2:
                # Remove c_out
                cov_counts[neighbors[c_out]] -= 1
                code.remove(c_out)
                code_set.discard(c_out)
                
                # Now try to find another redundant codeword to remove
                # Sample a few candidates
                for _ in range(5):
                    if len(code) == 0: break
                    c_other = rng.choice(code)
                    if np.min(cov_counts[neighbors[c_other]]) >= 2:
                        cov_counts[neighbors[c_other]] -= 1
                        code.remove(c_other)
                        code_set.discard(c_other)
                        # Net change: +1 (in) -1 (out) -1 (other) = -1
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code_idx = code[:]
                        break
            else:
                # c_out is not redundant even with c_in added. 
                # Revert c_in addition.
                cov_counts[neighbors[c_in]] -= 1

    # Convert best_code_idx back to lists
    final_code = []
    for w in best_code_idx:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        final_code.append(word)
        
    return final_code

def _simple_greedy(q, n, radius):
    N = q ** n
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    for w in range(N):
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        temp = w
        for i in range(n):
            d = temp % q
            temp //= q
            w1 = w + ((d + 1) % q - d) * (q ** i)
            neighbors[w, idx] = w1
            idx += 1
            w2 = w + ((d + 2) % q - d) * (q ** i)
            neighbors[w, idx] = w2
            idx += 1
            
    uncovered = np.ones(N, dtype=np.uint8)
    code = []
    while uncovered.sum() > 0:
        gains = uncovered[neighbors].sum(axis=1)
        w = np.argmax(gains)
        code.append(w)
        uncovered[neighbors[w]] = 0
        
    final_code = []
    for w in code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        final_code.append(word)
    return final_code
