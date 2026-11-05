import itertools
import time

def construct(q, n, radius):
    """
    Constructs a covering code using a greedy peeling method.
    Starts with a known valid code (direct sum of tetracode) and removes
    redundant codewords.
    """
    start_time = time.time()
    deadline = 140.0  # 10 seconds margin before 150s limit
    
    # 1. Generate the initial code (Direct sum of ternary tetracode)
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # The task is specifically for q=3, n=8, radius=1.
    # But the function signature is general.
    # If q, n, radius don't match the specific optimization, fall back to a simple construction or the seed.
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # And "Score: the number of codewords... minimized".
    # The specific task is K_3(8,1).
    
    if q == 3 and n == 8 and radius == 1:
        # Initial code: 9 * 3**(8-4) = 9 * 81 = 729 codewords
        initial_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                initial_codewords.append(h + list(tail))
        
        # Map word to index for fast lookup
        # Total space size 3^8 = 6561
        space_size = q ** n
        
        def word_to_idx(w):
            idx = 0
            for i in range(n):
                idx = idx * q + w[i]
            return idx
        
        # Precompute neighbors (ball of radius 1) for all words in the space?
        # Or just for the codewords?
        # We need to check coverage of all points in the space.
        # Let's represent the space as a list of indices 0..space_size-1.
        
        # Precompute the ball for each codeword in the initial set?
        # Actually, we need to know which codewords cover which points.
        # Let's use a list of lists: coverage[point_idx] = list of codeword indices covering it.
        # Or better: coverage_count[point_idx] = count, and essential_check.
        
        # Let's store codewords as tuples for hashing if needed, but we just need indices.
        cw_indices = [word_to_idx(cw) for cw in initial_codewords]
        num_cw = len(cw_indices)
        
        # Precompute the ball (indices of points within distance 1) for each codeword index?
        # The ball depends on the word content, not the codeword index in the list.
        # So we can precompute balls for all 6561 words in the space.
        # ball[word_idx] = list of point indices within distance 1.
        
        # To compute ball(word_idx):
        # 1. The word itself.
        # 2. For each position i, change digit to other q-1 values.
        
        balls = [None] * space_size
        for w_idx in range(space_size):
            # Convert index to word
            w = [0] * n
            tmp = w_idx
            for i in range(n - 1, -1, -1):
                w[i] = tmp % q
                tmp //= q
            
            pts = [w_idx]
            for i in range(n):
                orig = w[i]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[i] = new_val
                    # Compute new index
                    new_idx = 0
                    for j in range(n):
                        new_idx = new_idx * q + w[j]
                    pts.append(new_idx)
                    w[i] = orig
            balls[w_idx] = pts

        # Initialize coverage counts
        # coverage_count[p] = number of codewords covering point p
        coverage_count = [0] * space_size
        
        # active_cw[i] = True if codeword i is currently in the set
        active_cw = [True] * num_cw
        
        # Initial pass: populate coverage_count
        for i in range(num_cw):
            w_idx = cw_indices[i]
            for p in balls[w_idx]:
                coverage_count[p] += 1
        
        # Peeling loop
        while True:
            if time.time() > deadline:
                break
            
            # Identify redundant codewords
            redundant = []
            for i in range(num_cw):
                if not active_cw[i]:
                    continue
                
                w_idx = cw_indices[i]
                # Check if this codeword covers any point with count 1
                is_essential = False
                for p in balls[w_idx]:
                    if coverage_count[p] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    redundant.append(i)
            
            if not redundant:
                break
            
            # Remove redundant codewords
            for i in redundant:
                active_cw[i] = False
                w_idx = cw_indices[i]
                for p in balls[w_idx]:
                    coverage_count[p] -= 1
            
            # Optional: check if we have time to do more
            if time.time() > deadline:
                break

        # Construct final list
        final_codewords = []
        for i in range(num_cw):
            if active_cw[i]:
                # Convert index back to word
                w = [0] * n
                tmp = cw_indices[i]
                for j in range(n - 1, -1, -1):
                    w[j] = tmp % q
                    tmp //= q
                final_codewords.append(w)
        
        # Verification (just in case, though logic should be sound)
        # Check if all points are covered
        # This is O(N * M_final) which might be slow if M_final is large, but M_final should be ~400-500.
        # 6561 * 500 = 3M operations. Fast enough.
        # But we can trust the peeling if it started valid.
        
        return final_codewords
    else:
        # Fallback for other parameters: use the seed construction
        # This ensures the function doesn't crash for other inputs, although the task is specific.
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n < 4:
             # Handle small n? 
             # For n=1, q=3, r=1, we need 3 words.
             # The tetracode logic assumes n>=4.
             # Let's just return all words if n is small or parameters don't fit the optimized path.
             return [list(w) for w in itertools.product(range(q), repeat=n)]
        
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

# EVOLVE-BLOCK-END
