import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: aim to finish by 115s
    deadline = start_time + 115.0
    
    total_words = q ** n
    
    # 1. Precompute neighbors (balls) for each word
    # For radius=1, ball size is 1 + n*(q-1)
    # We represent words as integers 0..total_words-1
    
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    def tuple_to_int(t, q):
        x = 0
        for i in range(n-1, -1, -1):
            x = x * q + t[i]
        return x

    # Precompute balls
    balls = [None] * total_words
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        if radius >= 1:
            for pos in range(n):
                orig_val = t_i[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    t_new = t_i[:]
                    t_new[pos] = new_val
                    ball.append(tuple_to_int(t_new, q))
        balls[i] = ball
        
    # Coverage matrix M: M[i, j] = 1 if j is in balls[i]
    M = np.zeros((total_words, total_words), dtype=np.uint8)
    for i in range(total_words):
        M[i, balls[i]] = 1

    def is_valid(codewords):
        if len(codewords) == 0:
            return total_words == 0
        if len(codewords) == 1:
            return np.all(M[codewords[0], :] >= 1)
        coverage_counts = M[codewords, :].sum(axis=0)
        return np.all(coverage_counts >= 1)

    def greedy_cover():
        """Standard greedy set cover to get a valid initial solution."""
        covered = np.zeros(total_words, dtype=bool)
        initial_codewords = []
        
        while True:
            if time.time() > deadline - 10:
                break
            uncovered_mask = ~covered
            if not np.any(uncovered_mask):
                break
            # Scores: number of currently uncovered words covered by each candidate
            scores = M.dot(uncovered_mask.astype(np.uint8))
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            initial_codewords.append(best_idx)
            covered[balls[best_idx]] = True
            
        return initial_codewords

    def local_search_remove(codewords):
        """
        Iteratively remove redundant codewords.
        A codeword is redundant if removing it does not uncover any word.
        We use a "First-Fit-Decreasing" style approach or just iterate until stable.
        To be faster, we check all and remove any that are redundant.
        """
        if not codewords:
            return codewords
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Compute coverage counts
            cover_count = M[codewords, :].sum(axis=0)
            
            # Identify redundant indices
            # A codeword i is redundant if min(cover_count[balls[i]]) > 1
            # Vectorized check:
            # For each codeword i, check if all words in its ball have count > 1
            # This is slightly expensive if done naively per codeword in a loop,
            # but len(codewords) is ~180, total_words is 2187.
            # M[codewords, :] is (K, 2187).
            
            for i in range(len(codewords)):
                c = codewords[i]
                # Check if removing c leaves all words in balls[c] covered by others
                # i.e., cover_count[word] > 1 for all word in balls[c]
                if np.min(cover_count[balls[c]]) > 1:
                    codewords.pop(i)
                    changed = True
                    break # Restart loop to update cover_count efficiently? 
                         # Actually, updating cover_count incrementally is better, 
                         # but recomputing is O(K*N) = 180*2187 ~ 400k ops, which is fast.
                         # Let's just break and recompute to be safe and simple.
        
        return codewords

    def error_repair(codewords):
        """
        If codewords is invalid, try to repair it by adding the single best codeword
        that covers the most uncovered words. This is a targeted repair vs random addition.
        Returns repaired list or None if it couldn't be made valid quickly.
        """
        if is_valid(codewords):
            return codewords
        
        # Find uncovered
        cover_count = np.zeros(total_words, dtype=np.uint8)
        for c in codewords:
            cover_count[balls[c]] += 1
        
        uncovered_mask = cover_count == 0
        if not np.any(uncovered_mask):
            return codewords # Should be valid
        
        # Find best candidate to add
        # Score: number of uncovered words it covers
        scores = M.dot(uncovered_mask.astype(np.uint8))
        
        # We might need to add more than one if one isn't enough? 
        # For radius 1, adding one might not fix everything if we removed many.
        # But in ILS perturbation, we usually remove 1 and add 1.
        # If we remove 1, the invalidity is small.
        
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            return None # Can't fix with one addition
        
        new_codewords = codewords + [best_idx]
        
        # Check if valid now
        if is_valid(new_codewords):
            return new_codewords
        
        # If not valid, fallback to greedy repair
        covered = cover_count > 0
        covered[balls[best_idx]] = True
        while True:
            if time.time() > deadline - 2:
                return None
            uncovered_mask = ~covered
            if not np.any(uncovered_mask):
                break
            scores = M.dot(uncovered_mask.astype(np.uint8))
            idx = np.argmax(scores)
            if scores[idx] == 0:
                break
            new_codewords.append(idx)
            covered[balls[idx]] = True
            
        return new_codewords

    # 2. Initial Solution
    best_codewords = greedy_cover()
    best_codewords = local_search_remove(best_codewords)
    best_size = len(best_codewords)
    
    current_codewords = best_codewords[:]
    
    # 3. Iterated Local Search (ILS)
    # Strategy: 
    # 1. Perturb current solution (remove k, add k random)
    # 2. Try to repair if invalid (using error_repair)
    # 3. Local search (remove redundant)
    # 4. If better, accept. If not, accept with probability (simulated annealing) or just reject?
    #    Let's use simple hill climbing first, maybe add SA later if time permits.
    #    Actually, ILS usually accepts worse solutions to escape local optima.
    #    Let's implement a simple SA-like acceptance.
    
    T = 5.0 # Initial temperature
    cooling = 0.995
    
    while time.time() < deadline - 10:
        # Perturbation
        child = current_codewords[:]
        
        # Remove 1 to 3 codewords
        num_remove = np.random.randint(1, 4)
        if num_remove > len(child):
            num_remove = len(child)
        
        for _ in range(num_remove):
            if not child: break
            idx = np.random.randint(len(child))
            child.pop(idx)
        
        # Add 1 to 3 random codewords
        num_add = np.random.randint(1, 4)
        for _ in range(num_add):
            new_c = np.random.randint(total_words)
            if new_c not in child:
                child.append(new_c)
        
        # Repair if invalid
        if not is_valid(child):
            repaired = error_repair(child)
            if repaired is None:
                # Repair failed, discard
                continue
            child = repaired
        
        # Local Search
        child = local_search_remove(child)
        
        # Evaluate
        child_size = len(child)
        
        # Acceptance
        if child_size < best_size:
            best_size = child_size
            best_codewords = child[:]
            current_codewords = child[:]
            T = max(T * cooling, 0.1) # Reset/cool temp? Or keep cooling?
            # Let's keep cooling but maybe reset if we hit a new best?
            # Standard ILS often just accepts if better.
        else:
            # Simulated Annealing acceptance
            delta = child_size - len(current_codewords)
            if delta < 0:
                current_codewords = child[:]
            else:
                # Accept with probability exp(-delta / T)
                if np.random.rand() < np.exp(-delta / T):
                    current_codewords = child[:]
        
        T = max(T * cooling, 0.01)
        
        # If T is very low, we are basically doing hill climbing.
        # This helps escape local optima initially.

    # Final Pruning
    best_codewords = local_search_remove(best_codewords)
    
    # Convert to output format
    result = []
    for c in best_codewords:
        result.append(int_to_tuple(c, n, q))
        
    return result
