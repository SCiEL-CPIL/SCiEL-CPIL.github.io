import numpy as np
import time
import random

def construct(q, n, radius):
    # The task is specifically for q=3, n=6, radius=1
    # We will implement a Simulated Annealing approach with a validity penalty.
    # This allows escaping local minima of the greedy/deletion approach.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: Simple greedy with Python ints (slower but general)
        # This is rarely tested as the contract specifies the football pool problem.
        start_time = time.time()
        deadline = start_time + 110
        total_words = q ** n
        powers = [q ** i for i in range(n)]
        
        # Precompute neighbors as lists for general case
        neighbors = []
        for x in range(total_words):
            nb_set = {x}
            for i in range(n):
                digit = (x // powers[i]) % q
                for delta in range(1, radius + 1):
                    if digit - delta >= 0:
                        nb_set.add(x - delta * powers[i])
                    if digit + delta < q:
                        nb_set.add(x + delta * powers[i])
            neighbors.append(list(nb_set))
            
        all_mask = (1 << total_words) - 1
        best_code = None
        best_size = total_words
        
        for trial in range(50):
            if time.time() > deadline:
                break
            uncovered = all_mask
            code = []
            while uncovered:
                max_cov = -1
                best_cands = []
                for w in range(total_words):
                    cov = 0
                    for nb in neighbors[w]:
                        if (uncovered >> nb) & 1:
                            cov += 1
                    if cov > max_cov:
                        max_cov = cov
                        best_cands = [w]
                    elif cov == max_cov:
                        best_cands.append(w)
                if not best_cands:
                    break
                chosen = random.choice(best_cands)
                code.append(chosen)
                for nb in neighbors[chosen]:
                    uncovered &= ~(1 << nb)
                    
            if len(code) < best_size:
                best_size = len(code)
                best_code = code.copy()
                
        if best_code:
            res = []
            for idx in best_code:
                word = []
                temp = idx
                for i in range(n):
                    word.append(temp % q)
                    temp //= q
                res.append(word)
            return res
        else:
            return [[0]*n]

    # Optimized path for q=3, n=6, radius=1
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for evaluation overhead
    
    N = 729
    powers = [1, 3, 9, 27, 81, 243]
    
    # Precompute the coverage matrix C (729x729) as a numpy array of uint8
    # C[w, u] = 1 if u is in the ball of w
    C = np.zeros((N, N), dtype=np.uint8)
    for w in range(N):
        C[w, w] = 1
        for i in range(6):
            digit = (w // powers[i]) % 3
            if digit != 0:
                C[w, w - powers[i]] = 1
            if digit != 2:
                C[w, w + powers[i]] = 1
                
    # 1. Get an initial valid code using Greedy
    def greedy_initial():
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        while uncovered.any():
            counts = C.dot(uncovered)
            max_val = np.max(counts)
            candidates = np.where(counts == max_val)[0]
            chosen = np.random.choice(candidates)
            code.append(chosen)
            uncovered &= (1 - C[chosen])
        return code

    best_code = greedy_initial()
    best_size = len(best_code)
    
    # Helper to calculate uncovered points for a given code (list of indices)
    def get_uncovered_mask(code):
        if not code:
            return np.ones(N, dtype=np.uint8)
        mask = np.zeros(N, dtype=np.uint8)
        for c in code:
            mask |= C[c]
        return 1 - mask # 1 where uncovered

    # Helper to check validity and get uncovered count
    def evaluate(code):
        if not code:
            return N
        mask = np.zeros(N, dtype=np.uint8)
        for c in code:
            mask |= C[c]
        return np.sum(1 - mask)

    # Simulated Annealing
    # State: list of codeword indices
    current_code = best_code[:]
    current_uncovered = evaluate(current_code)
    current_cost = len(current_code) + 100 * current_uncovered # Penalty for invalidity
    
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a move: Add or Remove a codeword
        # Bias towards removal to reduce size
        if random.random() < 0.7:
            # Remove a random codeword
            if not current_code:
                continue
            idx_to_remove = random.randint(0, len(current_code) - 1)
            c_removed = current_code[idx_to_remove]
            
            # Tentative removal
            new_code = current_code[:idx_to_remove] + current_code[idx_to_remove+1:]
            
            # Calculate new uncovered
            # We can optimize this, but for N=729, recalculating is okay if done sparsely?
            # Actually, let's just recalculate mask.
            new_mask = np.zeros(N, dtype=np.uint8)
            for c in new_code:
                new_mask |= C[c]
            new_uncovered = np.sum(1 - new_mask)
            
            new_cost = len(new_code) + 100 * new_uncovered
            
            delta_cost = new_cost - current_cost
            
            # Accept if better or with probability exp(-delta/T)
            if delta_cost < 0 or (random.random() < np.exp(-delta_cost / T)):
                current_code = new_code
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if new_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
        else:
            # Add a random codeword (to escape local minima or repair)
            c_add = np.random.randint(0, N)
            if c_add in current_code:
                continue
            new_code = current_code + [c_add]
            
            new_mask = np.zeros(N, dtype=np.uint8)
            for c in new_code:
                new_mask |= C[c]
            new_uncovered = np.sum(1 - new_mask)
            
            new_cost = len(new_code) + 100 * new_uncovered
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or (random.random() < np.exp(-delta_cost / T)):
                current_code = new_code
                current_uncovered = new_uncovered
                current_cost = new_cost
                
                if new_uncovered == 0:
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
        
        T *= cooling
        
        # Optional: If we have a valid code, try to delete redundant ones to tighten it
        if current_uncovered == 0:
             # Quick deletion pass
             changed = True
             while changed:
                 changed = False
                 for i in range(len(current_code) - 1, -1, -1):
                     c = current_code[i]
                     # Check if c is redundant
                     # Create mask without c
                     mask_others = np.zeros(N, dtype=np.uint8)
                     for j in range(len(current_code)):
                         if j != i:
                             mask_others |= C[current_code[j]]
                     if np.sum(mask_others) == N:
                         current_code.pop(i)
                         changed = True
                         break
                 if changed:
                     current_uncovered = 0 # Still valid
                     current_cost = len(current_code)
                     if len(current_code) < best_size:
                         best_size = len(current_code)
                         best_code = current_code[:]
                     break # Restart SA loop with tighter code? Or just continue?
                     # Let's just continue the SA loop, the cost is updated.

    # Final cleanup: ensure best_code is valid
    if best_code:
        res = []
        for idx in best_code:
            word = []
            temp = int(idx)
            for i in range(n):
                word.append(temp % q)
                temp //= q
            res.append(word)
        return res
    else:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
