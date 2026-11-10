import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # But we should handle general inputs if possible, or assume the task's specific parameters.
    # The prompt says: "the evaluator passes this task's alphabet size q, length n and covering radius."
    # So we must handle general q, n, radius.
    
    if radius == 0:
        return [list(map(int, word)) for word in itertools.product(range(q), repeat=n)]
        
    total_words = q ** n
    
    # Helper to convert int to base-q tuple
    def int_to_word(x):
        return [x // (q ** i) % q for i in range(n)]
        
    def word_to_int(w):
        x = 0
        for i in range(n):
            x += w[i] * (q ** i)
        return x

    # Precompute neighbors and build coverage matrix
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                for delta in range(1, q):
                    new_w = w.copy()
                    new_w[pos] = (w[pos] + delta) % q
                    neighbors[i].append(word_to_int(new_w))
        # Remove duplicates (shouldn't be any, but safe)
        neighbors[i] = list(set(neighbors[i]))
    
    # Build coverage matrix: M[i][j] = 1 if word j is within radius of word i
    M = np.zeros((total_words, total_words), dtype=np.float64)
    for i in range(total_words):
        for j in neighbors[i]:
            M[i][j] = 1.0
    
    # Greedy Set Cover with numpy acceleration
    best_code = None
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin before 150s kill
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Pure greedy (no randomness)
    # Strategy 2+: Greedy with random tie-breaking
    
    for restart in range(500):
        if time.time() - start_time > time_limit:
            break
            
        uncovered = np.ones(total_words, dtype=np.float64)
        code = []
        
        while uncovered.sum() > 0:
            # Compute scores: for each candidate, count uncovered neighbors
            scores = M @ uncovered
            max_score = scores.max()
            
            if max_score == 0:
                break
                
            # Find all candidates with max score
            best_indices = np.where(scores == max_score)[0]
            
            # Random tie-breaking based on restart number
            if len(best_indices) == 1:
                chosen = best_indices[0]
            else:
                # Use a deterministic seed per restart for reproducibility
                rng = random.Random(restart)
                chosen = rng.choice(best_indices.tolist())
            
            code.append(chosen)
            
            # Mark neighbors as covered
            neighbors_chosen = neighbors[chosen]
            uncovered[neighbors_chosen] = 0.0
            
        if not best_code or len(code) < len(best_code):
            best_code = code
            
        if len(best_code) <= 73:  # Known best
            break
            
    if not best_code:
        # Fallback to seed if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert back to list of lists of ints
    return [int_to_word(c) for c in best_code]
