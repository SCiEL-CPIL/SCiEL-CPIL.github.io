import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # For the specific task: q=3, n=6, radius=1
    # Best known: 73 (van Laarhoven et al. 1989), lower bound 71.
    # Seed: direct sum of tetracode (n=4) and full product on last 2 coords -> 9 * 9 = 81 words.
    
    base = q
    total_words = base ** n
    
    # Precompute powers for index calculation
    powers = np.array([base ** i for i in range(n)], dtype=np.int64)
    
    def word_to_idx(word):
        return int(np.dot(word, powers))
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % base
            idx //= base
        return w
    
    def get_ball_indices(word):
        """Return list of indices of all words within Hamming distance <= radius (1)."""
        idx = word_to_idx(word)
        neighbors = [idx]
        for i in range(n):
            for val in range(base):
                if val != word[i]:
                    neighbors.append(idx + (val - word[i]) * powers[i])
        return neighbors

    # Generate initial code: tetracode direct sum with full tail
    # Tetracode over GF(3): [a, b, a+b, a+2b] mod 3
    tetracode = [[a, b, (a + b) % base, (a + 2 * b) % base] for a in range(base) for b in range(base)]
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(base), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    # Convert to numpy arrays for faster operations
    # Represent code as list of word-index integers for compactness
    initial_indices = [word_to_idx(w) for w in initial_code]
    
    best_code = None
    best_size = float('inf')
    
    # Strategy: 
    # 1. Start from 81-word seed.
    # 2. Use greedy removal: repeatedly find and remove redundant codewords.
    # 3. Use random restarts with shuffled order to escape local minima.
    # 4. Also try a "swap" heuristic: if stuck, try swapping a codeword with an uncovered-adjacent word.
    
    # For speed, precompute the ball (neighborhood) for each possible index? 
    # 3^6 = 729 indices, each ball has 1 + 6*2 = 13 words.
    # Precomputing all balls: 729 * 13 = 9477 entries. Very fast.
    all_balls = {}
    for idx in range(total_words):
        w = idx_to_word(idx)
        all_balls[idx] = get_ball_indices(w)
    
    def try_minimize(code_indices, time_limit):
        """Greedy removal + optional repair. Returns minimized list of indices."""
        code = list(code_indices)
        # Build coverage count
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for nb in all_balls[c]:
                cover_count[nb] += 1
        
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            random.shuffle(code)
            i = 0
            while i < len(code):
                c = code[i]
                # Check if redundant: all neighbors covered >= 2 times
                redundant = True
                for nb in all_balls[c]:
                    if cover_count[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    for nb in all_balls[c]:
                        cover_count[nb] -= 1
                    code.pop(i)
                    changed = True
                else:
                    i += 1
        return code

    # Phase 1: Multiple restarts of greedy removal
    max_restarts = 200
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        shuffled = list(initial_indices)
        random.shuffle(shuffled)
        minimized = try_minimize(shuffled, deadline - 10)
        
        if len(minimized) < best_size:
            best_size = len(minimized)
            best_code = [idx_to_word(c) for c in minimized]
        
        if best_size <= 73:
            break
    
    # Phase 2: If we haven't reached 73, try a more aggressive search.
    # Use a "local search" with swaps: 
    # For each codeword, try replacing it with a random word that is not in the code,
    # and check if the new code is still valid and has the same size.
    # This doesn't reduce size, so it's not directly useful for minimization.
    # Instead, try: remove a word, then greedily add back the "most useful" word to restore coverage.
    
    if best_size > 73 and time.time() < deadline - 20:
        # Start from the best code found so far
        current = [idx_to_word(c) for c in (best_code if best_code else initial_code)]
        current_indices = [word_to_idx(w) for w in current]
        
        # Try to reduce further with a more careful greedy:
        # Instead of random shuffle, try removing in order of "least coverage" first.
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            code = list(current_indices)
            cover_count = np.zeros(total_words, dtype=np.int32)
            for c in code:
                for nb in all_balls[c]:
                    cover_count[nb] += 1
            
            # Sort by how "redundant" they are: prefer removing words whose neighbors are most covered
            def redundancy_score(c):
                # Lower score = more redundant (all neighbors well-covered)
                min_cov = min(cover_count[nb] for nb in all_balls[c])
                return min_cov
            
            code_sorted = sorted(code, key=redundancy_score)
            
            changed = True
            while changed and time.time() < deadline - 20:
                changed = False
                for i in range(len(code_sorted)):
                    c = code_sorted[i]
                    if c not in code:
                        continue
                    redundant = True
                    for nb in all_balls[c]:
                        if cover_count[nb] <= 1:
                            redundant = False
                            break
                    if redundant:
                        for nb in all_balls[c]:
                            cover_count[nb] -= 1
                        code.remove(c)
                        code_sorted.remove(c)
                        changed = True
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = [idx_to_word(c) for c in code]
                current_indices = code
            else:
                # Perturb: randomly swap one codeword
                if len(code) > 0:
                    idx_to_swap = random.randint(0, len(code) - 1)
                    old_c = code[idx_to_swap]
                    new_c = random.randint(0, total_words - 1)
                    if new_c != old_c and new_c not in code:
                        code[idx_to_swap] = new_c
                        current_indices = code
    
    if best_code is None:
        return initial_code
    
    return best_code
