import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        return _solve_k3_7_1()
    
    # Fallback for other cases (keep original logic)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    return initial_code

def _solve_k3_7_1():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    q = 3
    n = 7
    total_words = q ** n  # 2187
    
    # Precompute neighbors for all words
    # neighbors[i] is a list of indices j such that dist(i, j) <= 1
    neighbors = []
    for i in range(total_words):
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        
        nb = [i]
        for k in range(n):
            for v in range(q):
                if v != digits[k]:
                    new_val = i + (v - digits[k]) * (q ** k)
                    nb.append(new_val)
        neighbors.append(nb)
    
    # Convert to numpy arrays for faster indexing in reduction phase
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]

    # 1. Greedy Construction
    # We maintain a set of uncovered words.
    # At each step, we pick an uncovered word u.
    # We consider all candidates c that cover u (i.e., u in neighbors[c] <=> c in neighbors[u]).
    # We pick the c that covers the most *currently uncovered* words.
    
    uncovered = set(range(total_words))
    code = []
    coverage_count = np.zeros(total_words, dtype=np.int32)
    
    while uncovered and time.time() < deadline:
        # Pick an uncovered word. 
        # Heuristic: Pick the one that is "hardest" to cover? 
        # Or just pick the first one. Picking the first one is fast.
        # To improve quality, we could pick the uncovered word with the fewest candidate codewords?
        # But all words have the same number of candidates (15).
        # So any uncovered word is fine.
        u = next(iter(uncovered))
        
        best_c = -1
        max_new_covered = 0
        
        # Candidates are neighbors of u
        for c in neighbors[u]:
            # Count how many words in neighbors[c] are in uncovered
            # This is a set intersection size
            # len(set(neighbors[c]) & uncovered)
            # Since neighbors[c] is small (15), and uncovered is a set, this is fast.
            new_covered = 0
            for w in neighbors[c]:
                if w in uncovered:
                    new_covered += 1
            
            if new_covered > max_new_covered:
                max_new_covered = new_covered
                best_c = c
                if new_covered == len(neighbors[c]): # Perfect coverage of the ball
                    break
        
        if best_c == -1:
            break # Should not happen
            
        code.append(best_c)
        
        # Update coverage and uncovered set
        for w in neighbors[best_c]:
            coverage_count[w] += 1
            if w in uncovered:
                uncovered.remove(w)
                
    # 2. Reduction Phase
    # Try to remove redundant codewords.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code)):
            c = code[i]
            # If every word in the ball of c is covered by at least 2 codewords, 
            # then c is redundant (one of the others is the "second" coverer).
            # Actually, if min(coverage_count[ball[c]]) > 1, then removing c leaves min count >= 1.
            if np.min(coverage_count[neighbors_np[c]]) > 1:
                # Remove c
                for w in neighbors_np[c]:
                    coverage_count[w] -= 1
                code.pop(i)
                changed = True
                break # Restart to handle index shifts and new redundancies
                
    # 3. Optional: Local Search / Exchange
    # If we have time, try to improve by swapping.
    # Strategy: Pick a random codeword c. Try to replace it with a random word w.
    # If the new code is valid and allows removing another codeword, do it.
    # This is complex to implement correctly and fast. 
    # Given the deadline, and that Greedy+Reduction is robust, we might skip complex exchanges 
    # or do a very simple one.
    
    # Simple Exchange: 
    # Try to find a pair (c, c') in code and a word w not in code such that:
    # 1. Removing c and c' leaves some words uncovered (the "holes").
    # 2. Adding w covers all holes.
    # 3. Net change: -1 codeword.
    # This is expensive to search for.
    
    # Alternative: Randomized Restart of Greedy with different tie-breaking?
    # The current greedy is deterministic. 
    # We can run a few random restarts if time permits.
    
    best_code = code.copy()
    best_size = len(code)
    
    # If we have > 30 seconds left, do some random restarts of the greedy phase
    # to see if we get a better initial code.
    if time.time() < deadline - 30:
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            # Randomize the choice of 'u' or the tie-breaking in greedy
            # Let's just shuffle the order of candidates in greedy
            uncovered = set(range(total_words))
            code_trial = []
            coverage_count_trial = np.zeros(total_words, dtype=np.int32)
            
            while uncovered and time.time() < deadline - 10:
                u = random.choice(list(uncovered))
                best_c = -1
                max_new_covered = 0
                candidates = neighbors[u][:]
                random.shuffle(candidates)
                
                for c in candidates:
                    new_covered = 0
                    for w in neighbors[c]:
                        if w in uncovered:
                            new_covered += 1
                    if new_covered > max_new_covered:
                        max_new_covered = new_covered
                        best_c = c
                        if new_covered == len(neighbors[c]):
                            break
                
                if best_c == -1:
                    break
                
                code_trial.append(best_c)
                for w in neighbors[best_c]:
                    coverage_count_trial[w] += 1
                    if w in uncovered:
                        uncovered.remove(w)
                        
            # Reduction for trial
            changed = True
            while changed and time.time() < deadline - 10:
                changed = False
                for i in range(len(code_trial)):
                    c = code_trial[i]
                    if np.min(coverage_count_trial[neighbors_np[c]]) > 1:
                        for w in neighbors_np[c]:
                            coverage_count_trial[w] -= 1
                        code_trial.pop(i)
                        changed = True
                        break
                        
            if len(code_trial) < best_size:
                best_size = len(code_trial)
                best_code = code_trial.copy()

    # Convert best_code back to list of lists
    result = []
    for w in best_code:
        digits = []
        tmp = w
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        result.append(digits)
        
    return result
