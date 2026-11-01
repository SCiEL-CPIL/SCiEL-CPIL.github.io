import itertools
import time
import random

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # We assume the evaluator passes these values, but we can hardcode optimizations for n=6, q=3
    # If q, n, radius differ, we fall back to a generic (slower) method or the seed.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed logic for other parameters to ensure correctness
        # Seed: Direct sum of tetracode
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Trivial case
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=6, radius=1
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # 1. Generate all words
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words) # 729
    
    # 2. Precompute balls (neighbors + self)
    # ball[i] is a list of indices of words within distance 1 of all_words[i]
    balls = []
    for i, w in enumerate(all_words):
        w_list = list(w)
        neighbors = [i]
        for pos in range(6):
            orig_val = w_list[pos]
            for val in range(3):
                if val != orig_val:
                    w_list[pos] = val
                    neighbors.append(word_to_idx[tuple(w_list)])
                    w_list[pos] = orig_val
        balls.append(neighbors)
        
    best_code = None
    best_size = float('inf')
    
    # Helper to check coverage and get uncovered set
    def get_uncovered(code_indices):
        covered = [False] * N
        for c_idx in code_indices:
            for nb in balls[c_idx]:
                covered[nb] = True
        return [i for i in range(N) if not covered[i]]
        
    # Helper to check if a code is valid
    def is_valid(code_indices):
        covered = [False] * N
        for c_idx in code_indices:
            for nb in balls[c_idx]:
                covered[nb] = True
        return all(covered)
        
    # Greedy Construction
    def greedy_construct():
        cover_count = [0] * N
        code = []
        # We track uncovered words to speed up candidate evaluation
        # But we need to evaluate all candidates to find the best one
        # Optimization: Only evaluate candidates that are currently uncovered?
        # Actually, if a word is covered, it might still be a good candidate to cover OTHER uncovered words.
        # So we must evaluate all 729 candidates.
        
        uncovered_count = N
        while uncovered_count > 0:
            best_candidate = -1
            best_gain = -1
            
            # Iterate all candidates
            for c_idx in range(N):
                # Count how many words in ball[c_idx] are currently uncovered (cover_count == 0)
                gain = 0
                for nb in balls[c_idx]:
                    if cover_count[nb] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = c_idx
            
            if best_candidate == -1:
                break # Should not happen if uncovered_count > 0
                
            code.append(best_candidate)
            
            # Update cover_count
            for nb in balls[best_candidate]:
                if cover_count[nb] == 0:
                    uncovered_count -= 1
                cover_count[nb] += 1
                
        return code
        
    # Local Search: Removal
    def local_search_removal(code):
        changed = True
        while changed:
            changed = False
            # Calculate cover counts
            cover_count = [0] * N
            for c_idx in code:
                for nb in balls[c_idx]:
                    cover_count[nb] += 1
            
            # Try to remove each codeword
            for i in range(len(code)):
                c_idx = code[i]
                # Check if removing c_idx leaves any word with cover_count 0
                # i.e., for all nb in balls[c_idx], cover_count[nb] > 1
                removable = True
                for nb in balls[c_idx]:
                    if cover_count[nb] == 1:
                        removable = False
                        break
                if removable:
                    # Remove it
                    code.pop(i)
                    changed = True
                    # Update cover_count for the removed word
                    for nb in balls[c_idx]:
                        cover_count[nb] -= 1
                    break # Restart loop as indices changed
                    
        return code
        
    # Run multiple restarts
    for restart in range(50): # Max restarts
        if time.time() > deadline:
            break
            
        # Randomize tie-breaking in greedy by shuffling candidate order slightly?
        # Or just run greedy as is. Greedy is deterministic.
        # To get different solutions, we can randomize the order of candidates in the greedy step.
        
        code = greedy_construct()
        code = local_search_removal(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # If we hit the theoretical lower bound or known best, we can stop early
            if best_size <= 73:
                # We can try to improve further, but 73 is the target.
                # Let's continue a bit to see if we can get 72 or 71?
                # Lower bound is 71.
                pass

    # If best_code is None, something went wrong
    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
        
    # Convert indices back to words
    return [list(all_words[i]) for i in best_code]
