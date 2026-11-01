import time
import random
import itertools
import numpy as np
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = 120.0 # Keep a safety margin from 150s
    
    # Fallback for other parameters
    if q != 3 or n != 6 or radius != 1:
        return _greedy_cover(q, n, radius)

    # Specific optimization for 3, 6, 1
    N = q ** n
    # Generate all words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (ball of radius 1) for each word
    # ball[i] is a set of indices of words within distance 1 of all_words[i]
    balls = []
    for w in all_words:
        ball = set()
        ball.add(word_to_idx[tuple(w)])
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig: continue
                new_w = w.copy()
                new_w[pos] = val
                ball.add(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    # Symmetry score for tie-breaking
    # We use a simple canonical form: sort the coordinates.
    # This is not a full symmetry group reduction, but it helps break ties
    # in a way that favors "ordered" words, which often leads to better
    # packing in small search spaces.
    def symmetry_score(idx):
        w = all_words[idx]
        return tuple(sorted(w))
        
    # 1. Greedy Set Cover with Symmetry Tie-Breaking
    uncovered = set(range(N))
    code_indices = []
    picked = set()
    
    while uncovered:
        best_word = -1
        max_covered = 0
        best_score = None
        
        # Optimization: Iterate all words
        for i in range(N):
            if i in picked: continue
            inter = len(balls[i] & uncovered)
            if inter > max_covered:
                max_covered = inter
                best_word = i
                best_score = symmetry_score(i)
            elif inter == max_covered and inter > 0:
                # Tie-breaking: use symmetry score
                curr_score = symmetry_score(i)
                if best_score is None or curr_score < best_score:
                    best_word = i
                    best_score = curr_score
                    
        if best_word == -1:
            break # Should not happen
            
        code_indices.append(best_word)
        picked.add(best_word)
        uncovered -= balls[best_word]
        
    initial_code = [all_words[i] for i in code_indices]
    
    # 2. Greedy Reduction: Remove any redundant codeword
    def is_valid_set(code_set):
        covered = set()
        for idx in code_set:
            covered.update(balls[idx])
        return len(covered) == N

    # Remove redundant
    changed = True
    while changed:
        changed = False
        for i in range(len(code_indices)):
            new_set = set(code_indices)
            new_set.remove(code_indices[i])
            if is_valid_set(new_set):
                code_indices.pop(i)
                changed = True
                break # Restart loop as indices changed

    # 3. Local Search
    current_code_set = set(code_indices)
    best_code_set = set(code_indices)
    best_size = len(best_code_set)
    
    def build_cov_counts(code_set):
        counts = np.zeros(N, dtype=int)
        for idx in code_set:
            counts[list(balls[idx])] += 1
        return counts
        
    current_counts = build_cov_counts(current_code_set)
    
    def try_remove(code_set, counts, idx_to_remove):
        for w in balls[idx_to_remove]:
            if counts[w] == 1:
                return False
        return True
        
    def do_remove(code_set, counts, idx_to_remove):
        code_set.remove(idx_to_remove)
        for w in balls[idx_to_remove]:
            counts[w] -= 1
            
    def do_add(code_set, counts, idx_to_add):
        code_set.add(idx_to_add)
        for w in balls[idx_to_add]:
            counts[w] += 1

    # Local Search Loop
    # We use a simulated annealing-like approach but with deterministic symmetry-guided swaps
    temperature = 1.0
    cooling = 0.995
    no_improve_count = 0
    
    while time.time() - start_time < deadline - 5.0:
        if len(current_code_set) < best_size:
            best_size = len(current_code_set)
            best_code_set = set(current_code_set)
            no_improve_count = 0
            temperature = 1.0 # Reset temperature on improvement
        else:
            no_improve_count += 1
            
        # If we hit the known best (73), we can still try to go lower, but 
        # we should be careful with time.
        
        if not current_code_set:
            break
            
        # Strategy:
        # 1. Try to remove a random codeword.
        # 2. If that fails, try a swap: remove one, add one.
        #    To make the swap "smart", we pick the word to add from the 
        #    "most uncovered" or "most symmetric" candidates?
        #    Actually, a good heuristic is to pick the word to add that 
        #    covers the most words that were *only* covered by the removed word,
        #    or simply the word with the highest symmetry score among those
        #    that allow the swap.
        
        idx_remove = random.choice(list(current_code_set))
        
        # Check if pure removal is possible
        if try_remove(current_code_set, current_counts, idx_remove):
            do_remove(current_code_set, current_counts, idx_remove)
            continue
            
        # Pure removal failed. Try Swap.
        # We need to find an idx_add such that (current - idx_remove + idx_add) is valid.
        # Condition: For all w in balls[idx_remove] \ balls[idx_add], counts[w] > 1.
        # We pick idx_add randomly first, but we can bias it.
        
        # Generate a few candidates for idx_add and pick the best one
        # "Best" = allows swap AND has good symmetry score
        candidates = []
        # Sample a few random candidates to save time
        for _ in range(10):
            idx_add = random.randint(0, N - 1)
            if idx_add == idx_remove:
                continue
            if idx_add in current_code_set:
                continue # Swapping with an existing word is a no-op or reorder
            
            # Check validity of swap
            valid_swap = True
            for w in balls[idx_remove]:
                if w not in balls[idx_add]:
                    if current_counts[w] == 1:
                        valid_swap = False
                        break
                        
            if valid_swap:
                candidates.append(idx_add)
                
        if candidates:
            # Pick the candidate with the best symmetry score
            # This encourages moving towards "structured" codes
            best_add = min(candidates, key=lambda x: symmetry_score(x))
            
            # Perform swap
            do_remove(current_code_set, current_counts, idx_remove)
            do_add(current_code_set, current_counts, best_add)
            
            # After swap, try to remove a redundant word again
            # Pick a random word from the set to try removing
            if current_code_set:
                idx_try_remove = random.choice(list(current_code_set))
                if try_remove(current_code_set, current_counts, idx_try_remove):
                    do_remove(current_code_set, current_counts, idx_try_remove)
                    temperature = 1.0
        else:
            # No valid swap found in random samples, cool down
            temperature *= cooling
            if temperature < 0.1:
                 # Stuck? Try a bigger kick: remove 2, add 1? Or just break?
                 # For now, just keep cooling.
                 pass

    # Convert best_code_set back to list of words
    best_code_words = [all_words[i] for i in best_code_set]
    
    # Final check
    if not is_valid_set(best_code_set):
         # Fallback to initial if something went wrong
         return initial_code
         
    return best_code_words

def _greedy_cover(q, n, radius):
    # Fallback for other parameters
    N = q ** n
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = []
    for w in all_words:
        ball = set()
        ball.add(word_to_idx[tuple(w)])
        # Generate all neighbors within radius
        # For small radius, use product
        for delta in itertools.product(range(-radius, radius+1), repeat=n):
            # Check if distance is <= radius
            dist = sum(abs(d) for d in delta)
            if dist <= radius:
                new_w = [0]*n
                valid = True
                for i in range(n):
                    val = w[i] + delta[i]
                    if 0 <= val < q:
                        new_w[i] = val
                    else:
                        valid = False
                        break
                if valid:
                    ball.add(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    uncovered = set(range(N))
    code_indices = []
    
    while uncovered:
        best_word = -1
        max_covered = 0
        for i in range(N):
            if i in code_indices: continue
            inter = len(balls[i] & uncovered)
            if inter > max_covered:
                max_covered = inter
                best_word = i
        if best_word == -1: break
        code_indices.append(best_word)
        uncovered -= balls[best_word]
        
    return [all_words[i] for i in code_indices]

if __name__ == "__main__":
    # Simple test
    code = construct(3, 6, 1)
    print(f"Code size: {len(code)}")
