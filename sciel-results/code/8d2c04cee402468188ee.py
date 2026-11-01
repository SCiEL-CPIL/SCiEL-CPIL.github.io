import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = 118.0

    # Handle the specific case q=3, n=6, radius=1
    # This is the football pool problem K_3(6,1)
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        return _greedy_cover_generic(q, n, radius)
    
    # For q=3, n=6, radius=1
    # N = 3^6 = 729 words
    # Ball size = 1 + 6*2 = 13 words
    # We need to cover all 729 words
    
    N = q ** n  # 729
    
    # Precompute all words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls (neighborhoods) as sets
    balls = []
    for w in all_words:
        ball = set()
        ball.add(word_to_idx[tuple(w)])
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                ball.add(word_to_idx[tuple(new_w)])
        balls.append(ball)
    
    # Convert balls to lists for faster access
    balls_list = [list(b) for b in balls]
    
    # Greedy Set Cover to get initial solution
    uncovered = set(range(N))
    code_indices = []
    
    # Count how many uncovered words each candidate covers
    while uncovered:
        best_word = -1
        max_covered = 0
        for i in range(N):
            if i in code_indices:
                continue
            # Count how many uncovered words this covers
            covered_count = 0
            for w in balls_list[i]:
                if w in uncovered:
                    covered_count += 1
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = i
        
        if best_word == -1:
            break
        
        code_indices.append(best_word)
        uncovered.difference_update(balls_list[best_word])
    
    initial_code = [all_words[i] for i in code_indices]
    initial_size = len(code_indices)
    
    # If we're already at or below the known best, return
    if initial_size <= 73:
        return initial_code
    
    # Local search to improve
    # State: set of codeword indices, coverage counts
    current_set = set(code_indices)
    counts = np.zeros(N, dtype=np.int8)
    for idx in current_set:
        for w in balls_list[idx]:
            counts[w] += 1
    
    best_set = set(current_set)
    best_size = len(best_set)
    
    # Simulated annealing / local search
    # Moves:
    # 1. Remove a codeword (if valid)
    # 2. Swap: remove one, add another (if valid)
    
    temp = 1.0
    cooling = 0.995
    
    # Precompute for each codeword which words it uniquely covers (count == 1)
    # This helps identify which removals are immediately invalid
    
    while time.time() - start_time < deadline - 2.0:
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            if best_size <= 73:
                break  # Reached known best
        
        # Pick a random move type
        move_type = random.random()
        
        if move_type < 0.3 and len(current_set) > 1:
            # Try to remove a codeword
            idx_remove = random.choice(list(current_set))
            
            # Check if removal is valid (no word becomes uncovered)
            valid = True
            for w in balls_list[idx_remove]:
                if counts[w] == 1:
                    valid = False
                    break
            
            if valid:
                # Perform removal
                current_set.discard(idx_remove)
                for w in balls_list[idx_remove]:
                    counts[w] -= 1
                temp = 1.0  # Reset temperature on improvement
        else:
            # Swap: remove one, add another
            if len(current_set) == 0:
                break
            
            idx_remove = random.choice(list(current_set))
            idx_add = random.randint(0, N - 1)
            
            if idx_remove == idx_add:
                continue
            
            # Check if swap is valid
            # Words only covered by idx_remove (not by idx_add) must have count > 1
            valid = True
            for w in balls_list[idx_remove]:
                if w not in set(balls_list[idx_add]) and counts[w] == 1:
                    valid = False
                    break
            
            if valid:
                # Perform swap
                current_set.discard(idx_remove)
                for w in balls_list[idx_remove]:
                    counts[w] -= 1
                
                current_set.add(idx_add)
                for w in balls_list[idx_add]:
                    counts[w] += 1
                
                # After swap, try to remove a redundant codeword
                if len(current_set) > 1:
                    idx_try_remove = random.choice(list(current_set))
                    valid_remove = True
                    for w in balls_list[idx_try_remove]:
                        if counts[w] == 1:
                            valid_remove = False
                            break
                    if valid_remove:
                        current_set.discard(idx_try_remove)
                        for w in balls_list[idx_try_remove]:
                            counts[w] -= 1
                        temp = 1.0  # Reset temperature on improvement
        
        temp *= cooling
    
    # Convert best_set back to list of words
    best_code_words = [all_words[i] for i in best_set]
    
    # Final validity check
    covered = np.zeros(N, dtype=bool)
    for idx in best_set:
        for w in balls_list[idx]:
            covered[w] = True
    
    if not np.all(covered):
        # Fallback to initial
        return initial_code
    
    return best_code_words


def _greedy_cover_generic(q, n, radius):
    # Fallback for other parameters
    N = q ** n
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = []
    for w in all_words:
        ball = set()
        ball.add(word_to_idx[tuple(w)])
        for delta in itertools.product(range(-radius, radius + 1), repeat=n):
            dist = sum(abs(d) for d in delta)
            if dist <= radius:
                new_w = [0] * n
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
            if i in code_indices:
                continue
            inter = len(balls[i] & uncovered)
            if inter > max_covered:
                max_covered = inter
                best_word = i
        if best_word == -1:
            break
        code_indices.append(best_word)
        uncovered.difference_update(balls[best_word])
    
    return [all_words[i] for i in code_indices]
