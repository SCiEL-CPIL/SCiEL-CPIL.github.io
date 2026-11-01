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
        
    # Precompute ball as numpy arrays for faster operations
    ball_arrays = [np.array(sorted(balls[i]), dtype=np.int32) for i in range(N)]
    ball_size = n * (q - 1) + 1  # 13 for q=3, n=6
    
    # Symmetry score for tie-breaking
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
        
        for i in range(N):
            if i in picked: continue
            inter = len(balls[i] & uncovered)
            if inter > max_covered:
                max_covered = inter
                best_word = i
                best_score = symmetry_score(i)
            elif inter == max_covered and inter > 0:
                curr_score = symmetry_score(i)
                if best_score is None or curr_score < best_score:
                    best_word = i
                    best_score = curr_score
                    
        if best_word == -1:
            break
            
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

    changed = True
    while changed:
        changed = False
        for i in range(len(code_indices)):
            new_set = set(code_indices)
            new_set.remove(code_indices[i])
            if is_valid_set(new_set):
                code_indices.pop(i)
                changed = True
                break

    # 3. Local Search
    current_code_set = set(code_indices)
    best_code_set = set(code_indices)
    best_size = len(best_code_set)
    
    def build_cov_counts(code_set):
        counts = np.zeros(N, dtype=np.int32)
        for idx in code_set:
            counts[ball_arrays[idx]] += 1
        return counts
        
    current_counts = build_cov_counts(current_code_set)
    
    def try_remove(code_set, counts, idx_to_remove):
        ba = ball_arrays[idx_to_remove]
        return np.all(counts[ba] > 1)
        
    def do_remove(code_set, counts, idx_to_remove):
        code_set.remove(idx_to_remove)
        ba = ball_arrays[idx_to_remove]
        counts[ba] -= 1
            
    def do_add(code_set, counts, idx_to_add):
        code_set.add(idx_to_add)
        ba = ball_arrays[idx_to_add]
        counts[ba] += 1

    def find_best_swap(code_set, counts, idx_remove):
        """Find the best idx_add that allows a valid swap with idx_remove.
        Score: maximize coverage of words that would become uncovered or have low coverage."""
        ba_remove = ball_arrays[idx_remove]
        # Words that are only covered by idx_remove
        only_covered = ba_remove[counts[ba_remove] == 1]
        
        if len(only_covered) == 0:
            return None, 0  # Pure removal would work, but we're in swap mode
        
        best_add = -1
        best_score = -1
        
        # Try a subset of candidates - focus on words that cover many of the only_covered words
        # Sample strategically
        candidates_tried = 0
        max_candidates = 200
        
        # First, try words that are in the ball of some only_covered word
        candidate_set = set()
        for w in only_covered[:min(len(only_covered), 5)]:
            # Find words that cover w
            for i in range(N):
                if i in code_set:
                    continue
                if w in balls[i]:
                    candidate_set.add(i)
                    if len(candidate_set) >= max_candidates:
                        break
            if len(candidate_set) >= max_candidates:
                break
        
        # Add some random candidates
        for _ in range(50):
            candidate_set.add(random.randint(0, N - 1))
        
        for idx_add in candidate_set:
            if idx_add == idx_remove or idx_add in code_set:
                continue
            
            ba_add = ball_arrays[idx_add]
            # Check if swap is valid: for all w in ba_remove \ ba_add, counts[w] > 1
            # i.e., words in ba_remove that are NOT in ba_add must have count > 1
            # Equivalently: min(counts[w] for w in ba_remove if w not in ba_add) > 1
            
            # Fast check using numpy
            in_add = np.isin(ba_remove, ba_add)
            not_in_add = ~in_add
            if np.any(not_in_add):
                if np.min(counts[ba_remove[not_in_add]]) <= 1:
                    continue
            
            # Valid swap! Score it
            # Score: how many of the only_covered words does idx_add cover?
            covers_only = np.sum(np.isin(only_covered, ba_add))
            
            # Also consider: after swap, what's the minimum coverage?
            # New counts would be counts - counts[ba_remove] + counts[ba_add]
            # But that's expensive. Use a simpler heuristic.
            
            if covers_only > best_score:
                best_score = covers_only
                best_add = idx_add
                if best_score == len(only_covered):
                    break  # Perfect swap found
        
        return best_add, best_score

    # Local Search Loop
    temperature = 1.0
    cooling = 0.998
    no_improve_count = 0
    stuck_counter = 0
    
    while time.time() - start_time < deadline - 5.0:
        if len(current_code_set) < best_size:
            best_size = len(current_code_set)
            best_code_set = set(current_code_set)
            no_improve_count = 0
            stuck_counter = 0
            temperature = 1.0
        else:
            no_improve_count += 1
            stuck_counter += 1
            
        if not current_code_set:
            break
            
        # Strategy:
        # 1. Try to remove a random codeword.
        # 2. If that fails, try a smart swap.
        # 3. If stuck for a while, try a kick (remove 2, add 1).
        
        idx_remove = random.choice(list(current_code_set))
        
        if try_remove(current_code_set, current_counts, idx_remove):
            do_remove(current_code_set, current_counts, idx_remove)
            continue
            
        # Try smart swap
        best_add, swap_score = find_best_swap(current_code_set, current_counts, idx_remove)
        
        if best_add != -1:
            do_remove(current_code_set, current_counts, idx_remove)
            do_add(current_code_set, current_counts, best_add)
            
            # After swap, try to remove a redundant word
            if current_code_set:
                # Try to find a word that can be removed
                removable = [i for i in current_code_set if try_remove(current_code_set, current_counts, i)]
                if removable:
                    idx_try_remove = random.choice(removable)
                    do_remove(current_code_set, current_counts, idx_try_remove)
        else:
            # No valid swap found. Try a kick if stuck.
            if stuck_counter > 50:
                # Kick: remove 2 random words, add 1
                if len(current_code_set) >= 2:
                    kicks = list(current_code_set)
                    r1, r2 = random.sample(kicks, 2)
                    
                    # Try to find an idx_add that covers the union of uncovered words from r1 and r2
                    ba_r1 = ball_arrays[r1]
                    ba_r2 = ball_arrays[r2]
                    combined = np.union1d(ba_r1, ba_r2)
                    only_from_r1_or_r2 = combined[(current_counts[combined] == 1)]
                    
                    # Find a word that covers many of these
                    best_add_kick = -1
                    best_cover = 0
                    for _ in range(100):
                        idx_add = random.randint(0, N - 1)
                        if idx_add in current_code_set:
                            continue
                        ba_add = ball_arrays[idx_add]
                        cover = np.sum(np.isin(only_from_r1_or_r2, ba_add))
                        if cover > best_cover:
                            best_cover = cover
                            best_add_kick = idx_add
                    
                    if best_add_kick != -1 and best_cover > len(only_from_r1_or_r2) * 0.5:
                        do_remove(current_code_set, current_counts, r1)
                        do_remove(current_code_set, current_counts, r2)
                        do_add(current_code_set, current_counts, best_add_kick)
                        stuck_counter = 0
                    else:
                        stuck_counter = 0
            else:
                temperature *= cooling

    # Convert best_code_set back to list of words
    best_code_words = [all_words[i] for i in best_code_set]
    
    # Final check
    if not is_valid_set(best_code_set):
         return initial_code
         
    return best_code_words

def _greedy_cover(q, n, radius):
    N = q ** n
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    balls = []
    for w in all_words:
        ball = set()
        ball.add(word_to_idx[tuple(w)])
        for delta in itertools.product(range(-radius, radius+1), repeat=n):
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
    code = construct(3, 6, 1)
    print(f"Code size: {len(code)}")
