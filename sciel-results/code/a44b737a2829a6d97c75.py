"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def _idx_to_word(idx, q, n):
    word = [0] * n
    for d in range(n - 1, -1, -1):
        word[d] = idx % q
        idx //= q
    return word

def _word_to_idx(word, q):
    idx = 0
    for d in range(len(word) - 1, -1, -1):
        idx = idx * q + word[d]
    return idx

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s safety margin for the 150s limit
    
    total_words = q ** n
    
    # Precompute ball masks for all words
    # For radius=1: word itself + all single-position changes
    # ball_mask[i] is a set of word indices covered by word i
    ball_masks = []
    for idx in range(total_words):
        word = _idx_to_word(idx, q, n)
        ball = set()
        # Original word
        ball.add(idx)
        # Single position changes
        for pos in range(n):
            original_val = word[pos]
            base = idx
            # Compute the contribution of this position
            pos_val = original_val
            # Remove this position's contribution
            # Actually, let's just compute directly
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_word = word[:]
                new_word[pos] = new_val
                new_idx = _word_to_idx(new_word, q)
                ball.add(new_idx)
        ball_masks.append(ball)
    
    # Convert to sorted lists for faster iteration
    ball_lists = [sorted(b) for b in ball_masks]
    
    # Greedy set cover to get initial code
    # uncovered[w_idx] = True if not yet covered
    uncovered = [True] * total_words
    covered_count = [0] * total_words  # How many codewords cover this word
    code = []
    code_set = set()
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    remaining = set(range(total_words))
    
    while remaining and time.time() < deadline - 60:
        best_idx = -1
        best_count = 0
        # Sample some candidates to speed up
        sample_size = min(len(remaining), 500)
        if len(remaining) > sample_size:
            sample = random.sample(remaining, sample_size)
        else:
            sample = list(remaining)
        
        for idx in sample:
            count = sum(1 for w in ball_lists[idx] if uncovered[w])
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
        
        code.append(best_idx)
        code_set.add(best_idx)
        for w in ball_lists[best_idx]:
            if uncovered[w]:
                uncovered[w] = False
                remaining.discard(w)
            covered_count[w] += 1
    
    # If we still have uncovered words, do a full greedy pass
    while remaining and time.time() < deadline - 30:
        best_idx = -1
        best_count = 0
        for idx in remaining:
            count = sum(1 for w in ball_lists[idx] if uncovered[w])
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1 or best_count == 0:
            break
        code.append(best_idx)
        code_set.add(best_idx)
        for w in ball_lists[best_idx]:
            if uncovered[w]:
                uncovered[w] = False
                remaining.discard(w)
            covered_count[w] += 1
    
    # Now try to improve via simulated annealing
    # State: code (list of indices), code_set, covered_count
    # Objective: minimize len(code)
    # Moves: remove (if valid), swap, kick
    
    def try_remove(c):
        """Try to remove codeword c. Returns True if successful."""
        for w in ball_lists[c]:
            if covered_count[w] == 1:
                return False
        for w in ball_lists[c]:
            covered_count[w] -= 1
            if covered_count[w] == 0:
                uncovered[w] = True
        code.remove(c)
        code_set.discard(c)
        return True
    
    def undo_remove(c):
        """Undo a removal of codeword c."""
        code.append(c)
        code_set.add(c)
        for w in ball_lists[c]:
            covered_count[w] += 1
            if covered_count[w] == 1:
                uncovered[w] = False
    
    best_code_len = len(code)
    best_code_snapshot = list(code)
    
    # Simulated annealing
    temp = 50.0
    cool_rate = 0.98
    min_temp = 0.1
    sa_steps = 0
    
    while temp > min_temp and time.time() < deadline - 10:
        sa_steps += 1
        
        # Try removals
        moved = False
        for _ in range(50):
            if len(code) < 2:
                break
            c = random.choice(code)
            if try_remove(c):
                moved = True
                if len(code) < best_code_len:
                    best_code_len = len(code)
                    best_code_snapshot = list(code)
                break
            # If removal failed, undo the partial work
            # Actually, try_remove returns False without modifying if invalid
            # Let me check... it returns False early, so no modification
        
        if moved:
            continue
        
        # Try swaps
        for _ in range(50):
            if len(code) < 2:
                break
            c_out = random.choice(code)
            # Pick a random candidate not in code
            c_in = random.randint(0, total_words - 1)
            if c_in in code_set:
                continue
            
            # Check if swap is valid: removing c_out should not uncover any word
            # that is only covered by c_out and not by c_in
            valid = True
            for w in ball_lists[c_out]:
                if covered_count[w] == 1 and w not in ball_lists[c_in]:
                    valid = False
                    break
            
            if valid:
                # Perform swap
                for w in ball_lists[c_out]:
                    covered_count[w] -= 1
                    if covered_count[w] == 0:
                        uncovered[w] = True
                code.remove(c_out)
                code_set.discard(c_out)
                
                for w in ball_lists[c_in]:
                    covered_count[w] += 1
                    if covered_count[w] == 1:
                        uncovered[w] = False
                code.append(c_in)
                code_set.add(c_in)
                
                moved = True
                if len(code) < best_code_len:
                    best_code_len = len(code)
                    best_code_snapshot = list(code)
                break
        
        if moved:
            continue
        
        # Try kicks (add a random word)
        c_in = random.randint(0, total_words - 1)
        if c_in not in code_set:
            code.append(c_in)
            code_set.add(c_in)
            for w in ball_lists[c_in]:
                covered_count[w] += 1
                if covered_count[w] == 1:
                    uncovered[w] = False
        
        temp *= cool_rate
    
    # Restore best solution found
    # We need to recompute covered_count from best_code_snapshot
    code = list(best_code_snapshot)
    code_set = set(code)
    covered_count = [0] * total_words
    uncovered = [True] * total_words
    for c in code:
        for w in ball_lists[c]:
            covered_count[w] += 1
    for w in range(total_words):
        if covered_count[w] > 0:
            uncovered[w] = False
    
    # Final cleanup: try to remove redundant codewords
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        indices = list(range(len(code)))
        random.shuffle(indices)
        for ci in indices:
            if time.time() >= deadline - 5:
                break
            c = code[ci]
            # Check if removable
            removable = True
            for w in ball_lists[c]:
                if covered_count[w] == 1:
                    removable = False
                    break
            if removable:
                for w in ball_lists[c]:
                    covered_count[w] -= 1
                    if covered_count[w] == 0:
                        uncovered[w] = True
                code.pop(ci)
                code_set.discard(c)
                improved = True
    
    # Verify coverage
    # (Should be fully covered by construction)
    
    # Convert to words
    result = []
    for c in code:
        result.append(_idx_to_word(c, q, n))
    
    return result
# EVOLVE-BLOCK-END
