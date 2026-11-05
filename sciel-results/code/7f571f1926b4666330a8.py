import itertools
import random
import time
import math
import numpy as np

def construct(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls (neighbors within radius)
    balls = []
    for w in all_words:
        ball = [word_to_idx[w]]
        if radius == 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    ball.append(word_to_idx[tuple(new_w)])
        balls.append(ball)
        
    # Precompute reverse balls? Not strictly needed for SA if we scan, but useful for speed.
    # For now, let's use the direct balls.
    
    # Initial Greedy Code (same as before to get a good starting point)
    def get_initial_code():
        uncovered = set(range(num_words))
        code = []
        while uncovered:
            best_c = -1
            best_cov = -1
            for c in range(num_words):
                cov = 0
                for u in balls[c]:
                    if u in uncovered:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_c = c
            code.append(best_c)
            for u in balls[best_c]:
                if u in uncovered:
                    uncovered.remove(u)
        return code

    initial_code = get_initial_code()
    
    # State for SA
    # code_set: set of indices
    # cnt: count of how many times each word is covered
    # num_unc: number of uncovered words
    
    def calc_state(code_list):
        cnt = [0] * num_words
        for c in code_list:
            for u in balls[c]:
                cnt[u] += 1
        num_unc = sum(1 for c in cnt if c == 0)
        return set(code_list), cnt, num_unc

    current_code, current_cnt, current_num_unc = calc_state(initial_code)
    best_code = list(current_code)
    best_size = len(best_code)
    
    # Objective function: Size + Penalty * Uncovered
    # We want to minimize this.
    # If valid, objective = size.
    # If invalid, objective = size + 1000 * num_unc (or similar)
    
    def objective(code_size, num_unc):
        return code_size + 1000 * num_unc

    current_obj = objective(len(current_code), current_num_unc)
    best_obj = current_obj
    
    start_time = time.time()
    time_limit = 110.0
    
    # SA Parameters
    T_start = 100.0
    T_end = 0.1
    cooling_rate = 0.995 # Adjust based on iterations
    
    T = T_start
    
    while time.time() - start_time < time_limit:
        # Propose a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to remove a random codeword
            if not current_code:
                continue
            c_to_remove = random.choice(list(current_code))
            
            # Check if removal is valid (all neighbors still covered by others)
            valid_removal = True
            for u in balls[c_to_remove]:
                if current_cnt[u] <= 1:
                    valid_removal = False
                    break
            
            if valid_removal:
                # Apply move
                current_code.remove(c_to_remove)
                for u in balls[c_to_remove]:
                    current_cnt[u] -= 1
                    if current_cnt[u] == 0:
                        current_num_unc += 1
                # Note: If valid_removal was true, num_unc shouldn't increase, 
                # but let's re-evaluate or trust the logic.
                # Actually, if cnt[u] was 1, it becomes 0, so unc increases.
                # But valid_removal checks cnt[u] <= 1. If cnt[u] == 1, removal makes it 0.
                # So valid_removal should require cnt[u] > 1 for all u in ball.
                # Let's fix the check:
                # If I remove c, u is covered by cnt[u]-1.
                # We need cnt[u]-1 >= 1 => cnt[u] >= 2.
                # So check: if current_cnt[u] < 2, invalid.
                
                # Let's restart the move logic with correct check
                # Undo move
                current_code.add(c_to_remove)
                for u in balls[c_to_remove]:
                    current_cnt[u] += 1
                    if current_cnt[u] == 1: # It was 0, now 1, so unc decreases
                         current_num_unc -= 1
                 
                valid_removal = True
                for u in balls[c_to_remove]:
                    if current_cnt[u] < 2:
                        valid_removal = False
                        break
                
                if valid_removal:
                    # Apply move
                    current_code.remove(c_to_remove)
                    for u in balls[c_to_remove]:
                        current_cnt[u] -= 1
                        # num_unc does not change because cnt[u] was >= 2, so it stays >= 1
                        # Wait, if cnt[u] was 2, it becomes 1. Still covered.
                        # So num_unc remains same.
                else:
                    continue # Move invalid, skip
            
            new_obj = objective(len(current_code), current_num_unc)
            
        elif move_type < 0.8:
            # Try to add a random codeword
            if len(current_code) >= num_words:
                continue
            # Pick a random word not in code
            # Efficient way: pick random index, if in code, retry or just pick from all
            # Since num_words is small, we can just pick random and check
            c_to_add = random.randint(0, num_words - 1)
            if c_to_add in current_code:
                continue
            
            current_code.add(c_to_add)
            old_unc = current_num_unc
            for u in balls[c_to_add]:
                if current_cnt[u] == 0:
                    current_num_unc -= 1
                current_cnt[u] += 1
            
            new_obj = objective(len(current_code), current_num_unc)
            
        else:
            # Swap: remove one, add another
            if len(current_code) < 2:
                continue
            c1 = random.choice(list(current_code))
            c2 = random.randint(0, num_words - 1)
            if c2 in current_code:
                continue
            
            # Apply swap
            current_code.remove(c1)
            for u in balls[c1]:
                if current_cnt[u] == 1:
                    current_num_unc += 1
                current_cnt[u] -= 1
                
            current_code.add(c2)
            for u in balls[c2]:
                if current_cnt[u] == 0:
                    current_num_unc -= 1
                current_cnt[u] += 1
            
            new_obj = objective(len(current_code), current_num_unc)
        
        # Acceptance
        delta = new_obj - current_obj
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_code = list(current_code)
        else:
            # Reject: Undo move
            # We need to undo the specific move. 
            # This is tricky without storing the state. 
            # Better approach: Copy state before move? Too slow.
            # Or, just re-calculate state? Too slow.
            # Or, structure moves to be easily reversible.
            
            # For "Remove":
            if move_type < 0.5:
                 # Undo removal
                 current_code.add(c_to_remove)
                 for u in balls[c_to_remove]:
                     current_cnt[u] += 1
                     if current_cnt[u] == 1: # Was 0, now 1
                         current_num_unc -= 1
                 
            # For "Add":
            elif move_type < 0.8:
                 # Undo addition
                 current_code.remove(c_to_add)
                 for u in balls[c_to_add]:
                     current_cnt[u] -= 1
                     if current_cnt[u] == 0:
                         current_num_unc += 1
                 
            # For "Swap":
            else:
                 # Undo swap
                 current_code.remove(c2)
                 for u in balls[c2]:
                     current_cnt[u] -= 1
                     if current_cnt[u] == 0:
                         current_num_unc += 1
                         
                 current_code.add(c1)
                 for u in balls[c1]:
                     if current_cnt[u] == 0:
                         current_num_unc -= 1
                     current_cnt[u] += 1
            
            current_obj = objective(len(current_code), current_num_unc)
            
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return [list(all_words[c]) for c in best_code]
