import itertools
import random
import time
import numpy as np

def construct(q, n, radius):
    # Parameters for the specific task: q=3, n=7, radius=1
    # Handle generally if possible, but optimize for q=3, n=7, radius=1
    
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls (neighbors within radius)
    # balls[i] is a list of indices of words within distance <= radius from all_words[i]
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
        
    # Initial Greedy Code
    def get_initial_code():
        uncovered = set(range(num_words))
        code = []
        while uncovered:
            best_c = -1
            best_cov = -1
            # Use random tie-breaking to diversify
            candidates = []
            for c in range(num_words):
                cov = 0
                for u in balls[c]:
                    if u in uncovered:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    candidates = [c]
                elif cov == best_cov:
                    candidates.append(c)
            if candidates:
                best_c = random.choice(candidates)
            else:
                best_c = random.randint(0, num_words - 1)
            
            code.append(best_c)
            for u in balls[best_c]:
                if u in uncovered:
                    uncovered.remove(u)
        return code

    best_code = get_initial_code()
    best_size = len(best_code)
    
    # Setup state for Simulated Annealing
    # We track the set of codewords and the count of coverage for each word
    def calculate_state(code_list):
        cnt = [0] * num_words
        for c in code_list:
            for u in balls[c]:
                cnt[u] += 1
        num_uncovered = sum(1 for c in cnt if c == 0)
        return set(code_list), cnt, num_uncovered

    current_code, current_cnt, current_num_unc = calculate_state(best_code)
    
    # Energy function
    # We want to minimize size, but coverage is a hard constraint.
    # Penalty for uncovered words should be large.
    PENALTY = 100000
    
    def get_energy(code_set, cnt, num_unc):
        return len(code_set) + PENALTY * num_unc

    current_energy = get_energy(current_code, current_cnt, current_num_unc)
    best_energy = current_energy
    
    start_time = time.time()
    time_limit = 110.0
    
    T = 100.0 # Initial temperature
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        # Generate a random neighbor
        # Moves: Add, Remove, Swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random word
            if len(current_code) < num_words:
                new_c = random.randint(0, num_words - 1)
                if new_c not in current_code:
                    # Tentatively add
                    current_code.add(new_c)
                    for u in balls[new_c]:
                        if current_cnt[u] == 0:
                            current_num_unc -= 1
                        current_cnt[u] += 1
                    new_energy = get_energy(current_code, current_cnt, current_num_unc)
                    
                    # Acceptance criterion
                    delta = new_energy - current_energy
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_code = list(current_code)
                            best_size = len(current_code)
                    else:
                        # Revert
                        current_code.remove(new_c)
                        for u in balls[new_c]:
                            current_cnt[u] -= 1
                            if current_cnt[u] == 0:
                                current_num_unc += 1
                            
        elif move_type < 0.6:
            # Remove a random word
            if len(current_code) > 0:
                to_remove = random.choice(list(current_code))
                # Check if removal makes it invalid? 
                # In SA, we allow invalid states (with penalty)
                current_code.remove(to_remove)
                for u in balls[to_remove]:
                    current_cnt[u] -= 1
                    if current_cnt[u] == 0:
                        current_num_unc += 1
                new_energy = get_energy(current_code, current_cnt, current_num_unc)
                
                delta = new_energy - current_energy
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_code = list(current_code)
                        best_size = len(current_code)
                else:
                    # Revert
                    current_code.add(to_remove)
                    for u in balls[to_remove]:
                        if current_cnt[u] == 0:
                            current_num_unc -= 1
                        current_cnt[u] += 1
                            
        else:
            # Swap: Remove a random word from code, add a random word from space
            if len(current_code) > 0:
                to_remove = random.choice(list(current_code))
                to_add = random.randint(0, num_words - 1)
                if to_add != to_remove:
                    # Perform swap
                    current_code.remove(to_remove)
                    for u in balls[to_remove]:
                        current_cnt[u] -= 1
                        if current_cnt[u] == 0:
                            current_num_unc += 1
                    
                    current_code.add(to_add)
                    for u in balls[to_add]:
                        if current_cnt[u] == 0:
                            current_num_unc -= 1
                        current_cnt[u] += 1
                        
                    new_energy = get_energy(current_code, current_cnt, current_num_unc)
                    
                    delta = new_energy - current_energy
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_code = list(current_code)
                            best_size = len(current_code)
                    else:
                        # Revert
                        current_code.remove(to_add)
                        for u in balls[to_add]:
                            current_cnt[u] -= 1
                            if current_cnt[u] == 0:
                                current_num_unc += 1
                                
                        current_code.add(to_remove)
                        for u in balls[to_remove]:
                            if current_cnt[u] == 0:
                                current_num_unc -= 1
                            current_cnt[u] += 1
        
        # Cool down
        T *= cooling_rate
        
        # Optional: If we found a valid code (num_unc == 0), try to reduce size with local search?
        # The SA energy function handles this by penalizing uncovered words heavily,
        # so it will naturally try to fix coverage before reducing size.
        
    # Ensure the final result is valid
    # If best_code is not valid, we need to repair it.
    # But since we track best_energy which includes penalty, and we started valid,
    # and we only accept moves that don't increase energy too much, 
    # it's possible best_code is invalid if the penalty was overcome? No, penalty is huge.
    # However, to be safe, let's verify and repair if necessary.
    
    def is_valid(code_list):
        cnt = [0] * num_words
        for c in code_list:
            for u in balls[c]:
                cnt[u] += 1
        return all(c > 0 for c in cnt)

    if not is_valid(best_code):
        # Repair best_code
        uncovered = set(range(num_words))
        for c in best_code:
            for u in balls[c]:
                if u in uncovered:
                    uncovered.remove(u)
        
        while uncovered:
            best_c = -1
            best_cov = -1
            for c in range(num_words):
                if c in best_code:
                    continue
                cov = 0
                for u in balls[c]:
                    if u in uncovered:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_c = c
            if best_c != -1:
                best_code.append(best_c)
                for u in balls[best_c]:
                    if u in uncovered:
                        uncovered.remove(u)
            else:
                break

    return [list(all_words[c]) for c in best_code]
