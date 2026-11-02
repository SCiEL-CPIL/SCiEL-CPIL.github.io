import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1_sa()
    return _greedy_cover(q, n, radius)

def _construct_3_7_1_sa():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # 1. Generate initial code using Greedy Set Cover
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for all words
    neighbors = []
    for w in all_words:
        nb = [word_to_idx[w]]
        for i in range(7):
            for d in [1, 2]:
                new_w = list(w)
                new_w[i] = (w[i] + d) % 3
                nb.append(word_to_idx[tuple(new_w)])
        neighbors.append(nb)
    
    # Greedy Set Cover to get initial valid code
    covers = [set(nb) for nb in neighbors]
    uncovered = set(range(N))
    code = []
    code_set = set()
    
    while uncovered:
        best = -1
        best_count = -1
        for i in range(N):
            if i in code_set:
                continue
            count = len(covers[i] & uncovered)
            if count > best_count:
                best_count = count
                best = i
        
        if best == -1:
            break
            
        code.append(best)
        code_set.add(best)
        uncovered -= covers[best]
        
    # 2. Simulated Annealing
    # State: code_set (set of indices), cover_count (array)
    cover_count = np.zeros(N, dtype=np.int32)
    for c in code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    # Initial energy
    # E = size + penalty * uncovered
    penalty = 100.0 # Large enough to discourage invalidity
    current_size = len(code)
    current_uncovered = np.sum(cover_count == 0)
    current_energy = current_size + penalty * current_uncovered
    
    best_size = current_size
    best_code_set = set(code_set)
    
    T = 10.0 # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    # Helper to update counts and return change in uncovered count
    def apply_change(idx, delta):
        # delta is 1 for add, -1 for remove
        # Returns change in number of uncovered words (0->1 is +1, 1->0 is -1)
        change_uncovered = 0
        for nb in neighbors[idx]:
            old_val = cover_count[nb]
            new_val = old_val + delta
            cover_count[nb] = new_val
            if old_val == 0 and new_val > 0:
                change_uncovered -= 1 # Was uncovered, now covered
            elif old_val > 0 and new_val == 0:
                change_uncovered += 1 # Was covered, now uncovered
        return change_uncovered

    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.random()
        if move_type < 0.3:
            # Remove a random codeword
            if not code_set:
                continue
            c = random.choice(list(code_set))
            
            # Apply
            ch_uncovered = apply_change(c, -1)
            new_size = current_size - 1
            new_uncovered = current_uncovered + ch_uncovered
            new_energy = new_size + penalty * new_uncovered
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                # Accept
                code_set.remove(c)
                current_size = new_size
                current_uncovered = new_uncovered
                current_energy = new_energy
                
                if current_uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_code_set = set(code_set)
            else:
                # Reject
                apply_change(c, 1)
                
        elif move_type < 0.7:
            # Add a random word
            c_new = random.randint(0, N - 1)
            if c_new in code_set:
                continue
                
            ch_uncovered = apply_change(c_new, 1)
            new_size = current_size + 1
            new_uncovered = current_uncovered + ch_uncovered
            new_energy = new_size + penalty * new_uncovered
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                code_set.add(c_new)
                current_size = new_size
                current_uncovered = new_uncovered
                current_energy = new_energy
                
                if current_uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_code_set = set(code_set)
            else:
                apply_change(c_new, -1)
                
        else:
            # Swap a random codeword with a random other word
            if not code_set:
                continue
            c_old = random.choice(list(code_set))
            c_new = random.randint(0, N - 1)
            if c_new == c_old or c_new in code_set:
                continue
                
            # Apply swap: remove c_old, add c_new
            ch1 = apply_change(c_old, -1)
            ch2 = apply_change(c_new, 1)
            total_ch_uncovered = ch1 + ch2
            
            new_size = current_size # Size unchanged
            new_uncovered = current_uncovered + total_ch_uncovered
            new_energy = new_size + penalty * new_uncovered
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                code_set.remove(c_old)
                code_set.add(c_new)
                current_uncovered = new_uncovered
                current_energy = new_energy
                
                if current_uncovered == 0 and current_size < best_size:
                    best_size = current_size
                    best_code_set = set(code_set)
            else:
                # Revert
                apply_change(c_new, -1)
                apply_change(c_old, 1)
                
        T *= cooling_rate

    # Return the best valid code found
    # Ensure best_code_set is valid (it should be, as we only update best if uncovered==0)
    return [list(all_words[idx]) for idx in best_code_set]

def _greedy_cover(q, n, radius):
    # Fallback for other parameters (not used in this specific task instance)
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    covers = []
    for w in all_words:
        covered = set()
        covered.add(word_to_idx[w])
        for r in range(1, radius + 1):
            for positions in itertools.combinations(range(n), r):
                for deltas in itertools.product(range(1, q), repeat=r):
                    new_w = list(w)
                    for pos, d in zip(positions, deltas):
                        new_w[pos] = (w[pos] + d) % q
                    covered.add(word_to_idx[tuple(new_w)])
        covers.append(covered)
    
    uncovered = set(range(N))
    code = []
    while uncovered:
        best = None
        best_count = -1
        for i in range(N):
            count = len(covers[i] & uncovered)
            if count > best_count:
                best_count = count
                best = i
        code.append(list(all_words[best]))
        uncovered -= covers[best]
    
    return code
