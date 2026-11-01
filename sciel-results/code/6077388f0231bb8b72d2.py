"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def construct(q, n, radius):
    # For the specific task: q=3, n=7, radius=1
    # We use a refined Simulated Annealing approach.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Precomputation for q=3, n=7, r=1
    all_words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # neighbors[i] is a list of indices of words within distance 1 of all_words[i]
    neighbors = []
    for w in all_words:
        w_list = list(w)
        idxs = [word_to_idx[w]]
        for i in range(7):
            orig = w_list[i]
            for v in range(3):
                if v != orig:
                    w_list[i] = v
                    idxs.append(word_to_idx[tuple(w_list)])
            w_list[i] = orig
        neighbors.append(idxs)
        
    random.seed(42)
    
    # 1. Greedy Initialization: Max Coverage Greedy
    counts = [0] * 2187
    covered = 0
    code = []
    uncovered_set = set(range(2187))
    
    while uncovered_set:
        best_word_idx = -1
        max_cov = 0
        
        for w_idx in range(2187):
            cov = 0
            for nb in neighbors[w_idx]:
                if nb in uncovered_set:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_word_idx = w_idx
        
        if best_word_idx == -1:
            break
            
        code.append(all_words[best_word_idx])
        for nb in neighbors[best_word_idx]:
            if counts[nb] == 0:
                covered += 1
                uncovered_set.remove(nb)
            counts[nb] += 1
            
    # Start Simulated Annealing
    current_code = code
    current_counts = counts
    current_covered = covered
    current_size = len(current_code)
    
    # Higher penalty for uncovered words to prioritize coverage
    P = 20.0
    
    def get_cost(size, covered):
        return size + P * (2187 - covered)
        
    current_cost = get_cost(current_size, current_covered)
    
    best_code = list(current_code)
    best_size = current_size
    best_covered = current_covered
    best_cost = current_cost
    
    start_time = time.time()
    time_limit = 115.0
    
    T = 50.0
    T_min = 0.01
    cooling = 0.9998
    
    # Helper to update best
    def update_best():
        nonlocal best_code, best_size, best_covered, best_cost
        if current_covered == 2187 and current_size < best_size:
            best_size = current_size
            best_covered = current_covered
            best_code = list(current_code)
            best_cost = current_cost

    # Precompute sets of neighbors for faster diff? 
    # The current code uses sets in replace. Let's keep that but optimize logic.
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            T = T_min
            
        # Determine move strategy based on state
        if current_covered < 2187:
            # Prioritize fixing coverage
            r = random.random()
            if r < 0.7:
                # Replace to cover an uncovered word
                # Pick a random uncovered word
                # Efficiently pick uncovered: iterate counts? 
                # Since 2187 is small, we can just pick random and check
                attempts = 0
                while True:
                    u_idx = random.randint(0, 2186)
                    if current_counts[u_idx] == 0:
                        break
                    attempts += 1
                    if attempts > 100:
                        u_idx = random.randint(0, 2186)
                        break
                
                new_w_idx = random.choice(neighbors[u_idx])
                
                if not current_code:
                    # Add if empty
                    new_w = all_words[new_w_idx]
                    if new_w not in current_code:
                        current_code.append(new_w)
                        for nb in neighbors[new_w_idx]:
                            current_counts[nb] += 1
                        current_size += 1
                        current_covered += sum(1 for nb in neighbors[new_w_idx] if current_counts[nb] == 1) # This is wrong, need to recompute or track delta properly
                        # Actually, let's just recompute delta properly in a generic add/replace function
                        # For simplicity in this branch, if empty, just add greedily
                        pass 
                else:
                    idx = random.randint(0, current_size - 1)
                    old_w_idx = word_to_idx[current_code[idx]]
                    
                    if new_w_idx == old_w_idx:
                        continue
                    
                    old_nbs_set = set(neighbors[old_w_idx])
                    new_nbs_set = set(neighbors[new_w_idx])
                    
                    delta_covered = 0
                    for nb in old_nbs_set - new_nbs_set:
                        if current_counts[nb] == 1:
                            delta_covered -= 1
                    for nb in new_nbs_set - old_nbs_set:
                        if current_counts[nb] == 0:
                            delta_covered += 1
                    
                    new_covered = current_covered + delta_covered
                    new_cost = get_cost(current_size, new_covered)
                    delta_cost = new_cost - current_cost
                    
                    if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                        current_code[idx] = all_words[new_w_idx]
                        for nb in old_nbs_set:
                            current_counts[nb] -= 1
                        for nb in new_nbs_set:
                            current_counts[nb] += 1
                        current_covered = new_covered
                        current_cost = new_cost
                        update_best()
                        
            else:
                # Add a word to cover an uncovered word
                attempts = 0
                while True:
                    u_idx = random.randint(0, 2186)
                    if current_counts[u_idx] == 0:
                        break
                    attempts += 1
                    if attempts > 100:
                        u_idx = random.randint(0, 2186)
                        break
                        
                new_w_idx = random.choice(neighbors[u_idx])
                new_w = all_words[new_w_idx]
                
                if new_w not in current_code:
                    new_nbs = neighbors[new_w_idx]
                    delta_covered = 0
                    for nb in new_nbs:
                        if current_counts[nb] == 0:
                            delta_covered += 1
                            
                    new_size = current_size + 1
                    new_covered = current_covered + delta_covered
                    new_cost = get_cost(new_size, new_covered)
                    delta_cost = new_cost - current_cost
                    
                    if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                        current_code.append(new_w)
                        for nb in new_nbs:
                            current_counts[nb] += 1
                        current_size = new_size
                        current_covered = new_covered
                        current_cost = new_cost
                        update_best()
                        
        else:
            # Fully covered, minimize size
            r = random.random()
            if r < 0.4:
                # Try Remove
                if current_size > 0:
                    idx = random.randint(0, current_size - 1)
                    w_idx = word_to_idx[current_code[idx]]
                    
                    delta_covered = 0
                    for nb in neighbors[w_idx]:
                        if current_counts[nb] == 1:
                            delta_covered -= 1
                            
                    if delta_covered == 0:
                        # Always accept if no loss
                        current_code.pop(idx)
                        for nb in neighbors[w_idx]:
                            current_counts[nb] -= 1
                        current_size -= 1
                        current_cost = get_cost(current_size, current_covered)
                        update_best()
                    else:
                        new_size = current_size - 1
                        new_covered = current_covered + delta_covered
                        new_cost = get_cost(new_size, new_covered)
                        delta_cost = new_cost - current_cost
                        
                        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                            current_code.pop(idx)
                            for nb in neighbors[w_idx]:
                                current_counts[nb] -= 1
                            current_size = new_size
                            current_covered = new_covered
                            current_cost = new_cost
                            update_best()
            else:
                # Replace to rearrange
                if current_size > 0:
                    idx = random.randint(0, current_size - 1)
                    old_w_idx = word_to_idx[current_code[idx]]
                    
                    # Pick a random new word
                    new_w_idx = random.randint(0, 2186)
                    
                    if new_w_idx == old_w_idx:
                        continue
                        
                    old_nbs_set = set(neighbors[old_w_idx])
                    new_nbs_set = set(neighbors[new_w_idx])
                    
                    delta_covered = 0
                    for nb in old_nbs_set - new_nbs_set:
                        if current_counts[nb] == 1:
                            delta_covered -= 1
                    for nb in new_nbs_set - old_nbs_set:
                        if current_counts[nb] == 0:
                            delta_covered += 1
                            
                    new_covered = current_covered + delta_covered
                    new_cost = get_cost(current_size, new_covered)
                    delta_cost = new_cost - current_cost
                    
                    if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                        current_code[idx] = all_words[new_w_idx]
                        for nb in old_nbs_set:
                            current_counts[nb] -= 1
                        for nb in new_nbs_set:
                            current_counts[nb] += 1
                        current_covered = new_covered
                        current_cost = new_cost
                        update_best()
                        
        T *= cooling

    return [list(w) for w in best_code]
# EVOLVE-BLOCK-END
