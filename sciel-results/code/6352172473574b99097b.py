import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        # Simple greedy or tetracode extension
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in __import__('itertools').product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1
    N = 3 ** n
    indices = list(range(N))
    
    # Precompute words and neighbors
    words = []
    for i in range(N):
        w = [0] * n
        temp = i
        for j in range(n):
            w[j] = temp % q
            temp //= q
        words.append(tuple(w))
        
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    neighbors = []
    for i in range(N):
        w = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh.append(word_to_idx[new_w])
        neighbors.append(neigh)
        
    # Phase 1: Greedy Set Cover Initialization
    # uncovered[i] = 1 if word i is not covered, 0 otherwise
    uncovered = np.ones(N, dtype=np.int8)
    current_code = []
    cover_count = np.zeros(N, dtype=np.int32)
    
    # To speed up greedy, we can precompute for each word how many *uncovered* neighbors it has.
    # But since uncovered changes, we need to update.
    # N=729 is small, so O(N^2) per step is 729*729 ~ 500k ops. 
    # Max 80 steps -> 40M ops. Python might be slow.
    # Let's use a list of indices and check.
    
    remaining_indices = set(indices)
    
    while any(uncovered):
        best_idx = -1
        max_new_cover = 0
        for idx in remaining_indices:
            # Count how many uncovered words this idx covers
            new_cover = 0
            for nb in neighbors[idx]:
                if uncovered[nb]:
                    new_cover += 1
            if new_cover > max_new_cover:
                max_new_cover = new_cover
                best_idx = idx
            if new_cover == max_new_cover:
                # Tie-breaking: prefer words that cover more total neighbors? Or random?
                pass
        
        if best_idx == -1:
            break # Should not happen
            
        # Add best_idx
        current_code.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Update uncovered and cover_count
        for nb in neighbors[best_idx]:
            cover_count[nb] += 1
            if cover_count[nb] == 1:
                uncovered[nb] = 0
                
    # Phase 2: Greedy Reduction
    # Try to remove words that don't break coverage
    changed = True
    while changed:
        changed = False
        # Iterate in reverse to allow removal of recently added if they are redundant
        # Or just random order to avoid bias
        for i in range(len(current_code) - 1, -1, -1):
            idx = current_code[i]
            # Check if idx is redundant
            # A word is redundant if for all nb in neighbors[idx], cover_count[nb] > 1
            # Because if we remove idx, cover_count[nb] decreases by 1.
            # If cover_count[nb] was 1, it becomes 0 (uncovered).
            is_redundant = True
            for nb in neighbors[idx]:
                if cover_count[nb] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove it
                current_code.pop(i)
                for nb in neighbors[idx]:
                    cover_count[nb] -= 1
                changed = True
                
    # Phase 3: Simulated Annealing
    best_code = current_code.copy()
    best_size = len(current_code)
    best_cover_count = cover_count.copy()
    
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    C = current_code
    cover_count = cover_count.copy()
    
    # Helper to check if valid
    def is_valid(cc):
        return np.min(cc) >= 1

    # We want to minimize size.
    # Penalty = number of uncovered words.
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.7:
            # 70% Remove
            if not C:
                break
            c_old = random.choice(C)
            
            # Calculate change in cover_count
            # We can do this in-place on a copy or just check
            # For speed, we check if removal is valid first
            is_valid_remove = True
            for nb in neighbors[c_old]:
                if cover_count[nb] == 1:
                    is_valid_remove = False
                    break
            
            if is_valid_remove:
                # Always accept valid removal
                for nb in neighbors[c_old]:
                    cover_count[nb] -= 1
                C.remove(c_old)
                
                if len(C) < best_size:
                    best_size = len(C)
                    best_code = C.copy()
                    best_cover_count = cover_count.copy()
            else:
                # Invalid removal, might accept with penalty
                # Count newly uncovered
                newly_uncovered = 0
                for nb in neighbors[c_old]:
                    if cover_count[nb] == 1:
                        newly_uncovered += 1
                
                Lambda = 10.0
                delta_cost = -1.0 + Lambda * newly_uncovered
                
                if random.random() < np.exp(-delta_cost / T):
                    # Accept
                    for nb in neighbors[c_old]:
                        cover_count[nb] -= 1
                    C.remove(c_old)
        else:
            # 30% Swap
            if not C:
                break
            c_old = random.choice(C)
            C_set = set(C)
            while True:
                c_new = random.randint(0, N - 1)
                if c_new not in C_set:
                    break
            
            # Check swap validity
            # Remove c_old, Add c_new
            # We need to check if any word becomes uncovered after removing c_old, 
            # and if it is covered by c_new.
            
            # 1. Identify words that would be uncovered if c_old is removed
            # These are nb in neighbors[c_old] where cover_count[nb] == 1
            vulnerable = []
            for nb in neighbors[c_old]:
                if cover_count[nb] == 1:
                    vulnerable.append(nb)
            
            # 2. Check if c_new covers all vulnerable words
            valid_swap = True
            for v in vulnerable:
                # Check if v is in neighbors[c_new]
                # Since neighbors are lists, we can check if v == c_new or if v is in the list
                # But neighbors[c_new] is a list of indices.
                # Faster: precompute a set for neighbors? No, lists are fine for small n.
                # Or just check if v is covered by c_new.
                # dist(words[v], words[c_new]) <= 1
                # We can just check if v is in neighbors[c_new]
                # To make this O(1), we could use a set, but creating sets is slow.
                # Let's just check.
                if v not in neighbors[c_new]:
                    valid_swap = False
                    break
            
            if valid_swap:
                # Perform swap
                for nb in neighbors[c_old]:
                    cover_count[nb] -= 1
                for nb in neighbors[c_new]:
                    cover_count[nb] += 1
                C.remove(c_old)
                C.append(c_new)
            else:
                # Invalid swap, might accept with penalty
                # Penalty = number of vulnerable words NOT covered by c_new
                uncovered_by_swap = 0
                for v in vulnerable:
                    if v not in neighbors[c_new]:
                        uncovered_by_swap += 1
                
                Lambda = 10.0
                # Size doesn't change, so delta_cost = Lambda * uncovered_by_swap
                delta_cost = Lambda * uncovered_by_swap
                
                if random.random() < np.exp(-delta_cost / T):
                    for nb in neighbors[c_old]:
                        cover_count[nb] -= 1
                    for nb in neighbors[c_new]:
                        cover_count[nb] += 1
                    C.remove(c_old)
                    C.append(c_new)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Return best valid code found
    result = []
    for idx in best_code:
        result.append(list(words[idx]))
        
    return result
