import time
import random

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer
    
    def get_word_and_perms(m):
        if m == 1:
            return "1", [(1,)]
        if m == 2:
            return "121", [(1,2), (2,1)]
        
        prev_word, prev_perms = get_word_and_perms(m-1)
        
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        perms = []
        seen = set()
        for i in range(len(new_word) - m + 1):
            window = tuple(new_word[i:i + m])
            if len(set(window)) == m and window not in seen:
                seen.add(window)
                perms.append(window)
                
        return "".join(map(str, new_word)), perms
    
    _, perms = get_word_and_perms(n-1)
    num_perms = len(perms)
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    max_k = n - 2
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        pi = perms[i]
        for j in range(num_perms):
            pj = perms[j]
            for k in range(max_k, -1, -1):
                if pi[-k:] == pj[:k]:
                    overlap_matrix[i][j] = k
                    break
    
    # Initial order: 0 to num_perms-1
    order = list(range(num_perms))
    
    def calc_length(ord_list):
        total = 2 * (n - 1) + 1
        for i in range(len(ord_list) - 1):
            u = ord_list[i]
            v = ord_list[i + 1]
            total += (2 * (n - 1) + 1) - overlap_matrix[u][v]
        return total
    
    best_length = calc_length(order)
    best_order = order[:]
    
    # Local Search with multiple move types
    improved = True
    move_types = ['swap', '2opt', 'random_swap']
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adjacent swaps
        for i in range(len(order) - 1):
            if time.time() >= deadline:
                break
            order[i], order[i+1] = order[i+1], order[i]
            new_len = calc_length(order)
            if new_len < best_length:
                best_length = new_len
                best_order = order[:]
                improved = True
            else:
                order[i], order[i+1] = order[i+1], order[i]
        
        if time.time() >= deadline:
            break
            
        # Try 2-opt moves (reverse segment)
        if not improved:
            improved_2opt = False
            for i in range(len(order) - 2):
                if time.time() >= deadline:
                    break
                for j in range(i + 2, min(i + 20, len(order))):  # Limit segment length for speed
                    segment = order[i:j+1]
                    order[i:j+1] = segment[::-1]
                    new_len = calc_length(order)
                    if new_len < best_length:
                        best_length = new_len
                        best_order = order[:]
                        improved = True
                        improved_2opt = True
                    else:
                        order[i:j+1] = segment
                if time.time() >= deadline:
                    break
            if improved_2opt:
                improved = True
        
        if time.time() >= deadline:
            break
        
        # If stuck, try some random restarts / perturbations
        if not improved:
            # Random perturbation: swap a few random pairs
            for _ in range(100):
                if time.time() >= deadline:
                    break
                i = random.randint(0, len(order) - 2)
                order[i], order[i+1] = order[i+1], order[i]
            new_len = calc_length(order)
            if new_len < best_length:
                best_length = new_len
                best_order = order[:]
                improved = True
    
    # Construct final string
    final_word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(final_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if final_word[len(final_word) - o:] == block[:o]:
                overlap = o
                break
        final_word += block[overlap:]
        
    return "".join(map(str, final_word))
