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
    
    # Precompute overlap matrix and edge costs
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
    
    block_len = 2 * (n - 1) + 1
    # cost[i][j] = length contributed when transitioning from perm i to perm j
    cost = [[block_len - overlap_matrix[i][j] for j in range(num_perms)] for i in range(num_perms)]
    
    # Initial order: 0 to num_perms-1
    order = list(range(num_perms))
    
    def calc_length(ord_list):
        total = block_len
        for i in range(len(ord_list) - 1):
            total += cost[ord_list[i]][ord_list[i + 1]]
        return total
    
    cur_len = calc_length(order)
    best_length = cur_len
    best_order = order[:]
    
    # Local Search with incremental length updates
    improved = True
    max_passes = 5000
    passes = 0
    
    while improved and time.time() < deadline and passes < max_passes:
        improved = False
        passes += 1
        
        # Try adjacent swaps with O(1) delta computation
        for i in range(len(order) - 1):
            if time.time() >= deadline:
                break
            a, b, c = order[i-1] if i > 0 else None, order[i], order[i+1]
            d = order[i+2] if i+2 < len(order) else None
            old_cost = 0
            if i > 0:
                old_cost += cost[a][b]
            old_cost += cost[b][c]
            if i+2 < len(order):
                old_cost += cost[c][d]
            new_cost = 0
            if i > 0:
                new_cost += cost[a][c]
            new_cost += cost[c][b]
            if i+2 < len(order):
                new_cost += cost[b][d]
            delta = new_cost - old_cost
            if delta < 0:
                order[i], order[i+1] = order[i+1], order[i]
                cur_len += delta
                if cur_len < best_length:
                    best_length = cur_len
                    best_order = order[:]
                improved = True
        
        if time.time() >= deadline:
            break
            
        # Try 2-opt moves (reverse segment) with O(1) delta
        if not improved:
            improved_2opt = False
            max_seg = min(30, len(order))
            for i in range(len(order) - 2):
                if time.time() >= deadline:
                    break
                for j in range(i + 2, min(i + max_seg, len(order))):
                    # Reversing order[i..j]: boundary edges change
                    # Old: cost[order[i-1]][order[i]] + cost[order[j]][order[j+1]] (if exist)
                    # New: cost[order[i-1]][order[j]] + cost[order[i]][order[j+1]] (if exist)
                    old_delta = 0
                    new_delta = 0
                    if i > 0:
                        old_delta += cost[order[i-1]][order[i]]
                        new_delta += cost[order[i-1]][order[j]]
                    if j + 1 < len(order):
                        old_delta += cost[order[j]][order[j+1]]
                        new_delta += cost[order[i]][order[j+1]]
                    # Internal edges: cost[order[k]][order[k+1]] becomes cost[order[k+1]][order[k]]
                    internal_old = 0
                    internal_new = 0
                    for k in range(i, j):
                        internal_old += cost[order[k]][order[k+1]]
                        internal_new += cost[order[k+1]][order[k]]
                    delta = (new_delta + internal_new) - (old_delta + internal_old)
                    if delta < 0:
                        order[i:j+1] = order[i:j+1][::-1]
                        cur_len += delta
                        if cur_len < best_length:
                            best_length = cur_len
                            best_order = order[:]
                        improved = True
                        improved_2opt = True
                    if time.time() >= deadline:
                        break
                if time.time() >= deadline:
                    break
            if improved_2opt:
                improved = True
        
        if time.time() >= deadline:
            break
        
        # If stuck, try random perturbations
        if not improved:
            # Random perturbation: swap a few random pairs
            num_swaps = min(50, len(order) // 10)
            for _ in range(num_swaps):
                if time.time() >= deadline:
                    break
                i = random.randint(0, len(order) - 2)
                a, b = order[i-1] if i > 0 else None, order[i]
                c, d = order[i+1], order[i+2] if i+2 < len(order) else None
                old_cost = 0
                if i > 0:
                    old_cost += cost[a][b]
                old_cost += cost[b][c]
                if i+2 < len(order):
                    old_cost += cost[c][d]
                new_cost = 0
                if i > 0:
                    new_cost += cost[a][c]
                new_cost += cost[c][b]
                if i+2 < len(order):
                    new_cost += cost[b][d]
                order[i], order[i+1] = order[i+1], order[i]
                cur_len += (new_cost - old_cost)
            if cur_len < best_length:
                best_length = cur_len
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
