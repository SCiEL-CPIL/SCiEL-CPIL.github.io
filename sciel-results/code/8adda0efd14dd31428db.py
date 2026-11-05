import time
import itertools
import sys
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    if n == 1:
        return "1"
    if n == 2:
        return "121"

    # 1. Generate standard superpermutation for n-1
    word_prev = [1]
    for m in range(2, n):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word_prev) - m + 2):
            window = tuple(word_prev[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word_prev = new_word

    # Extract the (n-1)-permutations from word_prev in order
    perms, seen = [], set()
    for i in range(len(word_prev) - n + 2):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    num_perms = len(perms)
    
    # 2. Precompute overlap matrix using NumPy for speed
    blocks = []
    for p in perms:
        blocks.append(list(p) + [n] + list(p))
    
    # Vectorized overlap computation
    block_len = 2 * (n - 1) + 1
    # Convert blocks to numpy array
    blocks_arr = np.array(blocks, dtype=np.int8)  # shape (num_perms, block_len)
    
    M = np.zeros((num_perms, num_perms), dtype=np.int8)
    
    # For each pair (i,j), find longest k such that blocks[i][-k:] == blocks[j][:k]
    # We can check from largest k down to 1
    for k in range(block_len - 1, 0, -1):
        # For all pairs where M[i][j] is still 0, check if suffix of length k matches
        # blocks[i][-k:] is shape (num_perms, k)
        # blocks[j][:k] is shape (num_perms, k)
        # We need to check for each (i,j) if blocks_arr[i, -k:] == blocks_arr[j, :k]
        # This is an outer comparison: for each i, compare its suffix with all j's prefixes
        suffixes = blocks_arr[:, -k:]  # (num_perms, k)
        prefixes = blocks_arr[:, :k]   # (num_perms, k)
        # match[i,j] = all(suffixes[i] == prefixes[j])
        # Use broadcasting: suffixes[:, None, :] == prefixes[None, :, :]
        match = np.all(suffixes[:, None, :] == prefixes[None, :, :], axis=2)
        # Only update where M is still 0
        mask = (M == 0) & match
        M[mask] = k
        if time.time() > deadline:
            break
    
    M_list = M.tolist()

    # 3. Find a good ordering using enhanced local search with 2-opt and SA
    order = list(range(num_perms))
    
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M_list[ord[i]][ord[i+1]]
        return score

    best_order = order[:]
    best_score = calc_score(order)
    current_order = order[:]
    current_score = best_score

    def get_delta_swap(ord, i):
        x = ord[i]
        y = ord[i+1]
        left = ord[i-1] if i > 0 else -1
        right = ord[i+2] if i+2 < len(ord) else -1
        delta = 0
        if i > 0:
            delta -= M_list[left][x]
        delta -= M_list[x][y]
        if i+1 < len(ord)-1:
            delta -= M_list[y][right]
        if i > 0:
            delta += M_list[left][y]
        delta += M_list[y][x]
        if i+1 < len(ord)-1:
            delta += M_list[x][right]
        return delta

    def get_delta_2opt(ord, i, j):
        # Reverse segment ord[i..j]
        # Old edges: (ord[i-1], ord[i]), (ord[i], ord[i+1]), ..., (ord[j-1], ord[j]), (ord[j], ord[j+1])
        # New edges: (ord[i-1], ord[j]), (ord[j], ord[j-1]), ..., (ord[i+1], ord[i]), (ord[i], ord[j+1])
        if i >= j:
            return 0
        left = ord[i-1] if i > 0 else -1
        right = ord[j+1] if j+1 < len(ord) else -1
        delta = 0
        # Remove old boundary edges
        if i > 0:
            delta -= M_list[left][ord[i]]
        delta -= M_list[ord[i]][ord[i+1]] if i+1 <= j else 0
        if j+1 < len(ord):
            delta -= M_list[ord[j]][right]
        # Add new boundary edges
        if i > 0:
            delta += M_list[left][ord[j]]
        delta += M_list[ord[i]][ord[j]] if i+1 <= j else 0
        if j+1 < len(ord):
            delta += M_list[ord[j+1] if j+1 < len(ord) else ord[j]][ord[i]] if False else 0
        # Actually, let's do this more carefully
        # Old: ... left, ord[i], ord[i+1], ..., ord[j], right ...
        # New: ... left, ord[j], ord[j-1], ..., ord[i], right ...
        # Removed edges: (left, ord[i]), (ord[i], ord[i+1]), ..., (ord[j-1], ord[j]), (ord[j], right)
        # Added edges: (left, ord[j]), (ord[j], ord[j-1]), ..., (ord[i+1], ord[i]), (ord[i], right)
        
        delta = 0
        # Remove old edges
        if i > 0:
            delta -= M_list[left][ord[i]]
        for k in range(i, j):
            delta -= M_list[ord[k]][ord[k+1]]
        if j+1 < len(ord):
            delta -= M_list[ord[j]][right]
        
        # Add new edges
        if i > 0:
            delta += M_list[left][ord[j]]
        for k in range(i, j):
            delta += M_list[ord[k+1]][ord[k]]
        if j+1 < len(ord):
            delta += M_list[ord[i]][right]
        
        return delta

    random.seed(42)
    
    # Phase 1: Hill climbing with swaps
    max_iterations = 200000
    iters = 0
    no_improve = 0
    while iters < max_iterations and time.time() < deadline - 20:
        iters += 1
        i = random.randint(0, num_perms - 2)
        delta = get_delta_swap(current_order, i)
        if delta >= 0:
            current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
            current_score += delta
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_order = current_order[:]
        else:
            no_improve += 1
            if no_improve > 5000:
                break
    
    # Phase 2: 2-opt local search
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for i in range(num_perms - 2):
            if time.time() > deadline - 15:
                break
            for j in range(i + 2, min(i + 50, num_perms - 1)):
                delta = get_delta_2opt(current_order, i, j)
                if delta > 0:
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]
                    break  # Restart i loop after a successful move
    
    # Phase 3: Simulated Annealing
    if time.time() < deadline - 10:
        T = 5.0
        T_min = 0.1
        cooling = 0.9995
        sa_iters = 0
        max_sa = 100000
        while T > T_min and sa_iters < max_sa and time.time() < deadline - 5:
            sa_iters += 1
            move_type = random.random()
            if move_type < 0.7:
                # Swap
                i = random.randint(0, num_perms - 2)
                delta = get_delta_swap(current_order, i)
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]
            else:
                # 2-opt
                i = random.randint(0, num_perms - 3)
                j = random.randint(i + 2, min(i + 100, num_perms - 1))
                delta = get_delta_2opt(current_order, i, j)
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]
            T *= cooling

    # 4. Construct word from best_order using precomputed overlaps
    word = []
    for idx, p_idx in enumerate(best_order):
        block = blocks[p_idx]
        if idx == 0:
            word = block[:]
        else:
            # Use precomputed overlap with previous block
            prev_idx = best_order[idx - 1]
            ov = M_list[prev_idx][p_idx]
            word += block[ov:]
            
    S = word

    # 5. Pruning Loop to remove redundant symbols
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts
    perm_counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        perm_counts[p] = perm_counts.get(p, 0) + 1
        
    def is_valid_incremental(S, i, perm_counts):
        L = len(S)
        if L <= n:
            return False, None, None
        
        start_min = max(0, i - n + 1)
        start_max = min(i, len(S) - n)
        
        deltas = {}
        
        # Old windows to remove
        for j in range(start_min, start_max + 1):
            w = tuple(S[j:j+n])
            deltas[w] = deltas.get(w, 0) - 1
            
        # New windows to add
        for j in range(max(0, i - n + 1), i):
            if j > len(S) - 1 - n:
                continue
            part1 = S[j:i]
            part2 = S[i+1:j+n+1]
            new_win = tuple(part1 + part2)
            deltas[new_win] = deltas.get(new_win, 0) + 1
            
        for p, delta in deltas.items():
            if p in all_perms:
                new_count = perm_counts.get(p, 0) + delta
                if new_count < 1:
                    return False, None, None
                    
        return True, S[:i] + S[i+1:], deltas

    # Pruning - multiple passes
    for _pass in range(200):
        if time.time() > deadline - 3:
            break
        removed = False
        # Iterate forwards and backwards
        for direction in range(2):
            if time.time() > deadline - 3:
                break
            if direction == 0:
                indices = range(len(S) - 1, -1, -1)
            else:
                indices = range(len(S))
            for i in indices:
                if time.time() > deadline - 3:
                    break
                if len(S) <= n:
                    break
                valid, new_S, deltas = is_valid_incremental(S, i, perm_counts)
                if valid:
                    S = new_S
                    for p, delta in deltas.items():
                        perm_counts[p] = perm_counts.get(p, 0) + delta
                        if perm_counts[p] == 0:
                            del perm_counts[p]
                    removed = True
                    break
        if not removed:
            break
            
    return "".join(map(str, S))
