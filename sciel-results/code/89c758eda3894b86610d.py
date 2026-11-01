import itertools
import time
import random

def construct(n):
    if n < 7:
        # Use the classic recursive construction for small n
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n != 7:
        # Fallback for n > 7, though task implies n=7
        # Just return a valid but likely long superperm
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # n == 7
    # Generate all permutations of 1..6
    perms = list(itertools.permutations(range(1, 7)))
    N = len(perms) # 720
    
    # Precompute overlap matrix
    # M[i][j] = overlap length between perms[i] and perms[j]
    # Overlap is max k such that perms[i][-k:] == perms[j][:k]
    M = [[0]*N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            max_k = 5
            for k in range(max_k, -1, -1):
                if pi[-k:] == pj[:k]:
                    M[i][j] = k
                    break

    # Initial order: use the standard order from the recursive construction for n=6
    # This provides a good starting point.
    # Generate standard n=6 superperm to get order
    word6 = [1]
    for m in range(2, 7):
        perms_m, seen_m = [], set()
        for i in range(len(word6) - m + 2):
            window = tuple(word6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen_m:
                seen_m.add(window)
                perms_m.append(window)
        new6 = []
        for p in perms_m:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new6), len(block)), -1, -1) if new6[len(new6) - o:] == block[:o])
            new6 += block[overlap:]
        word6 = new6
        
    # Extract order of perms from word6
    order = []
    seen_order = set()
    for i in range(len(word6) - 6 + 1):
        window = tuple(word6[i:i+6])
        if len(set(window)) == 6 and window not in seen_order:
            seen_order.add(window)
            # Find index of this window in perms
            # This is O(N) per window, total O(N^2) ~ 500k, acceptable
            # But we can use a dict for mapping
            pass
            
    # Better: map perm to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    for i in range(len(word6) - 6 + 1):
        window = tuple(word6[i:i+6])
        if len(set(window)) == 6:
            if window in perm_to_idx:
                idx = perm_to_idx[window]
                if idx not in seen_order:
                    seen_order.add(idx)
                    order.append(idx)
    
    # Ensure all perms are in order
    if len(order) != N:
        # Add missing ones
        missing = [i for i in range(N) if i not in seen_order]
        order.extend(missing)
        
    # Local Search to maximize sum of overlaps
    # Cost = sum M[order[i]][order[i+1]]
    # We want to maximize this sum.
    
    def calc_total_overlap(ord):
        total = 0
        for i in range(len(ord) - 1):
            total += M[ord[i]][ord[i+1]]
        return total

    current_overlap = calc_total_overlap(order)
    best_order = order[:]
    best_overlap = current_overlap
    
    start_time = time.time()
    time_limit = 110 # Leave 10s for construction and return
    
    # Hill Climbing / Simulated Annealing
    # Simple Hill Climbing first
    improved = True
    while improved:
        improved = False
        for i in range(N - 1):
            # Swap i and i+1
            a = order[i]
            b = order[i+1]
            prev = order[i-1] if i > 0 else -1
            next = order[i+2] if i+2 < N else -1
            
            old_cost = 0
            if prev != -1:
                old_cost += M[prev][a]
            old_cost += M[a][b]
            if next != -1:
                old_cost += M[b][next]
                
            new_cost = 0
            if prev != -1:
                new_cost += M[prev][b]
            new_cost += M[b][a]
            if next != -1:
                new_cost += M[a][next]
                
            if new_cost > old_cost:
                order[i], order[i+1] = b, a
                current_overlap += (new_cost - old_cost)
                improved = True
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_order = order[:]
                    
    # If time remains, do random swaps (Simulated Annealing style)
    if time.time() - start_time < time_limit:
        # Use a simple random restart or random walk
        # Let's try random swaps
        while time.time() - start_time < time_limit:
            i = random.randint(0, N - 2)
            j = i + 1
            # Same swap logic
            a = order[i]
            b = order[j]
            prev = order[i-1] if i > 0 else -1
            next = order[i+2] if i+2 < N else -1
            
            old_cost = 0
            if prev != -1:
                old_cost += M[prev][a]
            old_cost += M[a][b]
            if next != -1:
                old_cost += M[b][next]
                
            new_cost = 0
            if prev != -1:
                new_cost += M[prev][b]
            new_cost += M[b][a]
            if next != -1:
                new_cost += M[a][next]
                
            if new_cost >= old_cost: # Accept if not worse
                order[i], order[j] = b, a
                current_overlap += (new_cost - old_cost)
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_order = order[:]
                    # Occasionally restart from best?
                    # No, just keep going.

    # Construct the word from best_order
    # Word starts with first block
    # B_i = P_i 7 P_i
    # Overlap between B_i and B_{i+1} is M[best_order[i]][best_order[i+1]]
    
    word = []
    for idx in best_order:
        p = perms[idx]
        block = list(p) + [7] + list(p)
        if not word:
            word = block
        else:
            overlap = M[best_order[best_order.index(idx) - 1]][idx] if best_order.index(idx) > 0 else 0
            # Wait, M is indexed by perm index.
            # We need the overlap between the previous perm and current perm.
            # Let's just iterate with index
            pass
    
    # Better construction loop
    word = []
    prev_idx = -1
    for i, idx in enumerate(best_order):
        p = perms[idx]
        block = list(p) + [7] + list(p)
        if i == 0:
            word = block
        else:
            overlap = M[prev_idx][idx]
            word.extend(block[overlap:])
        prev_idx = idx
        
    return "".join(map(str, word))
