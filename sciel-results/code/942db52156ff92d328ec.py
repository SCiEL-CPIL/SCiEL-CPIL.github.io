import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    time_limit = 115
    
    if n <= 3:
        if n == 1: return "1"
        if n == 2: return "121"
        if n == 3: return "123121321"
        
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Generate standard order
    def get_standard_order(n):
        # Re-use the logic to get the word, then extract order
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
        
        order = []
        seen = set()
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if p not in seen:
                seen.add(p)
                order.append(p)
        return [perm_to_idx[p] for p in order]

    current_order = get_standard_order(n)
    
    # Cache for overlaps
    overlap_cache = {}
    
    def get_overlap(i, j):
        key = (i, j)
        if key in overlap_cache:
            return overlap_cache[key]
        p1 = perms[i]
        p2 = perms[j]
        k = 0
        for k_check in range(n - 1, 0, -1):
            if p1[n - k_check:] == p2[:k_check]:
                k = k_check
                break
        overlap_cache[key] = k
        return k

    def calc_length(order):
        if not order: return 0
        length = n
        prev = order[0]
        for j in range(1, len(order)):
            curr = order[j]
            length += n - get_overlap(prev, curr)
            prev = curr
        return length

    current_len = calc_length(current_order)
    
    # Hill Climbing
    t_start = time.time()
    while time.time() - t_start < time_limit:
        i, j = random.sample(range(N), 2)
        
        # Get neighbors
        prev_i = current_order[i - 1] if i > 0 else None
        next_i = current_order[i + 1] if i < N - 1 else None
        prev_j = current_order[j - 1] if j > 0 else None
        next_j = current_order[j + 1] if j < N - 1 else None
        
        A = current_order[i]
        B = current_order[j]
        
        cost_removed = 0
        if prev_i is not None:
            cost_removed += n - get_overlap(prev_i, A)
        if next_i is not None:
            cost_removed += n - get_overlap(A, next_i)
        if prev_j is not None:
            cost_removed += n - get_overlap(prev_j, B)
        if next_j is not None:
            cost_removed += n - get_overlap(B, next_j)
            
        cost_added = 0
        if prev_i is not None:
            cost_added += n - get_overlap(prev_i, B)
        if next_i is not None:
            cost_added += n - get_overlap(B, next_i)
        if prev_j is not None:
            cost_added += n - get_overlap(prev_j, A)
        if next_j is not None:
            cost_added += n - get_overlap(A, next_j)
            
        delta = cost_added - cost_removed
        
        if delta < 0:
            current_order[i], current_order[j] = B, A
            current_len += delta

    # Build final word
    def build_word(order):
        if not order: return ""
        word = list(perms[order[0]])
        for j in range(1, len(order)):
            prev_idx = order[j-1]
            curr_idx = order[j]
            k = get_overlap(prev_idx, curr_idx)
            word.extend(perms[curr_idx][k:])
        return "".join(map(str, word))

    return build_word(current_order)
