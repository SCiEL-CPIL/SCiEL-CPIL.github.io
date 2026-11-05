import time
import sys
import random
import math
import numpy as np
from itertools import permutations as all_perms

def construct(n):
    """
    Constructs a superpermutation for n symbols.
    For n=7, uses a probabilistic greedy construction with urgency penalties
    to find a shorter ordering of the 6-permutations.
    """
    if n == 1:
        return "1"
    
    def classic_construct(n):
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
                max_ov = min(len(new), len(block))
                overlap = 0
                for o in range(max_ov, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    if n != 7:
        return "".join(map(str, classic_construct(n)))
    
    # Specialized optimization for n=7
    # 1. Generate the superpermutation for n=6
    word_6 = classic_construct(6)
    
    # 2. Extract the list of 6-permutations in the order they appear in word_6
    perms_6, seen = [], set()
    for i in range(len(word_6) - 6 + 1):
        window = tuple(word_6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms_6.append(window)
            
    num_perms = len(perms_6) # Should be 720
    
    # 3. Precompute overlap matrix M[i][j] = longest suffix of perms_6[i] matching prefix of perms_6[j]
    M = np.zeros((num_perms, num_perms), dtype=np.int8)
    
    for i in range(num_perms):
        p1 = perms_6[i]
        for j in range(num_perms):
            p2 = perms_6[j]
            for k in range(6, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break
    
    M_list = M.tolist()
    
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M_list[ord[i]][ord[i+1]]
        return score
    
    # Probabilistic greedy construction with urgency penalties
    def probabilistic_greedy(temperature=0.1, urgency_weight=0.5, deadline=None):
        """
        Builds a permutation order using probabilistic greedy with urgency.
        At each step, choose next 6-perm based on:
        - Overlap with current last perm (immediate gain)
        - Urgency: how many remaining perms become unreachable if we don't visit them soon
        """
        if deadline is None:
            deadline = time.time() + 5.0
        
        visited = [False] * num_perms
        path = []
        remaining = set(range(num_perms))
        
        # Start with a random perm
        start = random.choice(list(remaining))
        path.append(start)
        visited[start] = True
        remaining.remove(start)
        
        for step in range(num_perms - 1):
            if time.time() > deadline:
                break
            
            last = path[-1]
            best_next = -1
            best_val = float('-inf')
            candidates = []
            
            for nxt in remaining:
                # Immediate gain: overlap
                overlap = M_list[last][nxt]
                
                # Urgency: count how many unvisited perms have low overlap with nxt
                # (i.e., they would be hard to reach later)
                urgency = 0
                for r in remaining:
                    if r == nxt:
                        continue
                    # If overlap from nxt to r is low, r is becoming urgent
                    if M_list[nxt][r] < 3:
                        urgency += 1
                
                # Combined score
                val = overlap - urgency_weight * urgency / max(1, len(remaining))
                
                candidates.append((val, nxt))
            
            # Apply temperature-based selection
            if temperature > 0 and candidates:
                # Soft-max selection
                vals = [c[0] for c in candidates]
                max_val = max(vals)
                exps = [math.exp((v - max_val) / temperature) for v in vals]
                total = sum(exps)
                r = random.random() * total
                cumsum = 0
                chosen_idx = 0
                for i, e in enumerate(exps):
                    cumsum += e
                    if r <= cumsum:
                        chosen_idx = i
                        break
                best_next = candidates[chosen_idx][1]
            else:
                best_next = max(candidates, key=lambda x: x[0])[1]
            
            path.append(best_next)
            visited[best_next] = True
            remaining.remove(best_next)
        
        return path
    
    # Local search with multiple move types
    def local_search(order, max_iters=5000, deadline=None):
        current_order = order[:]
        current_score = calc_score(current_order)
        best_order = current_order[:]
        best_score = current_score
        
        if deadline is None:
            deadline = time.time() + 5.0
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            new_order = current_order[:]
            delta_score = 0
            
            if move_type < 0.4:
                # Swap adjacent
                i = random.randint(0, num_perms - 2)
                a = current_order[i]
                b = current_order[i+1]
                
                if i > 0:
                    prev = current_order[i-1]
                    old_prev = M_list[prev][a]
                    new_prev = M_list[prev][b]
                else:
                    old_prev = 0
                    new_prev = 0
                    
                if i+1 < num_perms - 1:
                    next_idx = current_order[i+2]
                    old_next = M_list[b][next_idx]
                    new_next = M_list[a][next_idx]
                else:
                    old_next = 0
                    new_next = 0
                    
                old_mid = M_list[a][b]
                new_mid = M_list[b][a]
                
                delta_score = (new_prev + new_mid + new_next) - (old_prev + old_mid + old_next)
                
                new_order[i], new_order[i+1] = b, a
                
            else:
                # Reverse segment i..j
                i = random.randint(0, num_perms - 2)
                j = random.randint(i + 1, num_perms - 1)
                
                if i > 0:
                    prev = current_order[i-1]
                    old_left = M_list[prev][current_order[i]]
                    new_left = M_list[prev][current_order[j]]
                else:
                    old_left = 0
                    new_left = 0
                    
                if j < num_perms - 1:
                    next_idx = current_order[j+1]
                    old_right = M_list[current_order[j]][next_idx]
                    new_right = M_list[current_order[i]][next_idx]
                else:
                    old_right = 0
                    new_right = 0
                    
                # Internal
                old_internal = 0
                new_internal = 0
                for k in range(i, j):
                    old_internal += M_list[current_order[k]][current_order[k+1]]
                    new_internal += M_list[current_order[k+1]][current_order[k]]
                    
                delta_score = (new_left + new_right + new_internal) - (old_left + old_right + old_internal)
                
                new_order[i:j+1] = new_order[i:j+1][::-1]
            
            new_score = current_score + delta_score
            
            if new_score > current_score:
                current_order = new_order
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
            else:
                if random.random() < 0.1:
                    current_order = new_order
                    current_score = new_score
        
        return best_order, best_score
    
    # Main search: multiple restarts with probabilistic greedy + local search
    start_time = time.time()
    deadline = start_time + 110.0
    
    random.seed(42)
    
    best_order = None
    best_score = -1
    
    # First, try the standard order
    standard_order = list(range(num_perms))
    std_score = calc_score(standard_order)
    if std_score > best_score:
        best_score = std_score
        best_order = standard_order[:]
    
    # Try several probabilistic greedy initializations with varying parameters
    num_restarts = 0
    while time.time() < deadline:
        # Vary temperature and urgency weight
        temp = random.uniform(0.01, 0.5)
        urg_w = random.uniform(0.1, 2.0)
        
        g_order = probabilistic_greedy(temperature=temp, urgency_weight=urg_w, deadline=deadline)
        
        if len(g_order) < num_perms:
            # Incomplete, fill in remaining
            visited = set(g_order)
            for p in range(num_perms):
                if p not in visited:
                    g_order.append(p)
        
        g_score = calc_score(g_order)
        
        # Local search from greedy
        ls_order, ls_score = local_search(g_order, max_iters=1500, deadline=deadline)
        
        if ls_score > best_score:
            best_score = ls_score
            best_order = ls_order[:]
        
        num_restarts += 1
        
        # Also try local search from standard order periodically
        if num_restarts % 3 == 0:
            ls_order2, ls_score2 = local_search(standard_order, max_iters=1500, deadline=deadline)
            if ls_score2 > best_score:
                best_score = ls_score2
                best_order = ls_order2[:]
    
    # 5. Construct word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            ov = M_list[best_order[idx-1]][p_idx]
            word += block[ov:]
            
    return "".join(map(str, word))

if __name__ == "__main__":
    pass
