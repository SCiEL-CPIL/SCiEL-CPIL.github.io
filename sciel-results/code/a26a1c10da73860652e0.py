import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 256
    V_small = list(range(2, 129))
    V_large = list(range(129, 257))
    
    n_small = len(V_small)
    adj = [set() for _ in range(n_small)]
    
    def lcm(a, b):
        return a // gcd(a, b) * b
        
    for i in range(n_small):
        ai = V_small[i]
        for j in range(i + 1, n_small):
            aj = V_small[j]
            if lcm(ai, aj) <= 256:
                adj[i].add(j)
                adj[j].add(i)
                
    block_lists = []
    for i in range(n_small):
        a = V_small[i]
        blocks = [b for b in V_large if b % a == 0]
        block_lists.append(blocks)
        
    weights_small = [1.0 / a for a in V_small]
    weights_large = {b: 1.0 / b for b in V_large}
    
    order = sorted(range(n_small), key=lambda i: -weights_small[i])
    
    suffix_sums = [0.0] * (n_small + 1)
    for i in range(n_small - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + weights_small[order[i]]
        
    best_score = 0.0
    best_set_s = []
    best_blocked = set()
    node_count = 0

    def dfs(idx, current_sum, chosen, available, blocked_set):
        nonlocal best_score, best_set_s, best_blocked, node_count
        
        if time.time() > deadline - 5.0:
            return
            
        node_count += 1
        
        if idx == n_small:
            score = current_sum
            if score > best_score:
                best_score = score
                best_set_s = list(chosen)
                best_blocked = set(blocked_set)
            return
            
        ub = current_sum + suffix_sums[idx]
        if ub <= best_score:
            return
            
        v_idx = order[idx]
        
        if v_idx not in available:
            dfs(idx + 1, current_sum, chosen, available, blocked_set)
            return
            
        # Include
        new_blocked = set(blocked_set)
        add_penalty = 0.0
        for b in block_lists[v_idx]:
            if b not in new_blocked:
                add_penalty += weights_large[b]
                new_blocked.add(b)
                
        new_sum = current_sum + weights_small[v_idx] - add_penalty
        
        new_available = set(available)
        new_available.remove(v_idx)
        for neighbor in adj[v_idx]:
            if neighbor in new_available:
                new_available.remove(neighbor)
                
        chosen.add(v_idx)
        dfs(idx + 1, new_sum, chosen, new_available, new_blocked)
        chosen.remove(v_idx)
        
        # Exclude
        available.remove(v_idx)
        dfs(idx + 1, current_sum, chosen, available, blocked_set)
        available.add(v_idx)
        
    dfs(0, 0.0, set(), set(range(n_small)), set())
    
    if best_set_s:
        final_set = [V_small[i] for i in best_set_s]
        for b in V_large:
            if b not in best_blocked:
                final_set.append(b)
        return final_set
    else:
        return [2]
