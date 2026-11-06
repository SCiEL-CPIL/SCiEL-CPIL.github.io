# EVOLVE-BLOCK-START
import time
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 85.0
    
    N = 128
    nums = list(range(1, N + 1))
    n = N
    
    # Precompute lcm for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    def get_lcm(a, b):
        if a <= b:
            return lcm_table[(a, b)]
        else:
            return lcm_table[(b, a)]
    
    # Precompute conflict: conflict[i][j] = bitmask of k such that {i+1, j+1, k+1} is a bad triple
    # 0-indexed: i, j, k in 0..127
    conflict = [[0] * n for _ in range(n)]
    
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            L = lcm_table[(a, b)]
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                c = k + 1
                if get_lcm(a, c) == L and get_lcm(b, c) == L:
                    mask |= (1 << k)
            conflict[i][j] = mask
            conflict[j][i] = mask
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(x, current_mask):
        """Check if element x (0-indexed) can be added to set represented by current_mask."""
        m = current_mask
        while m:
            lsb = m & (-m)
            a = lsb.bit_length() - 1
            if a != x:
                if a < x:
                    cx = conflict[a][x]
                else:
                    cx = conflict[x][a]
                # Check if cx has any bit set in current_mask (other than a and x)
                if cx & (current_mask & ~(1 << a) & ~(1 << x)):
                    return False
            m ^= lsb
        return True
    
    # Greedy to get initial solution
    def greedy(order):
        selected = []
        selected_mask = 0
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x - 1, selected_mask):
                selected.append(x)
                selected_mask |= (1 << (x - 1))
        return selected
    
    best = []
    
    s1 = greedy(nums)
    if len(s1) > len(best):
        best = s1
    
    if time.time() > deadline:
        return best if best else [1]
    
    s2 = greedy(nums[::-1])
    if len(s2) > len(best):
        best = s2
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Convert best to mask
    best_mask = 0
    for x in best:
        best_mask |= (1 << (x - 1))
    
    # Greedy add: try adding each remaining element
    improved_mask = best_mask
    improved_list = list(best)
    
    for x in range(1, N + 1):
        if (1 << (x - 1)) & improved_mask:
            continue
        if can_add(x - 1, improved_mask):
            improved_mask |= (1 << (x - 1))
            improved_list.append(x)
    
    best_found = len(improved_list)
    best_solution = set(improved_list)
    
    # Local search: try removing one element and adding others
    for e in list(improved_list):
        if time.time() > deadline:
            break
        candidate_mask = improved_mask & ~(1 << (e - 1))
        candidate_list = [x for x in improved_list if x != e]
        
        for x in range(1, N + 1):
            if x == e or (1 << (x - 1)) & candidate_mask:
                continue
            if can_add(x - 1, candidate_mask):
                candidate_mask |= (1 << (x - 1))
                candidate_list.append(x)
        
        if len(candidate_list) > best_found:
            best_found = len(candidate_list)
            best_solution = set(candidate_list)
            improved_mask = candidate_mask
            improved_list = candidate_list
    
    result = sorted(best_solution)
    return result if result else [1]
# EVOLVE-BLOCK-END
