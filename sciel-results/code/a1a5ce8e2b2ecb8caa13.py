import time
import random
import math
from itertools import permutations
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # For small n, use the standard construction which is optimal
    if n <= 3:
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
                overlap = 0
                for k in range(min(len(new), len(block)), 0, -1):
                    if new[len(new) - k:] == block[:k]:
                        overlap = k
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n >= 4, use optimization
    # We model the problem as finding an ordering of permutations of 1..n-1
    # such that the total overlap between consecutive blocks is maximized.
    # Block for permutation p of 1..n-1 is: p + [n] + p
    
    # 1. Generate all permutations of 1..n-1
    base_symbols = list(range(1, n))
    perms = list(permutations(base_symbols))
    M = len(perms)
    
    # 2. Convert to tuples for hashing and comparison
    perm_tuples = [tuple(p) for p in perms]
    
    # 3. Compute overlap matrix
    # O[i][j] = max k such that perm_tuples[i][-k:] == perm_tuples[j][:k]
    # Note: The block is p + [n] + p.
    # The suffix of block i is ... p[i][-k:]
    # The prefix of block j is p[j][:k] ...
    # So we just need the overlap between the suffix of p_i and prefix of p_j.
    
    overlap_matrix = np.zeros((M, M), dtype=np.int8)
    
    # Optimization: Group permutations by their last k and first k elements
    # For n=7, n-1=6. Max overlap is 6.
    # We can compute this in O(M * n^2) or similar.
    # Let's just do it simply for now, M=5040, M^2 = 25M ops. Might be slow in Python.
    # Let's use a dictionary approach for each length k.
    
    # For each k from n-1 down to 1:
    # Check if suffix of i matches prefix of j.
    
    # Better:
    # For each permutation p, store its suffixes and prefixes.
    # Actually, for a fixed k, we can map (suffix_k) -> list of indices, (prefix_k) -> list of indices.
    # If suffix_k(i) == prefix_k(j), then overlap is at least k.
    
    # We want the MAX overlap. So we iterate k from n-1 down to 1.
    # For each k, we check matches. If a match is found for pair (i,j), set O[i][j] = k and mark as found.
    
    found = np.zeros((M, M), dtype=bool)
    
    for k in range(n - 1, 0, -1):
        # Map suffix of length k to indices
        suffix_map = {}
        prefix_map = {}
        
        for idx, p in enumerate(perm_tuples):
            suff = p[-k:]
            pref = p[:k]
            if suff not in suffix_map:
                suffix_map[suff] = []
            suffix_map[suff].append(idx)
            
            if pref not in prefix_map:
                prefix_map[pref] = []
            prefix_map[pref].append(idx)
        
        # Iterate over all suffixes
        for suff, i_list in suffix_map.items():
            # Check if this suffix is a prefix for some j
            if suff in prefix_map:
                j_list = prefix_map[suff]
                for i in i_list:
                    for j in j_list:
                        if not found[i][j]:
                            overlap_matrix[i][j] = k
                            found[i][j] = True
    
    # 4. Greedy construction of initial order
    current_order = list(range(M))
    unvisited = set(range(M))
    unvisited.remove(0)
    current_order = [0]
    last = 0
    while unvisited:
        max_ov = -1
        best_next = None
        for j in unvisited:
            ov = overlap_matrix[last][j]
            if ov > max_ov:
                max_ov = ov
                best_next = j
        current_order.append(best_next)
        unvisited.remove(best_next)
        last = best_next
        
    # Calculate current total overlap
    def get_total_overlap(order):
        total = 0
        for i in range(len(order) - 1):
            total += overlap_matrix[order[i]][order[i+1]]
        return total
        
    current_overlap = get_total_overlap(current_order)
    best_overlap = current_overlap
    best_order = current_order[:]
    
    # Simulated Annealing for maximizing total overlap
    # Use swap moves with temperature annealing
    
    def delta_swap(order, i, j):
        if i == j: return 0
        if abs(i - j) == 1:
            if i > j: i, j = j, i
            left = order[i-1] if i > 0 else None
            mid1 = order[i]
            mid2 = order[i+1]
            right = order[i+2] if i+2 < M else None
            
            old_ov = 0
            if left is not None: old_ov += overlap_matrix[left][mid1]
            old_ov += overlap_matrix[mid1][mid2]
            if right is not None: old_ov += overlap_matrix[mid2][right]
            
            new_ov = 0
            if left is not None: new_ov += overlap_matrix[left][mid2]
            new_ov += overlap_matrix[mid2][mid1]
            if right is not None: new_ov += overlap_matrix[mid1][right]
            
            return new_ov - old_ov
        else:
            i, j = min(i, j), max(i, j)
            L1 = order[i-1] if i > 0 else None
            B = order[i]
            R1 = order[i+1]
            
            L2 = order[j-1]
            E = order[j]
            R2 = order[j+1] if j+1 < M else None
            
            old_ov = 0
            if L1 is not None: old_ov += overlap_matrix[L1][B]
            old_ov += overlap_matrix[B][R1]
            old_ov += overlap_matrix[L2][E]
            if R2 is not None: old_ov += overlap_matrix[E][R2]
            
            new_ov = 0
            if L1 is not None: new_ov += overlap_matrix[L1][E]
            new_ov += overlap_matrix[E][R1]
            new_ov += overlap_matrix[L2][B]
            if R2 is not None: new_ov += overlap_matrix[B][R2]
            
            return new_ov - old_ov

    # Simulated annealing parameters
    initial_temp = 10.0
    final_temp = 0.01
    cooling_rate = 0.99995
    temp = initial_temp
    
    # Track best solution
    sa_best_overlap = best_overlap
    sa_best_order = best_order[:]
    
    # Run simulated annealing until deadline
    while time.time() < deadline and temp > final_temp:
        # Check time periodically
        if time.time() > deadline - 5:
            break
            
        # Perform a swap
        i, j = random.sample(range(M), 2)
        d = delta_swap(current_order, i, j)
        
        # Accept if improvement, or with probability exp(-d/T) for worsening
        if d >= 0 or random.random() < math.exp(-d / temp):
            # Apply swap
            current_order[i], current_order[j] = current_order[j], current_order[i]
            current_overlap += d
            
            # Update best
            if current_overlap > sa_best_overlap:
                sa_best_overlap = current_overlap
                sa_best_order = current_order[:]
        
        # Cool down
        temp *= cooling_rate
        
        # Also do some hill climbing steps occasionally to exploit
        if random.random() < 0.1:
            # Try a few greedy improvements
            for _ in range(100):
                if time.time() > deadline:
                    break
                best_i, best_j, best_d = -1, -1, 0
                # Sample a few random swaps for hill climbing
                for _ in range(50):
                    ii, jj = random.sample(range(M), 2)
                    dd = delta_swap(current_order, ii, jj)
                    if dd > best_d:
                        best_d = dd
                        best_i, best_j = ii, jj
                if best_i >= 0:
                    current_order[best_i], current_order[best_j] = current_order[best_j], current_order[best_i]
                    current_overlap += best_d
                    if current_overlap > sa_best_overlap:
                        sa_best_overlap = current_overlap
                        sa_best_order = current_order[:]
    
    # Use the best solution found
    best_overlap = sa_best_overlap
    best_order = sa_best_order
    
    # Reconstruct string from best_order
    # Block for permutation p is p + [n] + p
    # We need to concatenate them with overlaps.
    
    best_string_list = []
    if not best_order:
        return ""
    
    # First block
    first_p = perm_tuples[best_order[0]]
    first_block = list(first_p) + [n] + list(first_p)
    best_string_list.extend(first_block)
    
    for idx in range(1, M):
        prev_idx = best_order[idx-1]
        curr_idx = best_order[idx]
        
        prev_p = perm_tuples[prev_idx]
        curr_p = perm_tuples[curr_idx]
        
        curr_block = list(curr_p) + [n] + list(curr_p)
        
        ov = overlap_matrix[prev_idx][curr_idx]
        
        # Append curr_block[ov:]
        # Note: ov is the overlap between suffix of prev_block and prefix of curr_block.
        # prev_block ends with prev_p. curr_block starts with curr_p.
        # So we append curr_block[ov:]
        best_string_list.extend(curr_block[ov:])
        
    result_str = "".join(map(str, best_string_list))
    
    # Verify validity
    # Check if all permutations of 1..n are in result_str
    seen = set()
    for i in range(len(result_str) - n + 1):
        window = tuple(map(int, result_str[i:i+n]))
        if len(set(window)) == n:
            seen.add(window)
            
    if len(seen) == len(list(permutations(range(1, n+1)))):
        return result_str
    else:
        # Fallback to standard construction if something went wrong
        word = [1]
        for m in range(2, n + 1):
            perms_list, seen_p = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen_p:
                    seen_p.add(window)
                    perms_list.append(window)
            new_word = []
            for p in perms_list:
                block = list(p) + [m] + list(p)
                overlap = 0
                for k in range(min(len(new_word), len(block)), 0, -1):
                    if new_word[len(new_word) - k:] == block[:k]:
                        overlap = k
                        break
                new_word += block[overlap:]
            word = new_word
        return "".join(map(str, word))
