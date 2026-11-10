import time
import random
from itertools import permutations
import numpy as np
import sys

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
    
    # 3. Compute overlap matrix efficiently
    # O[i][j] = max k such that perm_tuples[i][-k:] == perm_tuples[j][:k]
    # We compute this by checking each possible overlap length from longest to shortest
    # and using a dictionary to group permutations by their prefix/suffix of that length.
    
    overlap_matrix = np.zeros((M, M), dtype=np.int8)
    
    # For each k from n-1 down to 1:
    # If suffix_k(i) == prefix_k(j), then overlap is at least k.
    # We want the MAX overlap, so we iterate k from n-1 down to 1.
    
    # Use a dictionary to map each prefix of length k to the list of indices that have that prefix
    # Similarly for suffixes.
    
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
        
        # Iterate over all suffixes that are also prefixes
        for suff, i_list in suffix_map.items():
            if suff in prefix_map:
                j_list = prefix_map[suff]
                for i in i_list:
                    for j in j_list:
                        overlap_matrix[i][j] = k
    
    # 4. Local Search to maximize total overlap
    
    # Greedy construction of initial order
    # Start with a random node. Repeatedly pick the next unvisited node with max overlap.
    start_node = random.randint(0, M - 1)
    unvisited = set(range(M))
    unvisited.remove(start_node)
    current_order = [start_node]
    last = start_node
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
    
    # Local Search: Swap and 2-opt moves
    
    def delta_swap(order, i, j):
        if i == j: return 0
        if abs(i - j) == 1:
            # Adjacent swap
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
            # Non-adjacent swap
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
    
    def delta_2opt(order, i, j):
        # Reverse segment from i to j (inclusive)
        # i < j
        # Old: ... L, A, ..., B, R ...
        # New: ... L, B, ..., A, R ...
        # where A = order[i], B = order[j]
        # L = order[i-1] if i > 0 else None
        # R = order[j+1] if j+1 < M else None
        
        if i >= j: return 0
        i, j = min(i, j), max(i, j)
        
        L = order[i-1] if i > 0 else None
        A = order[i]
        B = order[j]
        R = order[j+1] if j+1 < M else None
        
        # Old overlaps involving the boundary of the segment
        old_ov = 0
        if L is not None: old_ov += overlap_matrix[L][A]
        if R is not None: old_ov += overlap_matrix[B][R]
        
        # Also need to account for internal overlaps that change
        # When we reverse the segment, the internal overlaps change.
        # Old internal: sum of overlap_matrix[order[k]][order[k+1]] for k from i to j-1
        # New internal: sum of overlap_matrix[order[k]][order[k-1]] for k from i to j-1 (reversed)
        # So the delta is: (new_boundary + new_internal) - (old_boundary + old_internal)
        
        old_internal = 0
        for k in range(i, j):
            old_internal += overlap_matrix[order[k]][order[k+1]]
        
        new_internal = 0
        for k in range(i, j):
            new_internal += overlap_matrix[order[k+1]][order[k]]
        
        new_ov = 0
        if L is not None: new_ov += overlap_matrix[L][B]
        if R is not None: new_ov += overlap_matrix[A][R]
        
        return (new_ov + new_internal) - (old_ov + old_internal)
    
    improved = True
    no_improve_count = 0
    while time.time() < deadline and improved:
        improved = False
        no_improve_count = 0
        # Mix of swap and 2-opt moves
        num_moves = 20000
        for _ in range(num_moves):
            if time.time() > deadline: break
            move_type = random.random()
            if move_type < 0.5:
                # Swap move
                i, j = random.sample(range(M), 2)
                d = delta_swap(current_order, i, j)
                if d > 0:
                    current_order[i], current_order[j] = current_order[j], current_order[i]
                    current_overlap += d
                    improved = True
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_order = current_order[:]
            else:
                # 2-opt move
                i, j = sorted(random.sample(range(M), 2))
                d = delta_2opt(current_order, i, j)
                if d > 0:
                    # Reverse the segment
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_overlap += d
                    improved = True
                    if current_overlap > best_overlap:
                        best_overlap = current_overlap
                        best_order = current_order[:]
        
        if not improved:
            # Try a few random restarts of local search from perturbed solutions
            for _ in range(5):
                if time.time() > deadline: break
                # Perturb the current best solution
                perturbed = best_order[:]
                for _ in range(10):
                    i, j = random.sample(range(M), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                
                perturbed_overlap = get_total_overlap(perturbed)
                if perturbed_overlap > best_overlap:
                    best_overlap = perturbed_overlap
                    best_order = perturbed[:]
                
                # Run a few iterations of local search on the perturbed solution
                temp_order = perturbed[:]
                temp_overlap = perturbed_overlap
                for _ in range(5000):
                    if time.time() > deadline: break
                    i, j = random.sample(range(M), 2)
                    d = delta_swap(temp_order, i, j)
                    if d > 0:
                        temp_order[i], temp_order[j] = temp_order[j], temp_order[i]
                        temp_overlap += d
                        if temp_overlap > best_overlap:
                            best_overlap = temp_overlap
                            best_order = temp_order[:]
    
    # Reconstruct string from best_order
    # Block for permutation p is p + [n] + p
    # We need to concatenate them with overlaps.
    
    if not best_order:
        return ""
    
    # Build the word by concatenating blocks with overlaps
    result_list = []
    
    # First block
    first_p = perm_tuples[best_order[0]]
    first_block = list(first_p) + [n] + list(first_p)
    result_list.extend(first_block)
    
    for idx in range(1, M):
        prev_idx = best_order[idx-1]
        curr_idx = best_order[idx]
        
        curr_p = perm_tuples[curr_idx]
        curr_block = list(curr_p) + [n] + list(curr_p)
        
        ov = overlap_matrix[prev_idx][curr_idx]
        
        # Append curr_block[ov:]
        result_list.extend(curr_block[ov:])
        
    result_str = "".join(map(str, result_list))
    
    # Verify validity
    seen = set()
    for i in range(len(result_str) - n + 1):
        window = tuple(map(int, result_str[i:i+n]))
        if len(set(window)) == n:
            seen.add(window)
    
    target_count = 1
    for i in range(2, n+1):
        target_count *= i
    
    if len(seen) == target_count:
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
