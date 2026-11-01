import time
import itertools
import random
from collections import deque

def construct(n):
    start_time = time.time()
    timeout = 110.0  # Safety margin before 150s kill
    
    # For n=1, return "1"
    if n == 1:
        return "1"
    
    # For small n, we can use known optimal values or fast construction
    # n=2: "121" (len 3)
    # n=3: "123121321" (len 9) - actually optimal is 9
    # n=4: optimal is 33
    # n=5: optimal is 153
    # n=6: recursive gives 873, optimal known? LB 872. Recursive is 873.
    # n=7: recursive gives 5913. Known best is 5905. LB 5884.
    
    # The recursive construction S_n:
    # S_1 = "1"
    # S_n = for each perm P in S_{n-1} (in order of appearance):
    #         append P, n, P  (with maximal overlap)
    # This gives length 1! + 2! + ... + n!
    # For n=7: 1+2+6+24+120+720+5040 = 5913
    
    def get_recursive_superpermutation(k):
        if k == 1:
            return [1]
        
        prev = get_recursive_superpermutation(k - 1)
        
        # Extract unique permutations of length k-1 from prev in order
        perms = []
        seen = set()
        for i in range(len(prev) - (k - 1) + 1):
            window = tuple(prev[i:i + k - 1])
            if len(set(window)) == k - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
                
        new_word = []
        for p in perms:
            block = list(p) + [k] + list(p)
            # Overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
            
        return new_word

    # Get the recursive solution
    recursive_word = get_recursive_superpermutation(n)
    best_word = recursive_word
    best_len = len(best_word)
    
    # For n <= 5, recursive is optimal, return it
    if n <= 5:
        return "".join(map(str, best_word))
    
    # For n=6, recursive gives 873. Let's try to improve.
    # For n=7, recursive gives 5913. Known best is 5905.
    
    # Strategy: The recursive construction produces a specific ordering of permutations.
    # We can try to find a better ordering using local search on the permutation sequence.
    # The cost of a sequence of permutations p1, p2, ..., p_{n!} is:
    #   n + sum_{i=1}^{n!-1} (n - overlap(p_i, p_{i+1}))
    # where overlap(a, b) = max{k < n : a[n-k:] == b[:k]}
    
    # For n=7, n! = 5040. We have 5040 permutations.
    # The recursive ordering gives total length 5913.
    # We want to minimize the sum of (n - overlap) between consecutive permutations.
    
    # Let's extract the permutation order from the recursive word
    def extract_perm_order(word, n):
        perms = []
        seen = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n and window not in seen:
                seen.add(window)
                perms.append(window)
        return perms
    
    if n <= 7 and time.time() - start_time < timeout - 5:
        perm_order = extract_perm_order(best_word, n)
        num_perms = len(perm_order)
        
        # Verify we have all permutations
        all_perms = set(itertools.permutations(range(1, n+1)))
        if set(perm_order) != all_perms:
            # Fallback to recursive
            pass
        else:
            # Precompute overlap matrix for speed
            # overlap_matrix[i][j] = overlap(perm_order[i], perm_order[j])
            # This is expensive to compute fully (5040^2 = 25M entries), 
            # but we can compute it lazily or use a smarter approach.
            
            # Instead, let's use a dictionary cache for overlaps
            overlap_cache = {}
            
            def calc_overlap(a, b):
                # max k < n such that a[n-k:] == b[:k]
                key = (a, b)
                if key in overlap_cache:
                    return overlap_cache[key]
                for k in range(n-1, 0, -1):
                    if a[n-k:] == b[:k]:
                        overlap_cache[key] = k
                        return k
                overlap_cache[key] = 0
                return 0
            
            def total_length(order):
                total = n
                for i in range(len(order)-1):
                    total += n - calc_overlap(order[i], order[i+1])
                return total
            
            current_order = list(perm_order)
            current_len = total_length(current_order)
            
            # Subgroup orbit decomposition strategy:
            # The recursive construction has a hierarchical structure.
            # Permutations can be grouped by their "level" in the recursion.
            # At each level, we can try to reorder blocks to minimize transitions.
            
            # For n=7, the recursive structure means that consecutive permutations
            # in the recursive order share long suffixes. We can identify "blocks"
            # where permutations share a common prefix of length n-1 or n-2, etc.
            
            # A better approach: identify the natural block structure and try
            # to reorder within and between blocks.
            
            # Let's identify blocks by the longest common suffix with the previous perm
            # In the recursive construction, many consecutive pairs have overlap n-1
            # (they differ only in the first position).
            
            # 2-opt move: reverse a segment of the permutation sequence
            def apply_2opt(i, j):
                """Reverse segment from i to j (inclusive) in current_order"""
                nonlocal current_len
                if i >= j:
                    return False
                
                # Compute old cost for edges affected
                old_cost = 0
                new_cost = 0
                
                # Edge before segment
                if i > 0:
                    old_cost += n - calc_overlap(current_order[i-1], current_order[i])
                    new_cost += n - calc_overlap(current_order[i-1], current_order[j])
                
                # Edges within segment (they get reversed)
                for k in range(i, j):
                    old_cost += n - calc_overlap(current_order[k], current_order[k+1])
                    new_cost += n - calc_overlap(current_order[k+1], current_order[k])
                
                # Edge after segment
                if j + 1 < num_perms:
                    old_cost += n - calc_overlap(current_order[j], current_order[j+1])
                    new_cost += n - calc_overlap(current_order[i], current_order[j+1])
                
                delta = new_cost - old_cost
                if delta < 0:
                    # Apply the reversal
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_len += delta
                    return True
                return False
            
            # Apply adjacent swap (1-opt)
            def apply_swap(i):
                nonlocal current_len
                old_cost = 0
                new_cost = 0
                
                if i > 0:
                    old_cost += n - calc_overlap(current_order[i-1], current_order[i])
                    new_cost += n - calc_overlap(current_order[i-1], current_order[i+1])
                
                old_cost += n - calc_overlap(current_order[i], current_order[i+1])
                new_cost += n - calc_overlap(current_order[i+1], current_order[i])
                
                if i+2 < num_perms:
                    old_cost += n - calc_overlap(current_order[i+1], current_order[i+2])
                    new_cost += n - calc_overlap(current_order[i], current_order[i+2])
                
                delta = new_cost - old_cost
                if delta < 0:
                    current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                    current_len += delta
                    return True
                return False
            
            # Hill climbing with 1-opt and 2-opt moves
            no_improvement_passes = 0
            max_no_improvement_passes = 30
            max_2opt_length = min(50, num_perms // 10)  # Limit segment length for 2-opt
            
            while time.time() - start_time < timeout - 10:
                if no_improvement_passes >= max_no_improvement_passes:
                    break
                
                improved_in_pass = False
                
                # Phase 1: Adjacent swaps (1-opt)
                for i in range(num_perms - 1):
                    if time.time() - start_time > timeout - 10:
                        break
                    if apply_swap(i):
                        improved_in_pass = True
                
                # Phase 2: 2-opt moves with small segments
                if time.time() - start_time < timeout - 10:
                    for i in range(0, num_perms - max_2opt_length, max(1, (num_perms - max_2opt_length) // 500)):
                        if time.time() - start_time > timeout - 10:
                            break
                        for j in range(i + 1, min(i + max_2opt_length, num_perms)):
                            if apply_2opt(i, j):
                                improved_in_pass = True
                                break  # Restart 2-opt after a successful move
                
                if not improved_in_pass:
                    no_improvement_passes += 1
                    # Structured perturbation based on subgroup structure
                    if no_improvement_passes < max_no_improvement_passes:
                        # Instead of random swap, try reversing a random small segment
                        seg_len = random.randint(2, min(10, num_perms - 1))
                        idx = random.randint(0, num_perms - seg_len)
                        current_order[idx:idx+seg_len] = current_order[idx:idx+seg_len][::-1]
                        current_len = total_length(current_order)
                else:
                    no_improvement_passes = 0
            
            # Reconstruct word from best order
            if current_len < best_len:
                # Build the word
                new_word = list(current_order[0])
                for i in range(1, num_perms):
                    a = current_order[i-1]
                    b = current_order[i]
                    ov = calc_overlap(a, b)
                    new_word.extend(b[ov:])
                
                best_word = new_word
                best_len = len(best_word)
    
    # Check time
    if time.time() - start_time > timeout:
        return "".join(map(str, best_word))
    
    return "".join(map(str, best_word))
