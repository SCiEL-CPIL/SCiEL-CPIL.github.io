import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Step 1: Generate W_{n-1} using standard construction
    # This gives us the standard order of permutations of 1..n-1
    # and the word W_{n-1} itself (though we might not need the full word, just the perms)
    
    # Standard construction of W_{n-1}
    word = [1]
    perms_list = [[1]] # List of permutations of 1..k for current k
    
    for m in range(2, n): # Build up to n-1
        new_perms = []
        seen = set()
        # Extract permutations of 1..m-1 from word
        # The standard construction ensures that the order of appearance is specific
        # We just need the list of permutations in the order they appear in the standard W_{m-1}
        # Actually, the standard construction defines the order.
        # We can just generate the permutations in the standard order.
        
        # The standard order for W_m is derived from W_{m-1}.
        # For each permutation p in W_{m-1} (in order of appearance), we form p, m, p.
        # The order of permutations in W_m is the order of p in W_{m-1}.
        
        # So, perms_list for m-1 is the order.
        # For m, we iterate over perms_list.
        
        # We don't actually need to build the full word W_{n-1} if we just want the order.
        # The order of permutations of 1..n-1 in the standard construction is:
        # Start with [1].
        # For m=2, perms are [1, 2] -> 12, 21? 
        # Let's trace:
        # W1 = [1]. Perms of 1: [1].
        # m=2:
        # p=1. Block: 1, 2, 1.
        # W2 = 121.
        # Perms of 1,2 in W2: 12 (at 0), 21 (at 1).
        # So order is [12, 21].
        
        # m=3:
        # p=12. Block: 12, 3, 12.
        # p=21. Block: 21, 3, 21.
        # Concatenate with overlap.
        # W3 = 123121321.
        # Perms of 1,2,3 in W3:
        # 123, 231, 312, 121? No, 121 is not a perm of 1,2,3.
        # Windows of size 3 in 123121321:
        # 123, 231, 312, 121 (skip), 213, 132, 321.
        # Order: 123, 231, 312, 213, 132, 321.
        
        # So the order of permutations of size m is determined by the windows in W_{m-1}.
        
        # We can generate this order recursively.
        
    # Let's implement the recursive generation of the permutation order.
    
    def get_perm_order(k):
        if k == 1:
            return [[1]]
        
        prev_order = get_perm_order(k - 1)
        # We need to simulate the construction of W_{k-1} to find the order of permutations of size k-1?
        # No, the order of permutations of size k-1 IS prev_order.
        # The order of permutations of size k is the order of windows of size k in W_k.
        # But W_k is constructed from blocks of prev_order.
        
        # This is getting complicated to simulate without the word.
        # Let's just build the word for n-1. n is at most 7, so n-1 is 6.
        # W_6 has length 873. Building it is fast.
        
    # Build W_{n-1}
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            if not new:
                new = block
            else:
                # Find max overlap
                max_ov = 0
                max_l = min(len(new), len(block))
                for o in range(max_l, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
        word = new
        
    # Now 'word' is W_{n-1}.
    # 'perms' from the last iteration is the list of permutations of 1..n-1 in the standard order.
    # Let's re-extract them to be sure.
    std_perms = []
    seen = set()
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            std_perms.append(window)
            
    # We have std_perms, a list of 720 tuples.
    # We want to reorder them to maximize overlap when inserting n.
    
    def calc_overlap(p1, p2, n_val):
        # Block 1: p1, n, p1
        # Block 2: p2, n, p2
        # Suffix of B1, Prefix of B2
        b1 = list(p1) + [n_val] + list(p1)
        b2 = list(p2) + [n_val] + list(p2)
        
        max_ov = 0
        max_l = min(len(b1), len(b2))
        for o in range(max_l, -1, -1):
            if b1[len(b1) - o:] == b2[:o]:
                return o
        return 0

    # Precompute overlaps? 720x720 matrix.
    # 720*720 = 518400 entries.
    # Each overlap calc is O(n). n=7.
    # 518400 * 7 operations ~ 3.6 million. Very fast.
    
    overlap_matrix = [[0]*len(std_perms) for _ in range(len(std_perms))]
    for i in range(len(std_perms)):
        for j in range(len(std_perms)):
            overlap_matrix[i][j] = calc_overlap(std_perms[i], std_perms[j], n)
            
    current_order = list(range(len(std_perms)))
    
    def total_overlap(order):
        total = 0
        for k in range(len(order) - 1):
            total += overlap_matrix[order[k]][order[k+1]]
        return total
        
    best_order = current_order[:]
    best_val = total_overlap(best_order)
    
    # Local Search
    # Swap two elements
    num_iters = 0
    while time.time() < deadline:
        num_iters += 1
        i = random.randint(0, len(current_order) - 2)
        j = random.randint(i + 1, len(current_order) - 1)
        
        # Swap i and j
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_val = total_overlap(current_order)
        
        if new_val > best_val:
            best_val = new_val
            best_order = current_order[:]
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
        # Occasionally try 2-opt (reverse segment)
        if num_iters % 10 == 0:
             i = random.randint(0, len(current_order) - 2)
             j = random.randint(i + 1, len(current_order) - 1)
             current_order[i:j+1] = reversed(current_order[i:j+1])
             new_val = total_overlap(current_order)
             if new_val > best_val:
                 best_val = new_val
                 best_order = current_order[:]
             else:
                 current_order[i:j+1] = reversed(current_order[i:j+1])

    # Construct final word using best_order
    final_word = []
    for idx in best_order:
        p = std_perms[idx]
        block = list(p) + [n] + list(p)
        if not final_word:
            final_word = block
        else:
            max_ov = 0
            max_l = min(len(final_word), len(block))
            for o in range(max_l, -1, -1):
                if final_word[len(final_word) - o:] == block[:o]:
                    max_ov = o
                    break
            final_word += block[max_ov:]
            
    return "".join(map(str, final_word))
