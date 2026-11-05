import time
import itertools
import random

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0 # Leave 5s safety margin
    
    # Step 1: Generate the initial permutation sequence using the classic recursive method
    # This matches the seed program's logic to get the base case.
    def generate_seed_perm_order(n):
        word = [1]
        perm_order = [(1,)]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            # Reconstruct the full word to extract the new permutation order
            # The classic algorithm builds the word by inserting m.
            # The order of permutations in the final word for n symbols is derived
            # from the order of (n-1)-symbol windows in the previous word.
            # Let's just build the word and then extract the unique n-windows in order of appearance.
            
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find overlap with current new
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or (o < len(new) and o < len(block) and new[len(new) - o:] == block[:o]):
                        overlap = o
                        break
                new += block[overlap:]
            
            word = new
        
        # Extract the sequence of n-permutations from the final word
        seq = []
        seen = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i + n])
            if window not in seen:
                seen.add(window)
                seq.append(window)
        return seq

    # For n=7, the sequence length is 5040.
    if n <= 7:
        # We only care about n=7 for the main task, but handle general n for completeness
        # The classic construction is optimal for n<=4, but for n>=5 it's not the absolute shortest.
        
        if n < 5:
            # For small n, the classic construction is optimal
            return generate_seed_word(n)

        # For n >= 5, use local search
        perm_seq = generate_seed_perm_order(n)
        
        # Calculate initial length
        def calc_length(seq):
            if not seq:
                return 0
            length = n
            for i in range(len(seq) - 1):
                # Calculate overlap between seq[i] and seq[i+1]
                p1 = seq[i]
                p2 = seq[i+1]
                max_k = n - 1
                # Check suffix of p1 vs prefix of p2
                # k=0 is always valid (no overlap, cost n)
                # We want max k such that p1[n-k:] == p2[:k]
                # Since they are permutations, they can't be equal.
                # Max possible overlap is n-1.
                found_k = 0
                for k in range(n - 1, 0, -1):
                    if p1[n-k:] == p2[:k]:
                        found_k = k
                        break
                length += (n - found_k)
            return length

        current_len = calc_length(perm_seq)
        best_len = current_len
        best_seq = perm_seq
        
        # Local Search: Swap adjacent permutations
        # We can also try 2-opt moves (reversing a segment), but swaps are faster.
        improved = True
        random.seed(42)
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Shuffle the order of indices to avoid local minima bias, or just iterate
            # Iterating sequentially is deterministic. Let's add a bit of randomness in starting point or order.
            indices = list(range(len(perm_seq) - 1))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 1.0:
                    break
                
                # Try swapping perm_seq[i] and perm_seq[i+1]
                # The neighbors are perm_seq[i-1] and perm_seq[i+2]
                
                p_left = perm_seq[i-1] if i > 0 else None
                p_i = perm_seq[i]
                p_i1 = perm_seq[i+1]
                p_right = perm_seq[i+2] if i+2 < len(perm_seq) else None
                
                # Calculate current cost of transitions involving p_i and p_i1
                # Transitions: (p_left, p_i), (p_i, p_i1), (p_i1, p_right)
                
                def overlap(a, b):
                    if a is None or b is None:
                        return 0
                    for k in range(n - 1, 0, -1):
                        if a[n-k:] == b[:k]:
                            return k
                    return 0
                
                cost_left_i = n - overlap(p_left, p_i) if p_left else 0
                cost_i_i1 = n - overlap(p_i, p_i1)
                cost_i1_right = n - overlap(p_i1, p_right) if p_right else 0
                current_local_cost = cost_left_i + cost_i_i1 + cost_i1_right
                
                # New order: p_left, p_i1, p_i, p_right
                new_cost_left_i1 = n - overlap(p_left, p_i1) if p_left else 0
                new_cost_i1_i = n - overlap(p_i1, p_i)
                new_cost_i_right = n - overlap(p_i, p_right) if p_right else 0
                new_local_cost = new_cost_left_i1 + new_cost_i1_i + new_cost_i_right
                
                if new_local_cost < current_local_cost:
                    # Perform swap
                    perm_seq[i], perm_seq[i+1] = perm_seq[i+1], perm_seq[i]
                    current_len -= (current_local_cost - new_local_cost)
                    improved = True
                    
                    # Update best if better
                    if current_len < best_len:
                        best_len = current_len
                        best_seq = perm_seq[:]
                        
        # Reconstruct word from best_seq
        # The word starts with best_seq[0]
        # Then append the non-overlapping part of each subsequent permutation
        word = list(best_seq[0])
        for i in range(1, len(best_seq)):
            p_prev = best_seq[i-1]
            p_curr = best_seq[i]
            # Find max overlap
            k = 0
            for kk in range(n - 1, -1, -1):
                if p_prev[n-kk:] == p_curr[:kk]:
                    k = kk
                    break
            word.extend(p_curr[k:])
        
        return "".join(map(str, word))
    
    else:
        # Fallback for larger n, though task is for n=7
        return generate_seed_word(n)

def generate_seed_word(n):
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

# Helper to run and validate locally if needed, but not required for submission
# The function construct is the entry point.
