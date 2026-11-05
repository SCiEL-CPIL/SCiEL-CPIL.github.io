import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def is_valid(seq):
        if not seq:
            return True
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_sums(seq):
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                sums.add(s)
        return sums

    # Greedy initialization
    def greedy(start_val, end_val, step):
        values = []
        for x in range(start_val, end_val, step):
            new_vals = values + [x]
            new_vals.sort()
            if is_valid(new_vals):
                values = new_vals
        return values

    # Greedy 1: Smallest to largest
    seq1 = greedy(1, 257, 1)
    
    # Greedy 2: Largest to smallest
    seq2 = greedy(256, 0, -1)
    
    best_seq = seq1 if len(seq1) >= len(seq2) else seq2

    # Backtracking DFS with memoization on (last_value, suffix_sum, frozenset_of_sums)
    # We build the sequence in order. The state is:
    # - last_val: the last element added (to ensure strictly increasing)
    # - suffix_sum: sum of elements from some position to the end (for checking new consecutive block sums)
    # - sums_set: frozenset of all consecutive block sums so far
    #
    # When adding a new element x > last_val at the end:
    # New consecutive blocks ending at x:
    #   - x itself
    #   - x + (suffix of previous sequence)
    # Specifically, if previous sequence is [a1, a2, ..., ak], suffix sums are:
    #   ak, ak + a{k-1}, ..., ak + ... + a1
    # New sums: x, x + ak, x + ak + a{k-1}, ..., x + ak + ... + a1
    #
    # We need all these new sums to not be in sums_set.
    #
    # For efficiency, we track:
    # - last_val: int
    # - suffix_sums: tuple of suffix sums (sorted or in order)
    # - sums_frozen: frozenset of all sums
    #
    # The memoization key could be (last_val, suffix_sums_tuple, sums_frozen) but that's huge.
    # Instead, let's use a simpler approach: DFS that builds the sequence, with pruning.
    
    def dfs_backtrack(current_seq, sums_set, suffix_sums, max_depth, time_limit):
        """
        current_seq: list of elements in order
        sums_set: set of all consecutive block sums
        suffix_sums: list of suffix sums [s_k, s_{k-1}+s_k, ..., total_sum] where s_i are elements
                     Actually: suffix_sums[i] = sum of current_seq[i:]
        max_depth: maximum length to try
        time_limit: deadline timestamp
        """
        if time.time() > time_limit:
            return current_seq
        
        L = len(current_seq)
        if L >= max_depth:
            return current_seq
        
        # Try adding each possible next value
        last_val = current_seq[-1] if current_seq else 0
        # Next value must be > last_val and <= 256
        # To be efficient, try values in a smart order
        
        best_result = list(current_seq)
        
        for x in range(last_val + 1, 257):
            if time.time() > time_limit:
                break
            
            # Compute new suffix sums
            # New suffix sums: [x, x + suffix_sums[0], x + suffix_sums[1], ..., x + suffix_sums[-1]]
            new_suffix_sums = [x]
            for ss in suffix_sums:
                new_suffix_sums.append(x + ss)
            
            # Check if any new sum is already in sums_set
            valid = True
            for ns in new_suffix_sums:
                if ns in sums_set:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Create new sums set
            new_sums_set = set(sums_set)
            for ns in new_suffix_sums:
                new_sums_set.add(ns)
            
            # Recurse
            new_seq = current_seq + [x]
            result = dfs_backtrack(new_seq, new_sums_set, new_suffix_sums, max_depth, time_limit)
            if len(result) > len(best_result):
                best_result = result
            
            # If we've hit max_depth, no point continuing
            if len(best_result) >= max_depth:
                break
        
        return best_result

    # Try DFS from small starting values
    best_dfs_seq = []
    for start in range(1, 20):
        if time.time() > deadline:
            break
        initial_suffix = [start]
        initial_sums = {start}
        result = dfs_backtrack([start], initial_sums, initial_suffix, 20, deadline)
        if len(result) > len(best_dfs_seq):
            best_dfs_seq = result
    
    if len(best_dfs_seq) > len(best_seq):
        best_seq = best_dfs_seq

    # Local Search
    current_seq = list(best_seq)
    best_len = len(current_seq)
    best_seq_final = list(current_seq)
    
    all_candidates = set(range(1, 257))
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.7: # Try to Add
            current_set = set(current_seq)
            available = [x for x in all_candidates if x not in current_set]
            if available:
                x = random.choice(available)
                new_seq = current_seq + [x]
                new_seq.sort()
                if is_valid(new_seq):
                    current_seq = new_seq
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq_final = list(current_seq)
                        
        elif r < 0.9: # Try to Swap
            if current_seq:
                current_set = set(current_seq)
                available = [x for x in all_candidates if x not in current_set]
                if available:
                    x_rem = random.choice(current_seq)
                    x_add = random.choice(available)
                    new_seq = [x for x in current_seq if x != x_rem]
                    new_seq.append(x_add)
                    new_seq.sort()
                    if is_valid(new_seq):
                        current_seq = new_seq
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq_final = list(current_seq)
                            
        else: # Try to Remove (escape local minima)
            if current_seq and len(current_seq) > 1:
                x_rem = random.choice(current_seq)
                current_seq.remove(x_rem)
                
    return sorted(best_seq_final)
