import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s kill
    
    best_seq = []
    best_len = 0
    nodes_visited = 0
    max_nodes_per_search = 50000
    
    def is_valid_candidate(x, prefix_sums, forbidden_sums):
        """
        Checks if adding x to the current sequence (ending at prefix_sums[-1])
        results in a valid sequence.
        """
        new_prefix = prefix_sums[-1] + x
        for p in prefix_sums:
            d = new_prefix - p
            if d in forbidden_sums:
                return False
        return True

    def backtrack(seq, prefix_sums, forbidden_sums, last_val):
        nonlocal best_seq, best_len, nodes_visited
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        if len(seq) == 256:
            return True
            
        if nodes_visited > max_nodes_per_search:
            return False
            
        # Generate valid candidates
        candidates = []
        for x in range(last_val + 1, 257):
            if is_valid_candidate(x, prefix_sums, forbidden_sums):
                candidates.append(x)
                
        if not candidates:
            return False
            
        # Randomize order to explore different branches
        random.shuffle(candidates)
        
        for x in candidates:
            nodes_visited += 1
            if nodes_visited > max_nodes_per_search:
                return False
                
            # Add x
            seq.append(x)
            new_prefix = prefix_sums[-1] + x
            new_sums = []
            for p in prefix_sums:
                s = new_prefix - p
                new_sums.append(s)
                forbidden_sums.add(s)
            prefix_sums.append(new_prefix)
            
            if backtrack(seq, prefix_sums, forbidden_sums, x):
                return True
                
            # Backtrack
            seq.pop()
            prefix_sums.pop()
            for s in new_sums:
                forbidden_sums.remove(s)
                
        return False

    # Run multiple backtracking searches with different random seeds
    for seed in range(1000):
        if time.time() - start_time > time_limit - 5:
            break
            
        random.seed(seed)
        nodes_visited = 0
        backtrack([], [0], set(), 0)
        
    return best_seq
