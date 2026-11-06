import time

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for the 90s limit
    
    # 1. Get a baseline using the greedy algorithm
    greedy_seq = greedy_construct()
    best_seq = greedy_seq[:]
    best_len = len(best_seq)
    
    # 2. Use DFS to try to find a longer sequence
    # We start DFS from an empty sequence.
    # The DFS will naturally explore the greedy path first (smallest x).
    # If the greedy path is a dead end or suboptimal, DFS will backtrack and find better paths.
    
    def try_add(seq, sums, x):
        """
        Check if x can be appended to seq.
        Returns (True, list_of_new_sums) if valid, (False, None) otherwise.
        """
        new_sums = []
        s = x
        new_sums.append(s)
        if s in sums:
            return False, None
        
        for a in reversed(seq):
            s += a
            new_sums.append(s)
            if s in sums:
                return False, None
        
        return True, new_sums

    def dfs(seq, sums):
        nonlocal best_seq, best_len
        
        # Time check
        if time.time() - start_time > time_limit:
            return
        
        # Update best if we found a longer sequence
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Pruning: Can we possibly beat the current best?
        if seq:
            last = seq[-1]
        else:
            last = 0
            
        # Maximum number of elements we can add is (256 - last)
        # Because the sequence must be strictly increasing and <= 256
        max_possible_len = len(seq) + (256 - last)
        if max_possible_len <= best_len:
            return

        # Try appending each possible next element
        # We iterate in increasing order to prioritize packing more elements
        for x in range(last + 1, 257):
            valid, new_sums = try_add(seq, sums, x)
            if valid:
                # Create new state
                # Using union creates a new set, which is O(N) where N is size of sums.
                # This is acceptable for small N.
                new_sums_set = sums.union(new_sums)
                dfs(seq + [x], new_sums_set)
                
                # Check time again after recursion
                if time.time() - start_time > time_limit:
                    return

    dfs([], set())
    
    return best_seq

def greedy_construct():
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values
