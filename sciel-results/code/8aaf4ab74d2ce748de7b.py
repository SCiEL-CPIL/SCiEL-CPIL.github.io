import time
import random
import sys

def construct():
    start_time = time.time()
    TIMEOUT = 58.0  # Safety margin before 90s kill (evaluator kills at 90s)
    
    def check_time():
        return time.time() - start_time > TIMEOUT

    def get_sums_and_tails(seq):
        """Compute the set of all consecutive block sums and the list of suffix sums."""
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        
        tails = []
        t = 0
        for x in reversed(seq):
            t += x
            tails.append(t)
        
        return sums, tails

    def try_append(seq, sums, tails, x):
        """
        Check if x can be appended to seq.
        Returns (new_sums, new_tails) if valid, else (None, None).
        """
        # New suffixes are x, x + last, x + last + prev, ...
        # tails is [last, last+prev, ...]
        # So new_tails = [x + t for t in [0] + tails]
        new_tails = [x + t for t in [0] + tails]
        
        # Check for collisions
        # Optimization: just check intersection
        if any(s in sums for s in new_tails):
            return None, None
            
        # If valid, compute new sums set
        # sums | set(new_tails)
        new_sums = sums.union(new_tails)
        
        return new_sums, new_tails

    def greedy_heuristic(direction='forward'):
        """
        Greedy construction.
        direction: 'forward' (1->256) or 'reverse' (256->1)
        """
        seq = []
        sums = set()
        tails = []
        
        if direction == 'forward':
            candidates = range(1, 257)
        else:
            candidates = range(256, 0, -1)
            
        for x in candidates:
            new_sums, new_tails = try_append(seq, sums, tails, x)
            if new_sums is not None:
                seq.append(x)
                sums = new_sums
                tails = new_tails
                
        # Ensure strictly increasing
        seq.sort()
        return seq

    def randomized_greedy(num_restarts=50):
        """
        Randomized greedy with restarts.
        Picks a random starting point, then greedily adds valid elements.
        """
        best_seq = []
        
        for _ in range(num_restarts):
            if check_time():
                break
                
            seq = []
            sums = set()
            tails = []
            
            # Random start
            start_val = random.randint(1, 250)
            
            # Try to build from start_val upwards
            # We don't strictly have to start at start_val, but it biases the search
            # Let's just pick a random first element
            x0 = random.randint(1, 256)
            seq = [x0]
            sums = {x0}
            tails = [x0]
            
            # Now greedily add from x0+1 to 256
            for x in range(x0 + 1, 257):
                new_sums, new_tails = try_append(seq, sums, tails, x)
                if new_sums is not None:
                    # Accept with some probability? Or just always accept for speed?
                    # Always accept is faster and often good.
                    seq.append(x)
                    sums = new_sums
                    tails = new_tails
                    
            if len(seq) > len(best_seq):
                best_seq = seq
                
        return best_seq

    def dfs_extend(seq, sums, tails, depth, max_depth=None):
        """
        DFS to extend a given sequence.
        Returns the longest sequence found in this subtree.
        """
        if check_time():
            return seq
            
        if max_depth is not None and depth >= max_depth:
            return seq
            
        best_seq = list(seq)
        last_val = seq[-1] if seq else 0
        
        # Candidates: last_val + 1 to 256
        # We can try to limit the search space if we just want to extend slightly
        for x in range(last_val + 1, 257):
            if check_time():
                break
                
            new_sums, new_tails = try_append(seq, sums, tails, x)
            if new_sums is None:
                continue
                
            seq.append(x)
            
            # Recurse
            extended = dfs_extend(seq, new_sums, new_tails, depth + 1, max_depth)
            
            if len(extended) > len(best_seq):
                best_seq = extended
                
            seq.pop()
            
        return best_seq

    # 1. Initial Heuristics
    seq_fwd = greedy_heuristic('forward')
    seq_rev = greedy_heuristic('reverse')
    
    best_seq = seq_fwd if len(seq_fwd) >= len(seq_rev) else seq_rev
    best_len = len(best_seq)
    
    # 2. Randomized Greedy
    if not check_time():
        rand_seq = randomized_greedy(100)
        if len(rand_seq) > best_len:
            best_seq = rand_seq
            best_len = len(best_seq)

    # 3. Local Search / DFS Extension from Best
    # Compute sums/tails for best_seq
    if best_seq:
        best_sums, best_tails = get_sums_and_tails(best_seq)
        # Try to extend the current best
        # Limit depth to avoid exploding if best is short, but allow full if best is long
        # Actually, just let it run until timeout or dead end.
        extended_seq = dfs_extend(list(best_seq), best_sums, best_tails, len(best_seq))
        if len(extended_seq) > best_len:
            best_seq = extended_seq
            best_len = len(best_seq)

    # 4. If time remains, try a few more random restarts and extend the best ones
    if not check_time():
        for _ in range(20):
            if check_time():
                break
                
            # Quick random greedy
            seq = []
            sums = set()
            tails = []
            x0 = random.randint(1, 256)
            seq = [x0]
            sums = {x0}
            tails = [x0]
            
            for x in range(x0 + 1, 257):
                new_sums, new_tails = try_append(seq, sums, tails, x)
                if new_sums is not None:
                    seq.append(x)
                    sums = new_sums
                    tails = new_tails
            
            if len(seq) > best_len - 2: # Only extend if it's close to best
                # Extend it
                ext_seq = dfs_extend(seq, sums, tails, len(seq))
                if len(ext_seq) > best_len:
                    best_seq = ext_seq
                    best_len = len(best_seq)

    return best_seq
