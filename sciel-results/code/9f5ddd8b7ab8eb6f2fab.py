import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit

    def is_valid_incremental(seq, sums_set, suffix_sums, x):
        """
        Check if adding x to seq is valid.
        Returns (is_valid, new_sums_set, new_suffix_sums)
        """
        # New suffix sums when appending x:
        # [x, x + suffix_sums[0], x + suffix_sums[1], ...]
        # Note: suffix_sums[0] is the last element of seq, suffix_sums[-1] is total sum.
        
        # The new sums are exactly these new suffix sums.
        # We need to check if any new sum is already in sums_set.
        
        # Optimization: If x itself is in sums_set, invalid.
        if x in sums_set:
            return False, None, None
            
        # Check other new sums
        # new_sums = [x + ss for ss in suffix_sums]
        # We can check on the fly
        
        # For efficiency in the search loop, we might just want to check validity first.
        # If valid, we need to update the state.
        
        for ss in suffix_sums:
            if x + ss in sums_set:
                return False, None, None
        
        return True, None, None

    def get_new_state(seq, sums_set, suffix_sums, x):
        """
        Compute new sums_set and suffix_sums after adding x.
        """
        new_suffix_sums = [x]
        for ss in suffix_sums:
            new_suffix_sums.append(x + ss)
        
        new_sums_set = sums_set.union(new_suffix_sums)
        return new_sums_set, new_suffix_sums

    def greedy_build(strategy, deadline):
        """
        Build a sequence using a specific greedy strategy.
        strategy: 'smallest', 'largest', 'random'
        """
        seq = []
        sums_set = set()
        suffix_sums = []
        last_val = 0
        
        while time.time() < deadline:
            found = False
            # Determine candidate range
            # To be strictly increasing, x > last_val
            # Also x <= 256
            
            if strategy == 'smallest':
                candidates = range(last_val + 1, 257)
            elif strategy == 'largest':
                candidates = range(256, last_val, -1)
            else: # random
                # Sample a few random candidates to speed up
                # Or just iterate but pick randomly from valid ones?
                # Iterating 1..256 is fast enough (256 checks)
                candidates = list(range(last_val + 1, 257))
                random.shuffle(candidates)
            
            for x in candidates:
                valid, _, _ = is_valid_incremental(seq, sums_set, suffix_sums, x)
                if valid:
                    seq.append(x)
                    sums_set, suffix_sums = get_new_state(seq, sums_set, suffix_sums, x)
                    last_val = x
                    found = True
                    break
            
            if not found:
                break
        
        return seq

    def local_search(seq, deadline, max_iters=5000):
        """
        Improve the sequence using local search.
        """
        if not seq:
            return seq
            
        current_seq = list(seq)
        current_set = set(current_seq)
        
        # Precompute sums and suffix sums for current state
        # We need to recompute these if we swap/remove/add
        # For simplicity in local search, we will recompute is_valid from scratch
        # or use a faster check.
        # Given L ~ 25, is_valid is O(325) ops. 5000 iters * 325 = 1.6M ops. Very fast.
        
        def is_valid_full(s):
            if not s:
                return True
            L = len(s)
            P = [0] * (L + 1)
            for i in range(L):
                P[i + 1] = P[i] + s[i]
            sums = set()
            for i in range(L):
                for j in range(i + 1, L + 1):
                    val = P[j] - P[i]
                    if val in sums:
                        return False
                    sums.add(val)
            return True

        best_seq = list(current_seq)
        best_len = len(best_seq)
        
        iters = 0
        while time.time() < deadline and iters < max_iters:
            iters += 1
            r = random.random()
            
            if r < 0.6:
                # Add
                available = [x for x in range(1, 257) if x not in current_set]
                if available:
                    x = random.choice(available)
                    new_seq = current_seq + [x]
                    new_seq.sort()
                    if is_valid_full(new_seq):
                        current_seq = new_seq
                        current_set = set(current_seq)
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = list(current_seq)
            elif r < 0.8:
                # Swap
                if current_seq:
                    available = [x for x in range(1, 257) if x not in current_set]
                    if available:
                        x_rem = random.choice(current_seq)
                        x_add = random.choice(available)
                        new_seq = [x for x in current_seq if x != x_rem]
                        new_seq.append(x_add)
                        new_seq.sort()
                        if is_valid_full(new_seq):
                            current_seq = new_seq
                            current_set = set(current_seq)
                            if len(current_seq) > best_len:
                                best_len = len(current_seq)
                                best_seq = list(current_seq)
            else:
                # Remove (to escape)
                if current_seq and len(current_seq) > 1:
                    x_rem = random.choice(current_seq)
                    current_seq.remove(x_rem)
                    current_set = set(current_seq)
                    
        return best_seq

    best_seq = []
    
    # Run multiple strategies
    strategies = ['smallest', 'largest', 'random']
    for strat in strategies:
        if time.time() > deadline - 5:
            break
        seq = greedy_build(strat, deadline - 5)
        if len(seq) > len(best_seq):
            best_seq = seq

    # Run local search on the best greedy result
    if time.time() < deadline - 2:
        improved = local_search(best_seq, deadline - 2, max_iters=20000)
        if len(improved) > len(best_seq):
            best_seq = improved

    # Try a few random restarts of greedy + local search
    restarts = 0
    while time.time() < deadline - 2:
        restarts += 1
        # Random greedy start
        seq = greedy_build('random', deadline - 2)
        # Quick local search
        improved = local_search(seq, deadline - 2, max_iters=5000)
        if len(improved) > len(best_seq):
            best_seq = improved

    return sorted(best_seq)
