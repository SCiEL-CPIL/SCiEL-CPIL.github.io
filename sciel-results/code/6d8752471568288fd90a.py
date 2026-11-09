import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85  # 5s buffer
    
    def compute_suffix_sums(seq):
        """
        Returns a set of all suffix sums of the sequence, including 0.
        For seq = [a, b, c], suffix sums are 0, c, b+c, a+b+c.
        """
        suffixes = [0]
        acc = 0
        for val in reversed(seq):
            acc += val
            suffixes.append(acc)
        return set(suffixes)

    def compute_all_sums(seq):
        """
        Returns a set of all non-empty consecutive block sums.
        """
        sums = set()
        n = len(seq)
        for i in range(n):
            acc = 0
            for j in range(i, n):
                acc += seq[j]
                sums.add(acc)
        return sums

    def try_extend(seq, strategy, max_val=256):
        """
        Attempts to extend the sequence using the given strategy.
        Returns the extended sequence.
        """
        if not seq:
            return seq
            
        current = list(seq)
        # Precompute sums and suffixes for the current state
        # It's more efficient to maintain these incrementally, 
        # but for simplicity and correctness in a restart-based search, 
        # we can recompute or update. 
        # Given small n, recomputing is fine, but updating is better.
        
        # Let's use incremental update for speed.
        # We need S_prev and Suffix.
        S_prev = compute_all_sums(current)
        Suffix = compute_suffix_sums(current)
        
        last_val = current[-1]
        
        while time.time() < DEADLINE:
            # Determine candidate range
            min_candidate = last_val + 1
            if min_candidate > max_val:
                break
            
            # Find all valid candidates
            # Forbidden = { a - b for a in S_prev for b in Suffix }
            # We can optimize this check.
            # Instead of building the full forbidden set (which can be large),
            # we can check each candidate x in [min_candidate, max_val]
            # and see if (x + Suffix) intersects S_prev.
            
            # However, building the forbidden set once per step might be faster if the range is large.
            # Range size is up to 256.
            # |S_prev| ~ n^2/2. |Suffix| ~ n.
            # Intersection check for one x: O(|Suffix| * log(|S_prev|)) or O(|Suffix|) if S_prev is a set.
            # Total check for all candidates: O(256 * n).
            # Building Forbidden: O(|S_prev| * |Suffix|) = O(n^3).
            # For n=20, n^3 = 8000. 256*n = 5120. Similar.
            # Let's just check candidates directly. It's simpler and avoids memory overhead of large sets if n grows.
            
            valid_candidates = []
            for x in range(min_candidate, max_val + 1):
                # Check if x + s in S_prev for any s in Suffix
                conflict = False
                for s in Suffix:
                    if (x + s) in S_prev:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                break
            
            # Select next value based on strategy
            if strategy == 'min':
                next_val = valid_candidates[0]
            elif strategy == 'max':
                next_val = valid_candidates[-1]
            elif strategy == 'random':
                next_val = random.choice(valid_candidates)
            else:
                next_val = valid_candidates[0]
                
            # Update state
            current.append(next_val)
            last_val = next_val
            
            # Update S_prev and Suffix
            # New sums are x + s for s in Suffix
            new_sums = set()
            for s in Suffix:
                new_sums.add(next_val + s)
            
            # Check for conflicts within new sums? 
            # No, new sums are distinct because Suffix is distinct.
            # But we already checked against S_prev.
            
            S_prev.update(new_sums)
            
            # Update Suffix
            # New suffixes are next_val + old_suffixes, plus next_val (which is next_val + 0)
            # Actually, Suffix for new sequence [seq..., x] is:
            # {x} U {x + s for s in Suffix_old}
            # Wait, Suffix_old includes 0. So x + 0 = x.
            # So Suffix_new = {x + s for s in Suffix_old}
            Suffix = {next_val + s for s in Suffix}
            
            # Optimization: If Suffix grows too large or S_prev, we might slow down.
            # But n is small.
            
        return current

    best_seq = []
    
    # Strategies to try
    strategies = ['min', 'max', 'random']
    
    # Run multiple random restarts
    num_restarts = 100000 # High number, but each run is fast
    
    for i in range(num_restarts):
        if time.time() > DEADLINE:
            break
            
        # Use 'min' and 'max' deterministically first, then random
        if i < 2:
            strat = strategies[i]
        else:
            strat = 'random'
            
        # Start with a random initial sequence?
        # Or start empty?
        # Starting empty is standard.
        # To diversify, we can start with a random first element?
        # The first element can be anything in 1..256.
        # If we start with 1, we are constrained.
        # Let's try starting with a random first element for random restarts.
        
        if strat == 'random' and i > 2:
            # Pick a random start
            start_val = random.randint(1, 256)
            seq = [start_val]
        else:
            seq = []
            
        extended = try_extend(seq, strat)
        
        if len(extended) > len(best_seq):
            best_seq = extended
            # print(f"New best length: {len(best_seq)} at iteration {i}") # Debug
            
    # Final verification
    if not best_seq:
        return []
        
    # Check strictly increasing
    for i in range(len(best_seq) - 1):
        if best_seq[i] >= best_seq[i+1]:
            return best_seq[:i+1]
            
    # Check distinct sums
    sset = set()
    n = len(best_seq)
    for i in range(n):
        acc = 0
        for j in range(i, n):
            acc += best_seq[j]
            if acc in sset:
                # Invalid, trim
                return best_seq[:j]
            sset.add(acc)
            
    return best_seq
