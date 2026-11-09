import time
import random
import sys

# Set recursion limit higher just in case, though iterative is safer
sys.setrecursionlimit(10000)

def construct():
    """
    Returns a strictly increasing sequence of integers in [1, 256] 
    with distinct consecutive block sums, maximizing the length.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_sequence = []
    best_length = 0
    
    # We will perform multiple randomized backtracking searches.
    # Each search tries to build a sequence by picking the next element
    # from the remaining candidates in a randomized order.
    
    def get_consecutive_sums(seq):
        """
        Returns a set of all non-empty consecutive block sums of seq.
        """
        sums = set()
        n = len(seq)
        for i in range(n):
            current_sum = 0
            for j in range(i, n):
                current_sum += seq[j]
                sums.add(current_sum)
        return sums

    def is_valid_extension(seq, new_val):
        """
        Checks if adding new_val to seq results in a valid sequence.
        Assumes seq is already valid and strictly increasing.
        We only need to check sums involving new_val.
        """
        if not seq or new_val <= seq[-1]:
            return False
        
        # The new consecutive blocks are those ending at new_val.
        # They are: [new_val], [seq[-1], new_val], [seq[-2], seq[-1], new_val], etc.
        
        # We need the set of all existing sums to check for collisions.
        # To be efficient, we can maintain the set of sums incrementally, 
        # but for simplicity in the check function, we might just recompute 
        # or pass the set. Let's pass the set to avoid recomputation.
        
        # Actually, let's restructure to maintain the set of sums globally 
        # in the backtracking function for efficiency.
        pass

    def backtrack(seq, current_sums, remaining_candidates):
        """
        Recursively builds the sequence.
        seq: current sequence
        current_sums: set of all consecutive block sums in seq
        remaining_candidates: sorted list of available values > seq[-1]
        """
        nonlocal best_sequence, best_length
        
        if time.time() > deadline:
            return
            
        if len(seq) > best_length:
            best_length = len(seq)
            best_sequence = seq[:]
            
        # If no candidates left, we can't extend further
        if not remaining_candidates:
            return
            
        # Try each candidate in a randomized order
        # To improve performance, we can shuffle a copy of remaining_candidates
        # but that's expensive. Instead, we pick a random index each time.
        # Or, we can sort by some heuristic? 
        # Randomization is key to escaping local optima.
        
        # Optimization: If we pick a value, we must remove it and all values <= it 
        # from consideration for future steps? No, we must pick strictly increasing.
        # So if we pick x, the next must be > x.
        
        # Let's iterate through candidates. To randomize, we can shuffle the list 
        # of candidates at each level? That's O(N^2) overhead.
        # Better: pick a random candidate, try it, then remove it and all smaller ones 
        # from the *available* set for the next level? 
        # Wait, the next value must be > current value. 
        # So if we have remaining_candidates sorted, and we pick index i, 
        # the next candidates are remaining_candidates[i+1:].
        
        # To randomize, we can shuffle the remaining_candidates list at the start of 
        # each backtrack call? No, that changes the order for siblings.
        # Standard randomized DFS: shuffle the children at each node.
        
        # Let's copy and shuffle remaining_candidates for this level.
        # This is O(K log K) where K is number of candidates.
        # Max length is small (maybe < 20?), so K is at most 256.
        # This is acceptable.
        
        candidates = remaining_candidates[:]
        random.shuffle(candidates)
        
        for val in candidates:
            # Since we shuffled, we don't know if val is > seq[-1] relative to original order,
            # but remaining_candidates only contains values > seq[-1] (if we filter correctly).
            # Let's ensure remaining_candidates passed is already filtered.
            
            # Check validity
            # New sums are formed by appending val to all suffixes of seq.
            # Suffixes: seq[-1], seq[-2:-1], ..., seq
            # New sums: val, seq[-1]+val, seq[-2]+seq[-1]+val, ..., sum(seq)+val
            
            valid = True
            new_sums_to_add = []
            current_sum = val
            for i in range(len(seq) - 1, -1, -1):
                # i goes from len(seq)-1 down to 0
                # The block is seq[i] + ... + seq[-1] + val
                # Wait, the block ending at val starting at i is sum(seq[i:]) + val
                # Let's compute suffix sums of seq to make this faster.
                # But len(seq) is small.
                if current_sum in current_sums:
                    valid = False
                    break
                new_sums_to_add.append(current_sum)
                current_sum += seq[i]
                
            # The last one was sum(seq[0:]) + val? 
            # Let's trace:
            # i = len(seq)-1: block is [seq[-1], val] -> sum = seq[-1] + val. 
            # My loop:
            # init current_sum = val.
            # i = len(seq)-1: check val? No, the block [val] is just val.
            # The blocks ending at val are:
            # 1. [val] -> sum = val
            # 2. [seq[-1], val] -> sum = seq[-1] + val
            # 3. [seq[-2], seq[-1], val] -> sum = seq[-2] + seq[-1] + val
            # ...
            # k. [seq[0], ..., seq[-1], val] -> sum = sum(seq) + val
            
            # My loop logic:
            # current_sum starts as val.
            # Iter 1 (i=len-1): check current_sum (val). Add to new_sums. current_sum += seq[len-1].
            # Iter 2 (i=len-2): check current_sum (val + seq[len-1]). Add. current_sum += seq[len-2].
            # ...
            # This covers [val], [seq[-1], val], ..., [seq[0..-1], val].
            # Correct.
            
            if valid:
                # Also need to check if any new_sum collides with each other?
                # No, they are all distinct because they are sums of different length suffixes 
                # of a strictly increasing positive integer sequence.
                # e.g., val < val+seq[-1] < val+seq[-1]+seq[-2] ...
                # So they are distinct from each other.
                # We only need to check against existing current_sums.
                
                # Update state
                next_sums = current_sums.union(new_sums_to_add)
                # Filter remaining candidates: must be > val
                # Since remaining_candidates is sorted, we can binary search or just filter
                # But we shuffled 'candidates'. We should filter the ORIGINAL remaining_candidates 
                # or just pass a filtered version.
                # Actually, for the next call, we need candidates > val from the ORIGINAL pool.
                # Let's find the index of val in the sorted remaining_candidates.
                # Or simpler: just filter the original list.
                
                # Optimization: Since remaining_candidates is sorted, 
                # we can precompute indices.
                # For now, simple filter.
                next_remaining = [x for x in remaining_candidates if x > val]
                
                seq.append(val)
                backtrack(seq, next_sums, next_remaining)
                seq.pop()
                
                if time.time() > deadline:
                    return

    # Main loop: Multiple randomized restarts
    # The space of sequences is large. A single random path might not find the global optimum.
    # We run multiple backtracking searches, each starting with a random shuffle of initial candidates.
    
    # To improve efficiency, we can use a heuristic: 
    # Start with small numbers? Or random?
    # Randomization is safer for exploring different regions.
    
    # Let's run backtracking multiple times until deadline.
    # Each call to backtrack starts from empty sequence.
    # To make them different, we can seed the random generator differently or 
    # just rely on the internal shuffling.
    # However, the first few steps are critical. 
    # We can also try a "greedy with backtracking" hybrid.
    
    # Let's just run the randomized backtracking.
    # To avoid getting stuck in the same local optima, the shuffling at each node 
    # ensures different paths are explored in each top-level call if we reset the state.
    # But wait, if we call backtrack([], set(), list(range(1, 257))), 
    # the shuffling happens inside. 
    # So each call to backtrack is a full random search of the tree? 
    # No, it's a depth-first search with randomized children order. 
    # It will explore one path deeply, then backtrack. 
    # It will eventually explore the whole tree (if time permits). 
    # But 256 candidates, branching factor ~256. Depth ~15? 
    # 256^15 is huge. It will only explore a tiny fraction.
    # So each "restart" (new call to backtrack) is just a different random walk 
    # if we stop early? No, it's a DFS. It will go deep, hit a dead end, backtrack, 
    # try next sibling, etc.
    # So one call to backtrack is a comprehensive search of the accessible tree 
    # within the time limit.
    # However, if it gets stuck in a deep, unproductive branch, it might waste time.
    # Randomizing the order helps.
    
    # Let's run it once, but with a strong time check.
    # If we want multiple independent random walks, we would need to stop the DFS 
    # after a certain depth or time and restart. 
    # But standard DFS with randomization is usually good enough for these sizes.
    
    # One improvement: Sort candidates by a heuristic? 
    # E.g., prefer smaller numbers to allow more room? 
    # Or prefer numbers that create "sparse" sums?
    # Randomization is a good baseline.
    
    # Let's add a small optimization: 
    # If we have found a sequence of length L, we can prune branches 
    # that cannot possibly exceed L. 
    # Max possible length from current state: 
    # If we are at depth d, and we have candidates C, 
    # the max additional length is len(C). 
    # So if d + len(C) <= best_length, we can prune.
    
    def backtrack_optimized(seq, current_sums, remaining_candidates):
        nonlocal best_sequence, best_length
        
        if time.time() > deadline:
            return
            
        current_len = len(seq)
        max_possible_len = current_len + len(remaining_candidates)
        
        if max_possible_len <= best_length:
            return
            
        if current_len > best_length:
            best_length = current_len
            best_sequence = seq[:]
            
        if not remaining_candidates:
            return
            
        candidates = remaining_candidates[:]
        random.shuffle(candidates)
        
        for val in candidates:
            # Pruning: if even with this val and all larger ones we can't beat best, skip?
            # We don't know how many larger ones will be usable.
            
            # Check validity
            valid = True
            new_sums_to_add = []
            current_sum = val
            for i in range(len(seq) - 1, -1, -1):
                if current_sum in current_sums:
                    valid = False
                    break
                new_sums_to_add.append(current_sum)
                current_sum += seq[i]
                
            if valid:
                next_sums = current_sums.union(new_sums_to_add)
                next_remaining = [x for x in remaining_candidates if x > val]
                
                seq.append(val)
                backtrack_optimized(seq, next_sums, next_remaining)
                seq.pop()
                
                if time.time() > deadline:
                    return

    # Run the search
    # We can run multiple iterations with different random seeds to explore different regions.
    # But the internal shuffling already randomizes the path.
    # Let's just run it once. If it finishes early, we can run more.
    
    initial_candidates = list(range(1, 257))
    
    # To ensure we explore diverse regions, we can run the backtracking 
    # a few times. Each run is a full DFS with random order.
    # But DFS is deterministic given the random seed. 
    # So we can just call it multiple times.
    
    # Let's do it.
    while time.time() < deadline - 2.0:
        random.seed() # Re-seed to get different shuffles
        backtrack_optimized([], set(), initial_candidates)
        
        # If we've used most of the time, break
        if time.time() > deadline - 2.0:
            break
            
    return best_sequence

# Test/Entry point
if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
