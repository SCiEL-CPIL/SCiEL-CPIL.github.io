import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    N = 256
    best_seq = []
    best_len = 0

    def check_valid(seq):
        """Check if all non-empty consecutive block sums are distinct."""
        if not seq:
            return True
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_block_sums(seq):
        """Efficiently compute the set of all consecutive block sums."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def try_insert_and_extend(seq, candidates, depth_limit=5):
        """
        Attempt to extend the sequence by inserting elements from candidates.
        Uses a backtracking approach with pruning.
        Returns the longest valid sequence found.
        """
        nonlocal best_seq, best_len
        
        current = seq[:]
        current_set = set(current)
        current_sums = get_block_sums(current)
        
        if len(current) > best_len:
            best_len = len(current)
            best_seq = current[:]

        # If we have no candidates or reached depth limit, return
        if not candidates or depth_limit <= 0:
            return current

        # Sort candidates to prioritize certain strategies
        # We'll pass a sorted list or shuffled list depending on the caller
        
        # Try to add each candidate
        for x in candidates:
            if x in current_set:
                continue
            
            # Check if adding x is valid
            # We need to check if any new block sum involving x collides with existing sums
            # New sums are:
            # 1. x itself
            # 2. x + (suffixes of current)
            # 3. (prefixes of current) + x
            # 4. (prefixes of current) + x + (suffixes of current) ... wait, x is inserted in sorted order.
            
            # Since the sequence must be strictly increasing, x is inserted at a specific position.
            # Let's find the position.
            pos = 0
            while pos < len(current) and current[pos] < x:
                pos += 1
            
            # The new sequence would be current[:pos] + [x] + current[pos:]
            # We can optimize the sum check.
            # Existing sums are in current_sums.
            # New sums involving x:
            # Sums starting at x: x, x+current[pos], x+current[pos]+current[pos+1], ...
            # Sums ending at x: x, current[pos-1]+x, current[pos-2]+current[pos-1]+x, ...
            # Sums containing x in the middle: current[i..pos-1] + x + current[pos..j]
            
            # This optimization is complex. For N=256 and len(seq) < 30, 
            # recomputing sums for the new sequence is O(k^2) which is fast enough.
            # k ~ 20-30. k^2 ~ 900. 
            # Number of candidates ~ 256. 
            # 256 * 900 = 230,400 ops per level. 
            # Depth 5: 230k * 5 = 1.15M ops. Very fast.
            
            new_seq = current[:pos] + [x] + current[pos:]
            
            # Quick check: is x already a sum?
            if x in current_sums:
                continue
                
            # Full check
            if check_valid(new_seq):
                # Recurse
                remaining_candidates = [c for c in candidates if c != x]
                new_best = try_insert_and_extend(new_seq, remaining_candidates, depth_limit - 1)
                
                if len(new_best) > len(current):
                    current = new_best
                    current_set = set(current)
                    current_sums = get_block_sums(current)
                    if len(current) > best_len:
                        best_len = len(current)
                        best_seq = current[:]
                    
                    # If we improved, we should continue trying to add more to this new state
                    # But to keep it simple and fast, we just return the best found in this branch
                    # Actually, we want to explore further. The recursion handles that.
                    # However, if we find a valid insertion, we should prioritize extending *that* valid sequence.
                    # The current structure tries x, recurses, then tries next x.
                    # This is a depth-first search.
                    
        return current

    def greedy_build(strategy):
        """
        Build a sequence using a greedy strategy.
        strategy: 'asc', 'desc', 'random', 'sparse_asc', 'sparse_desc'
        """
        seq = []
        used = set()
        
        # Determine candidate order
        if strategy == 'asc':
            candidates = list(range(1, N + 1))
        elif strategy == 'desc':
            candidates = list(range(N, 0, -1))
        elif strategy == 'random':
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        elif strategy == 'sparse_asc':
            # Start with larger gaps
            candidates = list(range(1, N + 1, 2))
            random.shuffle(candidates)
        elif strategy == 'sparse_desc':
            candidates = list(range(N, 0, -2))
            random.shuffle(candidates)
        else:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)

        for x in candidates:
            if x in used:
                continue
            new_seq = seq + [x]
            new_seq.sort()
            if check_valid(new_seq):
                seq = new_seq
                used.add(x)
                
        return seq

    # Multi-start with different strategies
    strategies = [
        'desc', 'asc', 'random', 'sparse_desc', 'sparse_asc'
    ]
    
    # Add more random variations
    for _ in range(20):
        strategies.append('random')
        
    for strat in strategies:
        if time.time() > deadline - 5:
            break
            
        seq = greedy_build(strat)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Use the greedy result as a starting point for backtracking extension
        # We try to insert the remaining numbers
        used = set(seq)
        remaining = [x for x in range(1, N + 1) if x not in used]
        
        # Shuffle remaining to vary the insertion order in backtracking
        random.shuffle(remaining)
        
        # Limit the depth of backtracking to keep it fast
        # We don't want to explore the full tree, just a few levels of "what if I had inserted this earlier?"
        # Or rather, "what if I add this now?"
        # The backtracking function above tries to add *one* element and recurses.
        # This is effectively a depth-first search for the next k elements.
        
        extended_seq = try_insert_and_extend(seq, remaining, depth_limit=4)
        
        if len(extended_seq) > best_len:
            best_len = len(extended_seq)
            best_seq = extended_seq[:]

    # Final local search on the best sequence found
    if best_seq:
        current = best_seq[:]
        current_set = set(current)
        
        # Try simple swaps to improve
        for _ in range(50):
            if time.time() > deadline - 1:
                break
                
            improved = False
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                break
                
            random.shuffle(candidates)
            
            for i_val in current:
                for j_val in candidates[:10]: # Limit swap candidates
                    new_set = current_set - {i_val} | {j_val}
                    new_seq = sorted(new_set)
                    if check_valid(new_seq):
                        current = new_seq
                        current_set = new_set
                        improved = True
                        if len(current) > best_len:
                            best_len = len(current)
                            best_seq = current[:]
                        break
                if improved:
                    break
                    
            if not improved:
                break

    return best_seq

if __name__ == "__main__":
    res = construct()
    print(len(res))
