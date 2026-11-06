import time
import random
import math
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    N = 256
    
    # Precompute all possible block sums for a sequence of length L
    # But we don't know L in advance.
    
    # Strategy: Backtracking search for the longest sequence.
    # State: (last_value, frozenset_of_sums) is too large.
    # Instead, we build the sequence element by element.
    # Let a_1 < a_2 < ... < a_k be the sequence.
    # We choose a_1, then a_2, ..., a_k.
    # Constraint: a_i > a_{i-1} and all consecutive block sums are distinct.
    
    # Key Insight for Pruning:
    # If we have a partial sequence S, the set of block sums is fixed.
    # When adding a new element x at the end, the new block sums are:
    # x, x + S[-1], x + S[-1] + S[-2], ..., x + sum(S)
    # Let P_i be the suffix sums of S. P_0 = 0, P_1 = S[-1], P_2 = S[-1]+S[-2], etc.
    # New sums are {x + P_i for i=0..k-1}.
    # These must not intersect with existing sums, and must be distinct among themselves.
    # Since S is strictly increasing and positive, P_i are strictly increasing.
    # So x + P_i are distinct if x is fixed.
    # We just need (x + P_i) not in existing_sums for all i.
    
    # This check is O(k) per candidate x.
    # Total candidates for x: up to 256.
    # Depth: we want to maximize k.
    
    # Let's use a recursive backtracking search with pruning.
    # To make it fast, we should try to find a long sequence quickly.
    # Greedy first to get a baseline, then backtrack to improve?
    # Or just backtrack with good ordering.
    
    # Ordering: Try smaller x first? Or larger?
    # Smaller x keeps the sums smaller, potentially leaving more room?
    # Actually, larger x increases sums faster, might cause collisions sooner?
    # Let's try both or mix.
    
    # Memoization:
    # State is (last_val, tuple(sorted_sums))? No, sums set is huge.
    # But note: the set of sums is determined by the sequence.
    # Can we prune based on the "density" of sums?
    
    # Alternative: Use the fact that the number of block sums is k(k+1)/2.
    # Max possible sums in [1, 256*k] is 256*k.
    # So k(k+1)/2 <= 256*k => k+1 <= 512 => k <= 511.
    # But values are in [1, 256], so sum max is 256*k.
    # Actually, min sum is 1, max sum is 256*k.
    # Distinct sums count: k(k+1)/2.
    # Range of sums: [1, 256*k].
    # So k(k+1)/2 <= 256*k - 1.
    # k/2 + 0.5 <= 256 => k <= 511.
    # This bound is loose because values are bounded by 256.
    
    # Let's implement a DFS that builds the sequence from left to right.
    # We keep:
    # 1. Current sequence (list)
    # 2. Set of current block sums
    # 3. List of suffix sums (for quick check of new element)
    
    best_len = 0
    best_seq = []
    
    # To speed up, we can use a greedy initialization to find a decent length first,
    # then use DFS to try to beat it.
    # But DFS might be slow if branching factor is high.
    # Branching factor is ~256 at each step. Depth ~10-20?
    # 256^10 is too big.
    # We need strong pruning.
    
    # Pruning Idea:
    # If the current sequence length is L, and we are at value v,
    # the maximum possible additional length is bounded by how many values > v are left?
    # No, we can skip values.
    
    # Let's try a different angle:
    # The problem is equivalent to finding a subset of [1, 256] that can be ordered increasingly
    # such that all consecutive sums are distinct.
    # Since the order is fixed (increasing), we are just selecting a subset.
    # Let the selected values be v_1 < v_2 < ... < v_k.
    # Sums:
    # v_1
    # v_2
    # v_1+v_2
    # v_3
    # v_2+v_3
    # v_1+v_2+v_3
    # ...
    
    # Let's use a backtracking search where we decide for each value x in 1..256 whether to include it.
    # This is a subset selection problem.
    # State: (current_value, current_seq_suffix_sums, current_block_sums_set)
    # But current_block_sums_set is large.
    
    # However, we only need to check new sums against existing ones.
    # And we need to keep the existing ones to check future ones.
    
    # Is there a way to compress the state?
    # Not easily.
    
    # Let's stick to the DFS building the sequence.
    # To avoid exponential blowup, we will:
    # 1. Use a time limit.
    # 2. Use a "best found so far" to prune branches that can't beat it.
    #    How to bound potential length?
    #    If we are at depth d with last value v, the remaining values are v+1..256.
    #    Max additional length is 256 - v.
    #    So if d + (256 - v) <= best_len, prune.
    #    This is a weak bound.
    
    # Better bound:
    # The number of new sums added by extending the sequence by m elements is m(m+3)/2?
    # No, total sums is k(k+1)/2.
    # If we add m elements, we go from k to k+m.
    # New sums count: (k+m)(k+m+1)/2 - k(k+1)/2.
    # These new sums must be distinct from old ones and each other.
    # They must fit in the range [min_new_sum, max_new_sum].
    # Min new sum is at least v+1.
    # Max new sum is at most sum(current) + sum(remaining).
    
    # Let's just run the DFS with the simple bound and hope it's fast enough for small depths.
    # If not, we fall back to the greedy result.
    
    def dfs(last_val, seq, sums_set, suffix_sums):
        nonlocal best_len, best_seq
        
        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
        # Pruning: Max possible extension
        # Remaining values: last_val + 1 to 256
        # Max count: 256 - last_val
        # If current len + max_count <= best_len, return
        if len(seq) + (256 - last_val) <= best_len:
            return
            
        if time.time() > deadline - 2:
            return
            
        # Try next values
        # Heuristic: Try values that are "far" from last_val? Or close?
        # Close values create smaller new sums, which might collide with existing small sums.
        # Far values create larger new sums, which might be safer if existing sums are small.
        # But we also need distinctness among new sums.
        
        # Let's iterate through candidates
        # To speed up, we can skip candidates that are obviously bad?
        # Check: new_sums = {x + s for s in suffix_sums}
        # If any new_sum in sums_set, skip.
        
        # We can precompute for a given x? No, x changes.
        
        # Order of candidates:
        # Let's try a mix. Maybe randomize the order of candidates to explore different regions?
        # Or try largest first?
        
        # Let's try all candidates from last_val + 1 to 256.
        # To avoid deep recursion in Python, we might hit recursion limit.
        # Set recursion limit high?
        # Or use iterative deepening?
        
        # Let's use a stack for iterative DFS to avoid recursion limit and overhead.
        # But recursion is cleaner for backtracking.
        # Let's increase recursion limit.
        
        for x in range(last_val + 1, 257):
            # Check time
            if time.time() > deadline - 2:
                return
                
            # Compute new sums
            new_sums = []
            valid = True
            for s in suffix_sums:
                ns = x + s
                if ns in sums_set:
                    valid = False
                    break
                new_sums.append(ns)
            
            if not valid:
                continue
                
            # Check if new_sums are distinct (they are, since suffix_sums are distinct and x is constant)
            # Add to sums_set
            new_sums_set = sums_set.union(new_sums)
            
            # Update suffix_sums
            # Old suffix_sums: [0, v_k, v_k+v_{k-1}, ...]
            # New suffix_sums: [0, x, x+v_k, x+v_k+v_{k-1}, ...]
            # Actually, suffix_sums[i] is sum of last i elements.
            # New suffix_sums[0] = 0
            # New suffix_sums[1] = x
            # New suffix_sums[i] = x + old_suffix_sums[i-1] for i >= 2
            
            new_suffix_sums = [0]
            for s in suffix_sums[:-1]: # Exclude the last one which is sum of all?
                # Wait, suffix_sums list definition:
                # If seq is [a, b, c], suffix_sums should be [0, c, b+c, a+b+c]?
                # Let's define suffix_sums as list of sums of suffixes ending at last element.
                # s_0 = 0
                # s_1 = last
                # s_2 = last + second_last
                # ...
                # s_k = sum(all)
                
                # When adding x:
                # new_s_0 = 0
                # new_s_1 = x
                # new_s_2 = x + old_s_1
                # new_s_3 = x + old_s_2
                # ...
                # new_s_{k+1} = x + old_s_k
                
                new_suffix_sums.append(x + s)
                
            # Wait, my loop above was:
            # new_suffix_sums = [0]
            # for s in suffix_sums[:-1]: new_suffix_sums.append(x + s)
            # This misses the last one?
            # If suffix_sums has length k+1 (indices 0..k), then suffix_sums[:-1] has length k.
            # new_suffix_sums will have length 1 + k = k+1.
            # But it should have length k+2 (indices 0..k+1).
            # Let's fix.
            
            # Correct construction:
            new_suffix_sums = [0]
            for s in suffix_sums:
                new_suffix_sums.append(x + s)
                
            # Recurse
            dfs(x, seq + [x], new_sums_set, new_suffix_sums)
            
            # Optional: If we found a very long sequence, we might stop early?
            # No, we want the maximum.
            
            # If we used up time, return
            if time.time() > deadline - 2:
                return

    # Initialize DFS
    # Start with empty sequence?
    # last_val = 0
    # seq = []
    # sums_set = set()
    # suffix_sums = [0]
    
    # But this will try 1, 2, ... 256 as first element.
    # We can prune: if we start with a large number, we have fewer numbers left.
    # So starting small is better for length.
    
    # Let's start the DFS.
    # To improve speed, we can add a "greedy" phase first to set a high best_len.
    
    def greedy_construct():
        seq = []
        sums_set = set()
        suffix_sums = [0]
        last_val = 0
        
        for x in range(1, 257):
            # Check if x can be added
            new_sums = [x + s for s in suffix_sums]
            valid = True
            for ns in new_sums:
                if ns in sums_set:
                    valid = False
                    break
            
            if valid:
                seq.append(x)
                sums_set.update(new_sums)
                new_suffix_sums = [0] + [x + s for s in suffix_sums]
                suffix_sums = new_suffix_sums
                last_val = x
                
        return seq
        
    # Run greedy to get a baseline
    greedy_seq = greedy_construct()
    best_len = len(greedy_seq)
    best_seq = greedy_seq[:]
    
    # Now run DFS to try to beat it.
    # To avoid exploring too much, we can limit the depth or use time.
    # Also, we can randomize the order of candidates in DFS to find diverse solutions.
    # But standard DFS with increasing x is deterministic.
    
    # Let's modify DFS to accept a "mode" or just run it.
    # Since greedy might be stuck in a local optimum, DFS might find a better path.
    # However, DFS is exponential.
    # We will run DFS but with a tight time limit.
    
    # To make DFS more effective, we can try different starting strategies?
    # Or just let it run.
    
    # One issue: Python recursion depth.
    # Max depth is 256. Default limit is 1000. Should be fine.
    
    # Let's run DFS.
    # We'll use a global flag to stop if time is up.
    
    # To speed up DFS, we can skip x if x + min(suffix_sums) > max(sums_set) + some margin?
    # No, sums can be large.
    
    # Let's just run it.
    
    # To avoid re-exploring same states (if possible), we could memoize, but state is huge.
    
    # Let's try a bidirectional approach? No, DFS is simpler.
    
    # Let's add a small randomization to the candidate order in DFS to escape local optima?
    # No, DFS explores all. Randomization would just change the order.
    
    # Let's run DFS from the start.
    # But wait, if greedy found length 10, DFS will explore all sequences of length < 10 first?
    # No, DFS goes deep first.
    # So it will find a deep sequence quickly.
    
    # Let's run it.
    
    # One optimization: If we are at depth d, and the best possible length from here is d + (256 - last_val),
    # and d + (256 - last_val) <= best_len, we prune.
    # This is already in the code.
    
    # Let's call dfs.
    # To prevent the first few branches from taking too long, we can set a time limit per branch?
    # No, just global time.
    
    # Let's run.
    sys.setrecursionlimit(3000)
    
    # Start DFS
    # We can start from different initial points?
    # No, DFS covers all.
    
    # Let's run DFS.
    # Note: This might be slow if the branching factor is high.
    # We can reduce branching by only trying a subset of candidates?
    # No, we need completeness.
    
    # Let's try.
    
    dfs(0, [], set(), [0])
    
    # Final check
    if not best_seq:
        # Fallback
        best_seq = [1]
        
    # Verify
    def check_valid(seq):
        seen = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in seen:
                    return False
                seen.add(s)
        return True
        
    if not check_valid(best_seq):
        # If DFS produced invalid (shouldn't happen), fallback to greedy
        if check_valid(greedy_seq):
            best_seq = greedy_seq
        else:
            # Find a small valid one
            best_seq = []
            for x in range(1, 257):
                if check_valid(best_seq + [x]):
                    best_seq.append(x)
                    
    return best_seq

