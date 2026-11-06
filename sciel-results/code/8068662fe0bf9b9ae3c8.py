import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Structural approach: super-increasing gaps
    # If gaps g_i are super-increasing (g_i > sum of all previous gaps),
    # then all consecutive block sums are distinct.
    # We need a_1 < a_2 < ... < a_n with a_i in [1,256]
    # Let g_i = a_{i+1} - a_i for i=1..n-1, and a_1 >= 1
    # Super-increasing: g_i > g_1 + g_2 + ... + g_{i-1}
    # This ensures all consecutive block sums are distinct.
    
    def structural_construct():
        # Try to build a sequence with super-increasing gaps
        # Start with small a_1, then choose gaps that are super-increasing
        best = []
        
        # Try different starting values and gap sequences
        for a1 in range(1, 30):
            gaps = []
            a = a1
            seq = [a1]
            sum_gaps = 0
            
            # Choose super-increasing gaps
            # g_1 >= 1, g_2 > g_1, g_3 > g_1+g_2, etc.
            # To maximize length, keep gaps as small as possible while maintaining super-increasing
            # Minimum super-increasing sequence: 1, 2, 4, 8, 16, ...
            # But we can do better by choosing gaps more carefully
            
            # Greedy: choose smallest valid gap at each step
            while True:
                # Next gap must be > sum_gaps (if sum_gaps > 0) or >= 1 (if sum_gaps == 0)
                min_gap = sum_gaps + 1 if sum_gaps > 0 else 1
                # Also need a + min_gap <= 256
                if a + min_gap > 256:
                    break
                # Try smallest gap first
                g = min_gap
                a_next = a + g
                if a_next > 256:
                    break
                gaps.append(g)
                sum_gaps += g
                a = a_next
                seq.append(a)
            
            if len(seq) > len(best):
                best = seq[:]
        
        return best
    
    def greedy_construct():
        values = []
        prefix = [0]
        diffs = set()
        last_a = 0
        for a in range(1, 257):
            if a <= last_a:
                continue
            # Check if adding 'a' is valid
            new_p = prefix[-1] + a
            new_diffs = []
            valid = True
            for i in range(len(prefix)):
                d = new_p - prefix[i]
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                values.append(a)
                prefix.append(new_p)
                diffs.update(new_diffs)
                last_a = a
        return values
    
    # Try structural approach first
    best_seq = structural_construct()
    best_len = len(best_seq)
    
    # Try greedy approach
    greedy_seq = greedy_construct()
    if len(greedy_seq) > best_len:
        best_seq = greedy_seq
        best_len = len(greedy_seq)
    
    # DFS to try to extend the sequence
    def dfs(prefix, diffs, last_a, depth, current_seq):
        nonlocal best_seq, best_len, deadline
        
        if time.time() > deadline:
            return
        
        # Pruning: if we can't beat best_len, return
        if depth + (256 - last_a) <= best_len:
            return
            
        if depth > best_len:
            best_len = depth
            best_seq = current_seq[:]
        
        # Try next a with heuristic ordering
        # Try values that are "far" from existing prefix sums to reduce collisions
        candidates = list(range(last_a + 1, 257))
        
        # Sort by how "unique" the new diffs would be
        # Heuristic: prefer a where new_p - prefix[i] are large and distinct
        def score_candidate(a):
            new_p = prefix[-1] + a
            min_diff = float('inf')
            for i in range(len(prefix)):
                d = new_p - prefix[i]
                if d in diffs:
                    return -1  # Invalid
                min_diff = min(min_diff, d)
            return min_diff  # Prefer larger minimum diff
        
        # Sort candidates by score (descending), but keep some randomness
        scored = [(score_candidate(a), a) for a in candidates]
        scored = [x for x in scored if x[0] >= 0]  # Only valid ones
        scored.sort(key=lambda x: -x[0])
        
        # Limit branching factor for performance
        max_branches = 20 if depth < 10 else 10
        
        for _, a in scored[:max_branches]:
            new_p = prefix[-1] + a
            new_diffs = []
            valid = True
            for i in range(len(prefix)):
                d = new_p - prefix[i]
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Recurse
            prefix.append(new_p)
            diffs.update(new_diffs)
            current_seq.append(a)
            
            dfs(prefix, diffs, a, depth + 1, current_seq)
            
            # Backtrack
            prefix.pop()
            diffs.difference_update(new_diffs)
            current_seq.pop()
            
            if time.time() > deadline:
                return
    
    # Start DFS from the best sequence found so far
    if time.time() < deadline - 5:
        prefix = [0]
        diffs = set()
        last_a = 0
        for a in best_seq:
            new_p = prefix[-1] + a
            for i in range(len(prefix)):
                diffs.add(new_p - prefix[i])
            prefix.append(new_p)
            last_a = a
        
        dfs(prefix, diffs, last_a, len(best_seq), best_seq[:])
    
    # Local search: try to extend by replacing last few elements
    if time.time() < deadline - 2:
        # Try to extend by modifying the tail
        for trim in range(0, min(5, len(best_seq))):
            if time.time() > deadline - 1:
                break
            base = best_seq[:-trim] if trim > 0 else best_seq[:]
            if not base:
                continue
            prefix = [0]
            diffs = set()
            last_a = 0
            for a in base:
                new_p = prefix[-1] + a
                for i in range(len(prefix)):
                    diffs.add(new_p - prefix[i])
                prefix.append(new_p)
                last_a = a
            
            # DFS from this shorter base
            dfs(prefix, diffs, last_a, len(base), base[:])
    
    return best_seq
