"""Structurally different approach: bitset-based forbidden-sum tracking + iterative deepening DFS.

Key structural changes from the seed:
1. Bitset (Python int as arbitrary-precision bitmask) for O(1) collision checks.
2. Iterative deepening DFS: search for the longest valid sequence by incrementally
   trying to extend a known-good prefix.
3. Suffix-sum tracking maintained incrementally for fast new-sum computation.
4. Aggressive pruning via upper-bound on remaining elements.
"""
import time
import sys

_DEADLINE = 82.0  # Safety margin before 90s kill


def _compute_suffix_sums(seq):
    """Return list of suffix sums: suffix_sums[i] = sum(seq[i:])."""
    n = len(seq)
    ss = [0] * n
    if n > 0:
        ss[-1] = seq[-1]
        for i in range(n - 2, -1, -1):
            ss[i] = ss[i + 1] + seq[i + 1]
    return ss


def _all_block_sums(seq):
    """Compute set of all non-empty consecutive block sums for seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums


def _bitmask_from_set(sums):
    """Convert a set of positive integers to a bitmask (Python int)."""
    mask = 0
    for s in sums:
        if s >= 0:
            mask |= (1 << s)
    return mask


def _new_sums_bitmask(x, suffix_sums):
    """Compute bitmask of new block sums introduced by appending x.
    
    New sums are: x, x + suffix_sums[-1], x + suffix_sums[-2], ..., x + suffix_sums[0]
    i.e., x plus each suffix sum of the current sequence.
    If suffix_sums is empty, just {x}.
    """
    mask = 1 << x  # x itself
    for ss in suffix_sums:
        s = x + ss
        mask |= (1 << s)
    return mask


def construct():
    start_time = time.time()
    
    # Upper bound: we need L(L+1)/2 distinct sums.
    # Sums range from at least 1 to at most L*256.
    # For L=20: 210 sums, max sum 5120. Feasible range.
    # For L=30: 465 sums, max sum 7680. Still feasible.
    # The real constraint is the distinctness under the increasing structure.
    # Powers of 2 give 9. We expect maybe 12-15.
    
    best_seq = []
    best_sums_mask = 0
    best_sums_set = set()
    
    def _time_ok():
        return time.time() < _DEADLINE
    
    def _update_best(seq):
        nonlocal best_seq, best_sums_mask, best_sums_set
        if len(seq) > len(best_seq):
            best_seq = seq[:]
            best_sums_set = _all_block_sums(seq)
            best_sums_mask = _bitmask_from_set(best_sums_set)
    
    # Greedy baseline for initial best
    greedy_seq = []
    greedy_sums = set()
    for x in range(1, 257):
        if x in greedy_sums:
            continue
        # Check new suffix sums
        ok = True
        s = x
        for a in reversed(greedy_seq):
            s += a
            if s in greedy_sums:
                ok = False
                break
        if not ok:
            continue
        greedy_seq.append(x)
        s = x
        greedy_sums.add(s)
        for a in reversed(greedy_seq[:-1]):
            s += a
            greedy_sums.add(s)
    if len(greedy_seq) > len(best_seq):
        _update_best(greedy_seq)
    
    # Also try the powers-of-2 baseline
    pow2_seq = [2**i for i in range(9)]  # 1,2,4,...,256
    if len(pow2_seq) > len(best_seq):
        _update_best(pow2_seq)
    
    if not _time_ok():
        return best_seq
    
    # Iterative deepening DFS
    # Start from the best length found so far, try to find longer sequences.
    base_len = len(best_seq)
    
    def _dfs(current_seq, current_sums_mask, suffix_sums, depth):
        """DFS to build a sequence of length >= target_len.
        
        current_seq: list of chosen elements
        current_sums_mask: bitmask of all block sums so far
        suffix_sums: list of suffix sums of current_seq (suffix_sums[i] = sum(current_seq[i:]))
        depth: current depth
        Returns True if a valid extension to target_len was found (and updates best).
        """
        nonlocal best_seq, best_sums_mask, best_sums_set
        
        if not _time_ok():
            return False
        
        if depth == target_len:
            _update_best(current_seq)
            return True
        
        # Pruning: upper bound on remaining elements
        if current_seq:
            last = current_seq[-1]
            max_remaining = 256 - last
            if depth + max_remaining < target_len:
                return False
        
        start_val = current_seq[-1] + 1 if current_seq else 1
        if start_val > 256:
            return False
        
        # Iterate candidate x values
        # Heuristic: try values that are "spread out" to avoid sum collisions
        # But for finding any valid extension, try in order.
        # Optimization: skip x if (1 << x) & current_sums_mask != 0
        # This is a quick pre-check: x itself must not be a sum already.
        
        for x in range(start_val, 257):
            if not _time_ok():
                return False
            
            # Quick check: x not already a block sum
            if (current_sums_mask >> x) & 1:
                continue
            
            # Compute new suffix sums when appending x
            # New suffix sums: [x, x + suffix_sums[-1], x + suffix_sums[-2], ..., x + suffix_sums[0]]
            new_suffix = [x]
            if suffix_sums:
                for ss in reversed(suffix_sums):
                    new_suffix.append(x + ss)
            new_suffix.reverse()  # Now new_suffix[i] = sum of new sequence from i to end
            # Actually let's be more careful:
            # Old suffix_sums: [s0, s1, ..., s_{d-1}] where si = sum(current_seq[i:])
            # New sequence: current_seq + [x]
            # New suffix sums:
            #   new_ss[d] = x
            #   new_ss[i] = current_seq[i] + ... + current_seq[d-1] + x = suffix_sums[i] + x for i < d
            # So new_suffix = [suffix_sums[0]+x, suffix_sums[1]+x, ..., suffix_sums[d-1]+x, x]
            
            new_suffix = []
            if suffix_sums:
                for ss in suffix_sums:
                    new_suffix.append(ss + x)
            new_suffix.append(x)
            
            # Compute bitmask of new sums to add
            new_mask = 0
            for s in new_suffix:
                new_mask |= (1 << s)
            
            # Check for collision with existing sums
            if new_mask & current_sums_mask:
                continue
            
            # Valid move
            current_seq.append(x)
            new_sums_mask = current_sums_mask | new_mask
            
            result = _dfs(current_seq, new_sums_mask, new_suffix, depth + 1)
            if result:
                return True
            
            current_seq.pop()
            
            # Pruning: if even with the maximum possible remaining elements we can't reach target_len
            if depth + 1 + (256 - x) < target_len:
                return False
        
        return False
    
    # Try to find sequences longer than base_len
    # Start from base_len + 1 and go up
    # But IDDFS: try to find a sequence of exactly target_len
    # We can reuse the best_seq as a starting prefix if it's shorter than target_len
    
    max_search_len = min(base_len + 10, 25)  # Cap to avoid excessive search
    for target_len in range(base_len + 1, max_search_len + 1):
        if not _time_ok():
            break
        
        # Try DFS starting from empty
        # But we can also try starting from the best_seq prefix
        # Let's just do a fresh DFS with the target length
        
        found = _dfs([], 0, [], 0)
        if found:
            # We found a sequence of length target_len
            # Try to extend further
            continue
        else:
            # Could not find a sequence of this length
            # Since any sequence of length L+1 has a prefix of length L,
            # if no sequence of length L exists, no longer one exists either.
            # But this is only true if we exhaustively searched. Our DFS might have
            # hit the time limit and not fully explored.
            # So we break to be safe.
            break
    
    if not _time_ok():
        return best_seq
    
    # If we still have time, try to extend the best_seq further with a direct search
    if len(best_seq) < 25:
        current_seq = best_seq[:]
        current_sums_mask = best_sums_mask
        suffix_sums = _compute_suffix_sums(current_seq)
        
        def _extend_dfs(seq, mask, ss, depth):
            if not _time_ok():
                return False
            if depth > len(best_seq):
                _update_best(seq)
            
            if seq:
                last = seq[-1]
                max_rem = 256 - last
                if depth + max_rem <= len(best_seq):
                    return False
            
            start_val = seq[-1] + 1 if seq else 1
            if start_val > 256:
                return False
            
            improved = False
            for x in range(start_val, 257):
                if not _time_ok():
                    return improved
                
                if (mask >> x) & 1:
                    continue
                
                new_ss = [s + x for s in ss] if ss else []
                new_ss.append(x)
                
                new_mask = 0
                for s in new_ss:
                    new_mask |= (1 << s)
                
                if new_mask & mask:
                    continue
                
                seq.append(x)
                improved |= _extend_dfs(seq, mask | new_mask, new_ss, depth + 1)
                seq.pop()
                
                if depth + 1 + (256 - x) <= len(best_seq):
                    break
            
            return improved
        
        _extend_dfs(current_seq, current_sums_mask, suffix_sums, len(current_seq))
    
    if not best_seq:
        # Absolute fallback
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    return best_seq
