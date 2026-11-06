"""Rewritten program: CSP/backtracking with prefix-sum difference pruning."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

DEADLINE_MARGIN = 5.0  # seconds to leave before 90s limit

def _valid_addition(seq, x, forbidden):
    """Check if appending x to seq creates any duplicate consecutive-block sum.
    seq is strictly increasing. forbidden is a set of all current block sums.
    Returns True if x can be appended without creating a collision."""
    # New block sums formed by appending x:
    # For each suffix of seq (including empty suffix), the new block sum is x + sum(suffix)
    # The suffix sums of seq are: 0, seq[-1], seq[-1]+seq[-2], ...
    # So new sums are: x, x+seq[-1], x+seq[-1]+seq[-2], ...
    s = x
    if s in forbidden:
        return False
    for a in reversed(seq):
        s += a
        if s in forbidden:
            return False
    return True

def _add_sums(seq, x, forbidden):
    """Add all new block sums from appending x to seq into forbidden."""
    s = x
    forbidden.add(s)
    for a in reversed(seq):
        s += a
        forbidden.add(s)

def construct():
    start_time = time.time()
    
    # Greedy seed to get a reasonable starting sequence
    greedy_seq = []
    greedy_forbidden = set()
    for x in range(1, 257):
        if _valid_addition(greedy_seq, x, greedy_forbidden):
            greedy_seq.append(x)
            _add_sums(greedy_seq, x, greedy_forbidden)
    
    best = list(greedy_seq)
    best_len = len(best)
    
    # Backtracking search to try to extend beyond greedy
    # We'll try to find a longer sequence by exploring different choices
    
    def backtrack(seq, forbidden, depth, max_depth):
        nonlocal best, best_len, start_time
        
        if time.time() - start_time > DEADLINE_MARGIN:
            return
        
        if len(seq) > best_len:
            best = list(seq)
            best_len = len(seq)
        
        if len(seq) >= max_depth:
            return
        
        # Try to extend seq by choosing next value
        last = seq[-1] if seq else 0
        for x in range(last + 1, 257):
            if time.time() - start_time > DEADLINE_MARGIN:
                return
            
            if _valid_addition(seq, x, forbidden):
                # Add x
                new_forbidden = set(forbidden)
                _add_sums(seq, x, new_forbidden)
                seq.append(x)
                backtrack(seq, new_forbidden, depth + 1, max_depth)
                seq.pop()
    
    # Try to extend beyond the greedy solution
    # Start from the greedy sequence and try different choices at the end
    # We'll do a limited backtracking from the greedy seed
    
    # First, try extending the greedy sequence directly
    seq = list(greedy_seq)
    forbidden = set(greedy_forbidden)
    
    # Try to add more elements after the greedy sequence
    last = seq[-1] if seq else 0
    for x in range(last + 1, 257):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        if _valid_addition(seq, x, forbidden):
            seq.append(x)
            _add_sums(seq, x, forbidden)
    
    if len(seq) > best_len:
        best = list(seq)
        best_len = len(seq)
    
    # Now try backtracking from shorter prefixes to find better solutions
    # Try different starting points and explore
    for prefix_len in range(min(10, best_len), 0, -1):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
        
        # Use the first prefix_len elements of the greedy sequence as a seed
        seed_seq = greedy_seq[:prefix_len]
        seed_forbidden = set()
        for i in range(len(seed_seq)):
            # Compute all block sums for seed_seq
            for j in range(i + 1):
                s = sum(seed_seq[j:i+1])
                seed_forbidden.add(s)
        
        # Now try to extend from here with backtracking
        seq2 = list(seed_seq)
        forbidden2 = set(seed_forbidden)
        
        def bt2(seq, forbidden, max_depth):
            nonlocal best, best_len, start_time
            
            if time.time() - start_time > DEADLINE_MARGIN:
                return
            
            if len(seq) > best_len:
                best = list(seq)
                best_len = len(seq)
            
            if len(seq) >= max_depth:
                return
            
            last = seq[-1] if seq else 0
            # Try values in order, but limit branching for speed
            candidates = range(last + 1, 257)
            for x in candidates:
                if time.time() - start_time > DEADLINE_MARGIN:
                    return
                
                if _valid_addition(seq, x, forbidden):
                    new_forbidden = set(forbidden)
                    _add_sums(seq, x, new_forbidden)
                    seq.append(x)
                    bt2(seq, new_forbidden, max_depth)
                    seq.pop()
        
        # Limit max depth to avoid excessive exploration
        max_depth = best_len + 5
        bt2(seq2, forbidden2, max_depth)
    
    return best
# EVOLVE-BLOCK-END
