import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    candidates = list(range(2, 129))
    n = len(candidates)
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    
    # Precompute product sets for all possible consecutive blocks
    # We'll use a different approach: build a compatibility graph
    # Two numbers a, b can be adjacent if there exists a valid sequence containing ...a,b...
    # This is complex, so let's use a simpler approach:
    # Try to find the longest sequence using branch and bound with smart pruning
    
    def check_valid(seq):
        """Check if a sequence has all distinct block products"""
        prods = set()
        for i in range(len(seq)):
            p = 1
            for j in range(i, len(seq)):
                p *= seq[j]
                prods.add(p)
        return len(prods) == len(seq) * (len(seq) + 1) // 2
    
    def extend_sequence(seq, remaining, deadline):
        """Try to extend seq by adding elements from remaining"""
        if not remaining:
            return seq[:]
        
        # Try adding each remaining element at the end (since sequence must be increasing)
        last = seq[-1] if seq else 1
        for i, c in enumerate(remaining):
            if time.time() > deadline - 1:
                break
            if c <= last:
                continue
            
            new_seq = seq + [c]
            if check_valid(new_seq):
                new_remaining = remaining[:i] + remaining[i+1:]
                extended = extend_sequence(new_seq, new_remaining, deadline)
                if len(extended) > len(seq):
                    return extended
        
        return seq[:]
    
    # Strategy 1: Start with primes and try to insert composites
    best_seq = primes[:]
    best_len = len(best_seq)
    
    # Strategy 2: Greedy with backtracking
    def greedy_with_backtrack(start_seq, remaining, deadline):
        if not remaining:
            return start_seq[:]
        
        # Sort remaining by value for deterministic behavior
        remaining_sorted = sorted(remaining)
        
        for i, c in enumerate(remaining_sorted):
            if time.time() > deadline - 2:
                break
            
            # Try inserting c at each valid position
            inserted = False
            for pos in range(len(start_seq) + 1):
                if pos > 0 and start_seq[pos-1] >= c:
                    continue
                if pos < len(start_seq) and c >= start_seq[pos]:
                    continue
                
                new_seq = start_seq[:pos] + [c] + start_seq[pos:]
                if check_valid(new_seq):
                    new_remaining = remaining_sorted[:i] + remaining_sorted[i+1:]
                    result = greedy_with_backtrack(new_seq, new_remaining, deadline)
                    if len(result) > len(start_seq):
                        return result
                    inserted = True
                    break
            
            if inserted:
                continue
        
        return start_seq[:]
    
    # Run multiple greedy attempts with different starting points
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        
        # Start with a random subset of primes
        random.shuffle(primes)
        start_seq = primes[:min(5, len(primes))]
        start_seq.sort()
        
        if not check_valid(start_seq):
            continue
        
        used = set(start_seq)
        remaining = [x for x in candidates if x not in used]
        
        result = greedy_with_backtrack(start_seq, remaining, deadline)
        if len(result) > best_len:
            best_seq = result[:]
            best_len = len(best_seq)
    
    # Strategy 3: Try to insert composites one by one into the best sequence
    if time.time() < deadline - 3:
        current_seq = best_seq[:]
        used = set(current_seq)
        remaining = [x for x in candidates if x not in used]
        random.shuffle(remaining)
        
        progress = True
        while progress and time.time() < deadline - 2:
            progress = False
            for c in remaining:
                if time.time() > deadline - 1:
                    break
                for pos in range(len(current_seq) + 1):
                    if pos > 0 and current_seq[pos-1] >= c:
                        continue
                    if pos < len(current_seq) and c >= current_seq[pos]:
                        continue
                    
                    new_seq = current_seq[:pos] + [c] + current_seq[pos:]
                    if check_valid(new_seq):
                        current_seq = new_seq
                        used.add(c)
                        remaining.remove(c)
                        if len(current_seq) > best_len:
                            best_seq = current_seq[:]
                            best_len = len(current_seq)
                        progress = True
                        break
                if progress:
                    break
    
    return best_seq
