"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit (or 60s if stricter)
    
    # 1. Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n_primes = len(primes)
    prime_set = set(primes)
    
    # Helper to check if a sequence is valid (non-decreasing gaps)
    def is_valid(seq):
        if len(seq) < 2:
            return True
        prev_gap = 0
        for i in range(1, len(seq)):
            gap = seq[i] - seq[i-1]
            if gap < prev_gap:
                return False
            prev_gap = gap
        return True
        
    # Helper to get gaps
    def get_gaps(seq):
        return [seq[i] - seq[i-1] for i in range(1, len(seq))]
        
    # Strategy:
    # 1. Find a good initial seed using a greedy "maximize length" heuristic.
    #    We try to build a sequence by always picking the next prime that allows
    #    the smallest possible next gap (to keep options open) or just random.
    #    Actually, a better seed: Start with small primes, try to extend.
    
    best_seq = []
    
    # Heuristic 1: Greedy construction with multiple starts
    # Try starting with each of the first 50 primes, then greedily pick the next prime
    # that minimizes the new gap (while keeping it >= prev gap) to stay "tight".
    # Or, pick the next prime that maximizes the potential for future extensions?
    # Let's try a few random greedy strategies to get a decent initial length.
    
    def greedy_build(start_idx, strategy="min_gap"):
        seq = [primes[start_idx]]
        prev_gap = 0
        last_idx = start_idx
        while True:
            found = False
            # Look for next prime
            # To be efficient, we can just iterate through primes from last_idx+1
            # But we need to find the best candidate.
            # For "min_gap": find smallest p > primes[last_idx] such that p - primes[last_idx] >= prev_gap
            # For "max_gap": find largest p <= 10000 ... no, we want to extend length, so we want small gaps.
            # Actually, if we pick a very large gap, we can't pick many more.
            # So "min_gap" strategy is good for length.
            
            min_candidate = None
            # We can binary search for the first prime >= primes[last_idx] + prev_gap
            # Since primes are sorted, we can just scan or use bisect.
            # Let's use bisect for speed.
            import bisect
            target = primes[last_idx] + prev_gap
            pos = bisect.bisect_left(primes, target, last_idx + 1)
            if pos < n_primes:
                candidate = primes[pos]
                new_gap = candidate - primes[last_idx]
                # This is the smallest valid gap.
                seq.append(candidate)
                prev_gap = new_gap
                last_idx = pos
                found = True
            if not found:
                break
        return seq

    # Try multiple greedy starts
    for i in range(min(20, n_primes)):
        if time.time() > deadline:
            break
        seq = greedy_build(i)
        if len(seq) > len(best_seq):
            best_seq = seq
            
    # Heuristic 2: Randomized Local Search
    # Start from best_seq and try to improve.
    # Moves:
    # 1. Insert a prime at any position.
    # 2. Replace a prime with another prime.
    # 3. Swap two primes (if order allows, but they must remain increasing, so swap is hard unless we re-sort, but re-sorting breaks the specific subsequence property? No, the sequence must be increasing. So we can't just swap arbitrary elements. We can replace an element with a different prime and re-validate).
    
    current_seq = best_seq[:]
    current_len = len(current_seq)
    
    # Pre-compute gaps for current_seq
    def calculate_gaps(seq):
        return [seq[i] - seq[i-1] for i in range(1, len(seq))]
        
    current_gaps = calculate_gaps(current_seq)
    
    # To speed up validation, we can check locally.
    # But for simplicity, is_valid is O(L). L is small (~10-20).
    
    def try_insert(seq, new_prime):
        # Try inserting new_prime at every position
        # Must maintain increasing order
        # So new_prime must be > seq[i-1] and < seq[i] (or at ends)
        # Let's find valid insertion points based on value.
        # Then check gap constraints.
        
        # Binary search for position where new_prime fits value-wise
        import bisect
        pos = bisect.bisect_left(seq, new_prime)
        
        # Check pos and pos-1 (if valid indices)
        candidates = []
        if pos < len(seq) and seq[pos] != new_prime:
            candidates.append(pos)
        if pos > 0 and seq[pos-1] != new_prime:
            candidates.append(pos-1)
            
        # Also, if new_prime is equal to an existing prime, we can't have duplicates (primes are unique).
        if new_prime in seq:
            return None
            
        best_candidate = None
        best_candidate_len = len(seq)
        
        for p in candidates:
            new_seq = seq[:p] + [new_prime] + seq[p:]
            # Check validity
            if is_valid(new_seq):
                # We prefer insertions that keep gaps small? Or just any valid?
                # For local search, we just want to increase length.
                # All insertions increase length by 1.
                # We should probably return the first valid one or a random one.
                # Let's return the one that minimizes the "tightness" of the constraint?
                # Or just the first one.
                return new_seq
        return None
        
    def try_replace(seq, index, new_prime):
        # Replace seq[index] with new_prime
        if new_prime in seq:
            return None
        # Check increasing order
        if index > 0 and new_prime <= seq[index-1]:
            return None
        if index < len(seq)-1 and new_prime >= seq[index+1]:
            return None
            
        new_seq = seq[:]
        new_seq[index] = new_prime
        if is_valid(new_seq):
            return new_seq
        return None

    # Local Search Loop
    no_improvement_count = 0
    max_no_improvement = 1000
    
    while time.time() < deadline and no_improvement_count < max_no_improvement:
        # Pick a random move
        move_type = random.choice(['insert', 'replace', 'insert', 'replace']) # Bias towards insert
        
        if move_type == 'insert':
            # Pick a random prime not in current_seq
            # To be efficient, pick a random index in primes and check
            # Or maintain a list of available primes?
            # For now, random sampling.
            # We want primes that are "close" to existing ones to have higher chance of fitting gaps.
            # Let's pick a prime from the full list.
            p_idx = random.randint(0, n_primes - 1)
            p = primes[p_idx]
            if p in current_seq:
                continue
                
            new_seq = try_insert(current_seq, p)
            if new_seq:
                current_seq = new_seq
                current_len = len(current_seq)
                current_gaps = calculate_gaps(current_seq)
                no_improvement_count = 0
                if current_len > len(best_seq):
                    best_seq = current_seq[:]
            else:
                no_improvement_count += 1
                
        elif move_type == 'replace':
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            p_idx = random.randint(0, n_primes - 1)
            p = primes[p_idx]
            if p in current_seq:
                continue
            new_seq = try_replace(current_seq, idx, p)
            if new_seq:
                # Replacement doesn't change length, but might allow future insertions.
                # Accept with some probability or if it improves "slack"?
                # For now, accept if valid.
                current_seq = new_seq
                current_gaps = calculate_gaps(current_seq)
                # Do not reset no_improvement_count for replacement?
                # Or reset it? If we don't, we might get stuck in a loop of replacements.
                # Let's not reset, but allow some replacements.
            else:
                no_improvement_count += 1

    # Final check: ensure best_seq is valid
    if not is_valid(best_seq):
        # Fallback to a simple valid sequence
        return [2, 3]
        
    return best_seq
# EVOLVE-BLOCK-END
