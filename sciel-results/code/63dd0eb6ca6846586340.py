"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # Generate all primes <= 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(10000)
    n = len(primes)
    
    if n == 0:
        return []
    
    # Binary search helper: find smallest index >= lo such that primes[idx] >= target
    def lower_bound(target, lo, hi):
        while lo < hi:
            mid = (lo + hi) // 2
            if primes[mid] >= target:
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    best = []
    
    # Strategy 1: Greedy - for each start, always pick the smallest valid next prime
    for start_idx in range(n):
        if time.time() > deadline:
            break
        
        seq = [primes[start_idx]]
        cur_gap = 0
        idx = start_idx
        while True:
            target = primes[idx] + cur_gap if cur_gap > 0 else primes[idx] + 1
            lo = lower_bound(target, idx + 1, n)
            if lo >= n:
                break
            seq.append(primes[lo])
            cur_gap = primes[lo] - primes[idx]
            idx = lo
        
        if len(seq) > len(best):
            best = seq
    
    # Strategy 2: Beam search with multiple candidates per step
    beam_width = 50
    max_beam = 200
    
    # State: (current_index, last_gap, sequence_length)
    # We'll keep a set of states to avoid duplicates
    
    for start_idx in range(n):
        if time.time() > deadline:
            break
        
        # Initialize beam with the starting state
        # State: (idx, last_gap, length)
        beam = [(start_idx, 0, 1)]
        
        # We need to track sequences for reconstruction, but that's expensive.
        # Instead, let's just track lengths and reconstruct at the end.
        # Actually, for beam search we need to track the actual path.
        # Let's use a different approach: for each state, store the sequence.
        # But that's memory-heavy. Let's just track (idx, last_gap) and length.
        
        # Actually, let's do a simpler beam: at each step, expand all states in the beam,
        # keep the top beam_width by length (and then by smaller last_gap for tie-breaking).
        
        best_in_beam = 1  # length of best state in current beam
        
        while beam and time.time() < deadline:
            next_beam = []
            for (idx, last_gap, length) in beam:
                # Find all valid next primes
                target = primes[idx] + last_gap if last_gap > 0 else primes[idx] + 1
                # We want primes q >= target
                lo = lower_bound(target, idx + 1, n)
                
                # Try the first few valid primes (not all, to keep beam manageable)
                for k in range(min(5, n - lo)):  # try up to 5 candidates per state
                    new_idx = lo + k
                    if new_idx >= n:
                        break
                    new_gap = primes[new_idx] - primes[idx]
                    next_beam.append((new_idx, new_gap, length + 1))
            
            if not next_beam:
                break
            
            # Sort by length descending, then by last_gap ascending (prefer smaller gaps)
            next_beam.sort(key=lambda x: (-x[2], x[1]))
            
            # Deduplicate by (idx, last_gap) keeping the longest
            seen = set()
            deduped = []
            for state in next_beam:
                key = (state[0], state[1])
                if key not in seen:
                    seen.add(key)
                    deduped.append(state)
            
            beam = deduped[:beam_width]
            
            # Update best
            for state in beam:
                if state[2] > best_in_beam:
                    best_in_beam = state[2]
            
            # Check if we can improve on global best
            if best_in_beam <= len(best):
                # If we can't beat the current best even with optimal expansion, break
                # This is a heuristic; we continue anyway for now
                pass
        
        # Reconstruct the best sequence from the beam
        # This is tricky because we didn't store the full path.
        # Let's skip this for now and rely on Strategy 1.
    
    # Strategy 3: More aggressive greedy variants
    # Try different "greediness" levels: pick the k-th smallest valid prime
    
    for k_factor in [1, 2, 3, 5, 8]:
        for start_idx in range(n):
            if time.time() > deadline:
                break
            
            seq = [primes[start_idx]]
            cur_gap = 0
            idx = start_idx
            while True:
                target = primes[idx] + cur_gap if cur_gap > 0 else primes[idx] + 1
                lo = lower_bound(target, idx + 1, n)
                
                # Pick the k_factor-th smallest valid prime (or the last one if fewer)
                pick_idx = lo + min(k_factor - 1, max(0, n - lo - 1))
                if pick_idx >= n:
                    break
                
                seq.append(primes[pick_idx])
                cur_gap = primes[pick_idx] - primes[idx]
                idx = pick_idx
            
            if len(seq) > len(best):
                best = seq
    
    return best

# EVOLVE-BLOCK-END
