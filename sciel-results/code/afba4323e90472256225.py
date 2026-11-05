import time
from math import isqrt
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    # Forward DP: for each prime i, maintain Pareto frontier of (gap, length)
    # gap = gap from previous prime to i, length = total length of subsequence ending at i
    # We process primes in order and propagate forward
    # dp[i] = list of (gap_from_prev, length) sorted by gap, pruned to Pareto optimal
    
    # Initialize: each prime can be a start of a sequence with gap=0, length=1
    dp = [[] for _ in range(n)]
    for i in range(n):
        dp[i] = [(0, 1)]
    
    # Process in order: for each prime j, look at all i < j and try to extend
    # If we have a subsequence ending at i with last gap g, and primes[j] - primes[i] >= g,
    # then we can extend to j with new gap = primes[j] - primes[i], length + 1
    for j in range(n):
        if time.time() > deadline:
            break
        pj = primes[j]
        new_states = []
        for i in range(j):
            pi = primes[i]
            gap_ij = pj - pi
            # For each state in dp[i], check if we can extend
            for (g_prev, L_prev) in dp[i]:
                if gap_ij >= g_prev:
                    new_states.append((gap_ij, L_prev + 1))
        
        # Also include the singleton
        new_states.append((0, 1))
        
        # Prune to Pareto frontier: sort by gap, keep only states where
        # length is strictly greater than all states with smaller gap
        new_states.sort(key=lambda x: x[0])
        pruned = []
        max_L = 0
        for (g, L) in new_states:
            if L > max_L:
                pruned.append((g, L))
                max_L = L
        dp[j] = pruned
    
    # Find the best sequence from DP
    best_len = 0
    best_i = -1
    for i in range(n):
        if dp[i]:
            L = dp[i][-1][1]
            if L > best_len:
                best_len = L
                best_i = i
    
    if best_i == -1:
        return [primes[0]] if n > 0 else []
    
    # Reconstruct a valid sequence from the DP result
    # We need to trace back. Store predecessor info.
    # Let's redo with predecessor tracking.
    
    # Redo DP with full predecessor tracking
    dp = [[] for _ in range(n)]
    for i in range(n):
        dp[i] = [(0, 1, -1)]  # (gap, length, prev_index)
    
    for j in range(n):
        if time.time() > deadline:
            break
        pj = primes[j]
        new_states = []
        for i in range(j):
            pi = primes[i]
            gap_ij = pj - pi
            for (g_prev, L_prev, _) in dp[i]:
                if gap_ij >= g_prev:
                    new_states.append((gap_ij, L_prev + 1, i))
        
        new_states.append((0, 1, -1))
        
        new_states.sort(key=lambda x: x[0])
        pruned = []
        max_L = 0
        for (g, L, prev) in new_states:
            if L > max_L:
                pruned.append((g, L, prev))
                max_L = L
        dp[j] = pruned
    
    best_len = 0
    best_i = -1
    for i in range(n):
        if dp[i]:
            L = dp[i][-1][1]
            if L > best_len:
                best_len = L
                best_i = i
    
    if best_i == -1:
        return [primes[0]] if n > 0 else []
    
    # Reconstruct sequence
    seq = []
    curr = best_i
    while curr != -1:
        seq.append(curr)
        state = dp[curr][-1]
        curr = state[2]
    seq.reverse()
    
    def seq_gaps(seq_idx):
        """Compute gaps for a sequence of prime indices."""
        if len(seq_idx) < 2:
            return []
        gaps = []
        for k in range(1, len(seq_idx)):
            gaps.append(primes[seq_idx[k]] - primes[seq_idx[k-1]])
        return gaps
    
    def is_valid(seq_idx):
        """Check if gaps are nondecreasing."""
        if len(seq_idx) < 2:
            return True
        gaps = seq_gaps(seq_idx)
        for k in range(1, len(gaps)):
            if gaps[k] < gaps[k-1]:
                return False
        return True
    
    def try_extend(seq_idx):
        """Try to extend the sequence by adding a prime at the end.
        Returns the new sequence if extension is possible, else None."""
        if not seq_idx:
            return None
        last_idx = seq_idx[-1]
        if last_idx + 1 >= n:
            return None
        last_prime = primes[last_idx]
        if len(seq_idx) >= 2:
            prev_prime = primes[seq_idx[-2]]
            last_gap = last_prime - prev_prime
        else:
            last_gap = 0
        
        # Find smallest prime q > last_prime such that q - last_prime >= last_gap
        target = last_prime + last_gap
        pos = bisect.bisect_left(primes, target)
        if pos < n:
            new_seq = seq_idx + [pos]
            if is_valid(new_seq):
                return new_seq
        return None
    
    # Local search: repeatedly try to extend, and if stuck, try to tighten the tail
    best_seq = list(seq)
    best_len = len(best_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(best_seq)
        
        # Try to extend
        extended = try_extend(best_seq)
        if extended is not None:
            best_seq = extended
            best_len = len(best_seq)
            improved = True
            continue
        
        # Cannot extend. Try to tighten the tail to allow extension.
        # Strategy: try to increase primes in the sequence from position L-2 down to 0.
        # Increasing a prime at position k reduces the gap from k to k+1,
        # which makes it easier to satisfy the nondecreasing constraint at k+1.
        
        found = False
        for k in range(L - 2, -1, -1):
            if k == 0:
                # Can increase the first prime to the largest prime < primes[best_seq[1]]
                next_prime = primes[best_seq[1]]
                # Largest prime < next_prime
                idx = bisect.bisect_left(primes, next_prime) - 1
                if idx > best_seq[0]:
                    new_seq = list(best_seq)
                    new_seq[0] = idx
                    if is_valid(new_seq):
                        best_seq = new_seq
                        found = True
                        break
            else:
                prev_prime = primes[best_seq[k-1]]
                next_prime = primes[best_seq[k+1]]
                curr_prime = primes[best_seq[k]]
                
                # Constraint: (p - prev) <= (next - p) => p <= (prev + next) / 2
                max_allowed = (prev_prime + next_prime) // 2
                
                # Find largest prime <= max_allowed and > curr_prime and < next_prime
                # Use bisect to find candidate
                idx = bisect.bisect_right(primes, max_allowed) - 1
                if idx > best_seq[k] and primes[idx] < next_prime and primes[idx] > prev_prime:
                    new_seq = list(best_seq)
                    new_seq[k] = idx
                    if is_valid(new_seq):
                        best_seq = new_seq
                        found = True
                        break
        
        if found:
            improved = True
            continue
        
        # If no improvement from tightening, try a more aggressive approach:
        # Re-derive the best sequence from DP for the suffix starting at various points.
        # Or: try to find a longer sequence by restarting local search from different initial states.
        
        # Alternative: try to insert a prime in the middle of the sequence.
        # For each pair of consecutive elements (a, b), try to insert a prime p between them
        # such that the gaps remain nondecreasing.
        # gap_left = a - prev (or 0 if a is first)
        # gap_mid = p - a
        # gap_right = b - p
        # Need: gap_left <= gap_mid <= gap_right
        # i.e., p - a >= gap_left and b - p >= p - a
        # p >= a + gap_left and p <= (a + b) / 2
        
        inserted = False
        for k in range(len(best_seq) - 1):
            a_idx = best_seq[k]
            b_idx = best_seq[k + 1]
            a = primes[a_idx]
            b = primes[b_idx]
            
            if k == 0:
                gap_left = 0
            else:
                gap_left = a - primes[best_seq[k - 1]]
            
            # Find p such that p >= a + gap_left and p <= (a + b) // 2
            # and p is prime, a < p < b
            lo = a + gap_left
            hi = (a + b) // 2
            
            if lo > hi:
                continue
            
            # Find primes in [lo, hi]
            start_pos = bisect.bisect_left(primes, lo)
            end_pos = bisect.bisect_right(primes, hi)
            
            # Try each candidate prime in this range (from largest to smallest to keep gaps balanced)
            for p_idx in range(end_pos - 1, start_pos - 1, -1):
                p = primes[p_idx]
                if p <= a or p >= b:
                    continue
                # Check full validity
                new_seq = list(best_seq)
                new_seq.insert(k + 1, p_idx)
                if is_valid(new_seq):
                    best_seq = new_seq
                    inserted = True
                    break
            if inserted:
                break
        
        if inserted:
            improved = True
            continue
        
        # If still no improvement, try a random restart with perturbation
        if time.time() < deadline - 5.0:
            # Try to find a different initial sequence by perturbing the current one
            # and running local search from there.
            import random
            random.seed(int(time.time() * 1000))
            for _ in range(50):
                if time.time() > deadline - 2.0:
                    break
                # Perturb: remove a random element (not first or last)
                if len(best_seq) > 3:
                    pos = random.randint(1, len(best_seq) - 2)
                    perturbed = best_seq[:pos] + best_seq[pos+1:]
                else:
                    perturbed = list(best_seq)
                
                # Run local search from perturbed
                local_seq = list(perturbed)
                local_improved = True
                while local_improved and time.time() < deadline - 1.0:
                    local_improved = False
                    ext = try_extend(local_seq)
                    if ext is not None:
                        local_seq = ext
                        local_improved = True
                        continue
                    # Tighten
                    for k in range(len(local_seq) - 2, -1, -1):
                        if k == 0:
                            next_prime = primes[local_seq[1]]
                            idx = bisect.bisect_left(primes, next_prime) - 1
                            if idx > local_seq[0]:
                                ns = list(local_seq)
                                ns[0] = idx
                                if is_valid(ns):
                                    local_seq = ns
                                    local_improved = True
                                    break
                        else:
                            prev_prime = primes[local_seq[k-1]]
                            next_prime = primes[local_seq[k+1]]
                            max_allowed = (prev_prime + next_prime) // 2
                            idx = bisect.bisect_right(primes, max_allowed) - 1
                            if idx > local_seq[k] and primes[idx] < next_prime and primes[idx] > prev_prime:
                                ns = list(local_seq)
                                ns[k] = idx
                                if is_valid(ns):
                                    local_seq = ns
                                    local_improved = True
                                    break
                    if local_improved:
                        continue
                    # Try insertion
                    for k in range(len(local_seq) - 1):
                        a_idx = local_seq[k]
                        b_idx = local_seq[k + 1]
                        a = primes[a_idx]
                        b = primes[b_idx]
                        if k == 0:
                            gap_left = 0
                        else:
                            gap_left = a - primes[local_seq[k - 1]]
                        lo = a + gap_left
                        hi = (a + b) // 2
                        if lo > hi:
                            continue
                        start_pos = bisect.bisect_left(primes, lo)
                        end_pos = bisect.bisect_right(primes, hi)
                        for p_idx in range(end_pos - 1, start_pos - 1, -1):
                            p = primes[p_idx]
                            if p <= a or p >= b:
                                continue
                            ns = list(local_seq)
                            ns.insert(k + 1, p_idx)
                            if is_valid(ns):
                                local_seq = ns
                                local_improved = True
                                break
                        if local_improved:
                            break
                
                if len(local_seq) > best_len:
                    best_seq = local_seq
                    best_len = len(best_seq)
                    improved = True
                    break
    
    return [primes[i] for i in best_seq]
