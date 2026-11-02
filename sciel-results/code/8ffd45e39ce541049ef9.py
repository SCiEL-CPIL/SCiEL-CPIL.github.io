"""Erdos minimum overlap - periodic binary sequence approach."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _max_overlap(h):
    """
    Compute the maximum over shifts s in [0, 2] of the overlap integral
    integral h(x) * (1 - h(x+s)) dx over the valid domain.
    
    For a step function with N steps on [0, 2], we consider shifts that are
    multiples of dw = 2/N. The overlap for shift k*dw is:
    sum_{i=0}^{N-k-1} h[i] * (1 - h[i+k]) * dw
    
    We take the maximum over all k = 0, 1, ..., N-1.
    """
    dw = 2.0 / N
    max_overlap = 0.0
    for k in range(N):
        valid = N - k
        if valid <= 0:
            continue
        overlap = np.sum(h[:valid] * (1.0 - h[k:])) * dw
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _max_overlap_fast(h):
    """Faster version using vectorized operations."""
    dw = 2.0 / N
    max_overlap = 0.0
    for k in range(N):
        valid = N - k
        if valid <= 0:
            continue
        overlap = np.dot(h[:valid], 1.0 - h[k:]) * dw
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _generate_m_sequence(period):
    """Generate an m-sequence (maximal length LFSR sequence) of given period."""
    # For period 2^n - 1, we need an appropriate feedback polynomial
    # Common m-sequences:
    # n=3: period 7, poly x^3 + x + 1
    # n=4: period 15, poly x^4 + x + 1
    # n=5: period 31, poly x^5 + x^2 + 1
    # n=6: period 63, poly x^6 + x + 1
    
    # For N=100, we can use period 63 (n=6) and pad/truncate
    # Or we can try different periods
    
    # Let's implement a generic LFSR
    # For simplicity, use known good m-sequences
    
    if period == 7:
        # x^3 + x + 1
        taps = [2, 0]  # 0-indexed
        reg = [1, 0, 0]
        seq = []
        for _ in range(period):
            seq.append(reg[0])
            feedback = reg[taps[0]] ^ reg[taps[1]]
            reg = [feedback] + reg[:-1]
        return seq
    
    elif period == 15:
        # x^4 + x + 1
        taps = [3, 0]
        reg = [1, 0, 0, 0]
        seq = []
        for _ in range(period):
            seq.append(reg[0])
            feedback = reg[taps[0]] ^ reg[taps[1]]
            reg = [feedback] + reg[:-1]
        return seq
    
    elif period == 31:
        # x^5 + x^2 + 1
        taps = [4, 2]
        reg = [1, 0, 0, 0, 0]
        seq = []
        for _ in range(period):
            seq.append(reg[0])
            feedback = reg[taps[0]] ^ reg[taps[1]]
            reg = [feedback] + reg[:-1]
        return seq
    
    elif period == 63:
        # x^6 + x + 1
        taps = [5, 0]
        reg = [1, 0, 0, 0, 0, 0]
        seq = []
        for _ in range(period):
            seq.append(reg[0])
            feedback = reg[taps[0]] ^ reg[taps[1]]
            reg = [feedback] + reg[:-1]
        return seq
    
    else:
        # Fallback: use period 63 and truncate
        return _generate_m_sequence(63)[:period]

def _balanced_binary_sequence(n, period):
    """
    Generate a balanced binary sequence of length n with period 'period'.
    The sequence is repeated and truncated to length n.
    We adjust to ensure exactly n/2 ones.
    """
    base = _generate_m_sequence(period)
    # Repeat to get length n
    seq = []
    while len(seq) < n:
        seq.extend(base)
    seq = seq[:n]
    
    # Count ones
    ones = sum(seq)
    target = n // 2
    
    if ones > target:
        # Remove excess ones by flipping some 1s to 0s
        # Flip positions that minimize autocorrelation impact
        excess = ones - target
        indices = [i for i in range(n) if seq[i] == 1]
        # Flip every other one starting from the end
        for i in range(excess):
            idx = indices[len(indices) - 1 - i]
            seq[idx] = 0
    elif ones < target:
        # Add needed ones by flipping some 0s to 1s
        deficit = target - ones
        indices = [i for i in range(n) if seq[i] == 0]
        for i in range(deficit):
            idx = indices[len(indices) - 1 - i]
            seq[idx] = 1
    
    return seq

def _local_search_binary(h, max_iters=5000, time_limit=30.0):
    """
    Local search on binary sequences: flip individual bits if it improves max overlap.
    """
    start_time = time.time()
    h = h.copy()
    current_score = _max_overlap_fast(h)
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            # Try flipping bit i
            old_val = h[i]
            h[i] = 1.0 - old_val
            
            # Check sum constraint
            new_sum = np.sum(h)
            if abs(new_sum - N/2.0) > 1e-12:
                h[i] = old_val
                continue
            
            new_score = _max_overlap_fast(h)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                h[i] = old_val
            iter_count += 1
    
    return h, current_score

def _local_search_binary_with_swap(h, max_iters=10000, time_limit=30.0):
    """
    Local search: swap pairs of bits (one 1 and one 0) to maintain sum constraint.
    """
    start_time = time.time()
    h = h.copy()
    current_score = _max_overlap_fast(h)
    
    ones = [i for i in range(N) if h[i] == 1.0]
    zeros = [i for i in range(N) if h[i] == 0.0]
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for oi in ones:
            if time.time() - start_time > time_limit:
                break
            for zi in zeros:
                if oi == zi:
                    continue
                # Swap: h[oi] = 0, h[zi] = 1
                h[oi] = 0.0
                h[zi] = 1.0
                
                new_score = _max_overlap_fast(h)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                    # Update lists
                    ones.remove(oi)
                    zeros.remove(zi)
                    ones.append(zi)
                    zeros.append(oi)
                    # Restart inner loop to find more improvements
                    break
                else:
                    h[oi] = 1.0
                    h[zi] = 0.0
                iter_count += 1
            if improved:
                break
    
    return h, current_score

def _simulated_annealing_binary(h, time_limit=40.0):
    """
    Simulated annealing on binary sequences.
    """
    start_time = time.time()
    h = h.copy()
    current_score = _max_overlap_fast(h)
    best_h = h.copy()
    best_score = current_score
    
    T = 0.1
    T_min = 0.001
    cooling = 0.995
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Generate neighbor: flip a random pair (1->0, 0->1)
        ones = [i for i in range(N) if h[i] == 1.0]
        zeros = [i for i in range(N) if h[i] == 0.0]
        
        if not ones or not zeros:
            break
        
        oi = np.random.choice(ones)
        zi = np.random.choice(zeros)
        
        if oi == zi:
            continue
        
        # Swap
        h[oi] = 0.0
        h[zi] = 1.0
        
        new_score = _max_overlap_fast(h)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Revert
            h[oi] = 1.0
            h[zi] = 0.0
        
        T *= cooling
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Strategy: Use periodic binary sequences with m-sequence structure
    # then refine with local search and simulated annealing
    
    best_h = None
    best_score = float('inf')
    
    # Try different m-sequence periods
    periods_to_try = [63, 31, 15, 7]
    
    for period in periods_to_try:
        if time.time() > deadline - 20.0:
            break
        
        # Generate balanced binary sequence
        h_bin = np.array(_balanced_binary_sequence(N, period), dtype=float)
        
        # Verify sum
        if abs(np.sum(h_bin) - N/2.0) > 1e-12:
            # Fix sum
            current_sum = np.sum(h_bin)
            diff = N/2.0 - current_sum
            if diff > 0:
                # Need to add ones
                zeros = [i for i in range(N) if h_bin[i] == 0.0]
                for i in range(int(diff)):
                    h_bin[zeros[i]] = 1.0
            else:
                # Need to remove ones
                ones = [i for i in range(N) if h_bin[i] == 1.0]
                for i in range(int(-diff)):
                    h_bin[ones[i]] = 0.0
        
        score = _max_overlap_fast(h_bin)
        
        if score < best_score:
            best_score = score
            best_h = h_bin.copy()
        
        # Local search refinement
        if time.time() < deadline - 15.0:
            h_ls, score_ls = _local_search_binary_with_swap(h_bin, max_iters=5000, time_limit=10.0)
            if score_ls < best_score:
                best_score = score_ls
                best_h = h_ls.copy()
        
        # Simulated annealing
        if time.time() < deadline - 10.0:
            h_sa, score_sa = _simulated_annealing_binary(h_ls, time_limit=20.0)
            if score_sa < best_score:
                best_score = score_sa
                best_h = h_sa.copy()
    
    # Also try random binary sequences with local search
    if time.time() < deadline - 15.0:
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            
            # Random balanced binary sequence
            h_rand = np.zeros(N)
            ones_positions = np.random.choice(N, size=N//2, replace=False)
            h_rand[ones_positions] = 1.0
            
            score_rand = _max_overlap_fast(h_rand)
            
            if score_rand < best_score:
                best_score = score_rand
                best_h = h_rand.copy()
            
            # Local search
            if time.time() < deadline - 10.0:
                h_ls, score_ls = _local_search_binary_with_swap(h_rand, max_iters=3000, time_limit=8.0)
                if score_ls < best_score:
                    best_score = score_ls
                    best_h = h_ls.copy()
    
    # Final local search on best solution
    if best_h is not None and time.time() < deadline - 5.0:
        h_final, score_final = _local_search_binary_with_swap(best_h, max_iters=5000, time_limit=10.0)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final
    
    if best_h is None:
        # Fallback: uniform 0.5
        best_h = np.full(N, 0.5)
    
    # Ensure sum is exactly N/2
    current_sum = np.sum(best_h)
    if abs(current_sum - N/2.0) > 1e-12:
        diff = N/2.0 - current_sum
        # Adjust by flipping bits if possible
        if abs(diff) < 1e-6:
            # Small numerical error, just adjust
            best_h = best_h + diff / N
            best_h = np.clip(best_h, 0.0, 1.0)
        else:
            # Need to flip bits
            if diff > 0:
                zeros = [i for i in range(N) if best_h[i] < 0.5]
                for i in range(int(np.ceil(diff))):
                    if i < len(zeros):
                        best_h[zeros[i]] = 1.0
            else:
                ones = [i for i in range(N) if best_h[i] > 0.5]
                for i in range(int(np.ceil(-diff))):
                    if i < len(ones):
                        best_h[ones[i]] = 0.0
    
    # Final sum check
    current_sum = np.sum(best_h)
    if abs(current_sum - N/2.0) > 1e-9:
        # Last resort: distribute error
        diff = N/2.0 - current_sum
        # Find elements closest to 0.5 and adjust
        dists = np.abs(best_h - 0.5)
        idx = np.argmin(dists)
        best_h[idx] += diff
        best_h[idx] = np.clip(best_h[idx], 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
