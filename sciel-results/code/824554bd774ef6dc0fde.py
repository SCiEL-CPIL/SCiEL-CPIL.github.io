"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

def _score_set_fft(U):
    """Compute score using FFT for sumset/diffset counting - fast for large sets."""
    U_set = sorted(set(U))
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Use FFT to compute sumset and diffset sizes
    # Sumset: U + U, range [0, 2*max_u]
    # Diffset: U - U, range [-max_u, max_u]
    
    # For sumset, use convolution
    size_sum = 2 * max_u + 1
    # Pad to power of 2 for FFT efficiency
    nfft = 1
    while nfft < 2 * size_sum:
        nfft *= 2
    
    # Create indicator array for U
    ind_sum = np.zeros(nfft, dtype=np.float64)
    for u in U_set:
        ind_sum[u] = 1.0
    
    # Convolve with itself to get sumset counts
    fft_sum = np.fft.fft(ind_sum)
    conv_sum = np.fft.ifft(fft_sum * fft_sum).real
    sumset_size = np.sum(conv_sum[:size_sum] > 0.5)
    
    # For diffset: U - U, range [-max_u, max_u]
    # Shift by max_u to make non-negative: (U + max_u) - (U + max_u) ... 
    # Actually diffset is {u1 - u2 : u1, u2 in U}
    # We can compute via convolution of U with reversed U
    # Or simply: diffset count at position d = sum of ind[u] * ind[u-d] for all u
    # Use FFT: create array where ind_shifted[i] = 1 if i in U
    # Then cross-correlation gives diffset
    
    ind_diff = np.zeros(nfft, dtype=np.float64)
    for u in U_set:
        ind_diff[u] = 1.0
    
    # Cross-correlation: for each d in [-max_u, max_u], count pairs (u, v) with u - v = d
    # This is convolution of ind_diff with reversed ind_diff
    ind_rev = np.zeros(nfft, dtype=np.float64)
    for u in U_set:
        ind_rev[nfft - 1 - u] = 1.0
    
    fft_diff = np.fft.fft(ind_diff)
    fft_rev = np.fft.fft(ind_rev)
    conv_diff = np.fft.ifft(fft_diff * fft_rev).real
    
    # The diffset values range from -max_u to max_u
    # In the convolution result, position (nfft - 1 - u) + u' corresponds to u' - u
    # Let me think more carefully...
    # conv_diff[k] = sum_j ind_diff[j] * ind_rev[k-j] = sum_j ind_diff[j] * ind_rev[k-j]
    # ind_rev[m] = 1 iff m = nfft - 1 - u for some u in U, i.e., u = nfft - 1 - m
    # So conv_diff[k] = sum_j ind_diff[j] * (1 iff nfft - 1 - (k-j) in U)
    # = sum_j ind_diff[j] * (1 iff j - k + nfft - 1 in U)
    # Let d = j - k, so this counts pairs where j in U and j - k + nfft - 1 in U
    # Hmm, this is getting complex. Let me just use the direct approach for diffset.
    
    # Simpler: just compute diffset directly for reasonable sizes
    U_arr = np.array(U_set, dtype=np.int64)
    diffset = set(np.unique(U_arr[:, None] - U_arr[None, :]).tolist())
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    U_list = sorted(U_set)
    U_arr = np.array(U_list, dtype=np.int64)
    
    sumset = set(np.unique(U_arr[:, None] + U_arr[None, :]).tolist())
    diffset = set(np.unique(U_arr[:, None] - U_arr[None, :]).tolist())
    
    if not sumset:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    if ratio == 1.0:
        return 1.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
        
    return 1.0 + log_ratio / log_denom

def _fast_score(U):
    """Faster scoring for local search - assumes U is a sorted list with no duplicates."""
    if not U or 0 not in U:
        return 0.0
    n = len(U)
    max_u = U[-1]
    if max_u == 0:
        return 1.0
    
    U_arr = np.array(U, dtype=np.int64)
    sumset = set(np.unique(U_arr[:, None] + U_arr[None, :]).tolist())
    diffset = set(np.unique(U_arr[:, None] - U_arr[None, :]).tolist())
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _modular_lattice_construction(m, density, deadline=None):
    """
    Construct U using a modular lattice approach.
    Select residues mod m that form a dense sublattice, then lift to integers.
    """
    U = [0]
    start_time = time.time()
    
    # Generate candidate residues
    # Try different patterns of residue selection
    best_U = [0]
    best_score = _fast_score(best_U)
    
    # Strategy: pick a subset S of Z_m such that |S+S| is small but |S-S| is large
    # Then U = {k*m + s : k in range, s in S} for some range
    
    for attempt in range(5):
        if deadline and time.time() > deadline:
            break
            
        # Randomly select residues
        num_residues = max(2, int(m * density))
        residues = [0] + random.sample(range(1, m), min(num_residues - 1, m - 1))
        residues = sorted(residues)
        
        # Build U by taking multiple copies shifted by m
        # U = {k*m + r : k in 0..K, r in residues}
        # But we need to keep max_u reasonable
        
        # Estimate optimal K based on m
        # If m is large, K should be small
        K = max(1, int(50 / max(1, m)))
        
        U_candidate = []
        for k in range(K + 1):
            for r in residues:
                val = k * m + r
                if val not in U_candidate:
                    U_candidate.append(val)
        
        U_candidate = sorted(U_candidate)
        
        if not U_candidate or 0 not in U_candidate:
            continue
            
        s = _fast_score(U_candidate)
        if s > best_score:
            best_score = s
            best_U = U_candidate[:]
            
    return best_U, best_score

def _modular_optimize(m_range, density_range, deadline=None):
    """Optimize over modulus m and density for modular lattice construction."""
    best_U = [0, 1, 3]
    best_score = _fast_score(best_U)
    
    for m in m_range:
        if deadline and time.time() > deadline:
            break
        for density in density_range:
            if deadline and time.time() > deadline:
                break
            U, s = _modular_lattice_construction(m, density, deadline)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    return best_U, best_score

def _local_search(U, max_iters=2000, deadline=None):
    """Improve set U via local search with add/remove/move operations."""
    U = sorted(set(U))
    best_score = _fast_score(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 50
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        
        # Try moving one element
        for i in range(len(best_U)):
            if improved:
                break
            old_val = best_U[i]
            if old_val == 0:
                continue
            search_range = list(range(max(1, old_val - 15), min(upper_bound, old_val + 15) + 1))
            random.shuffle(search_range)
            
            for new_val in search_range:
                if new_val == old_val:
                    continue
                new_U = best_U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _fast_score(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try adding a new element
        existing = set(best_U)
        candidates_add = [v for v in range(1, upper_bound + 1) if v not in existing]
        random.shuffle(candidates_add)
        
        for new_val in candidates_add:
            new_U = best_U[:] + [new_val]
            new_U.sort()
            s = _fast_score(new_U)
            if s > best_score + 1e-10:
                best_score = s
                best_U = new_U[:]
                improved = True
                break
        
        if improved:
            continue
            
        # Try removing an element (if size > 3)
        if len(best_U) > 3:
            for i in range(1, len(best_U)):
                new_U = best_U[:i] + best_U[i+1:]
                s = _fast_score(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if not improved:
            break
            
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_U = [0, 1, 3]
    best_score = _fast_score(best_U)
    
    # 1. Modular lattice construction - new structural approach
    if time.time() < deadline:
        # Try various moduli and densities
        m_candidates = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]
        density_candidates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        
        # Start with smaller moduli for speed
        for m in m_candidates[:8]:
            if time.time() > deadline - 30:
                break
            for density in density_candidates:
                if time.time() > deadline - 30:
                    break
                U_mod, s_mod = _modular_lattice_construction(m, density, deadline)
                if s_mod > best_score:
                    best_score = s_mod
                    best_U = U_mod[:]
        
    # 2. Local search on the best candidate
    if time.time() < deadline - 20:
        best_U, best_score = _local_search(best_U, max_iters=1000, deadline=deadline)
        
    # 3. Try more moduli if time remains
    if time.time() < deadline - 30:
        for m in m_candidates[8:]:
            if time.time() > deadline - 30:
                break
            for density in density_candidates[:3]:
                if time.time() > deadline - 30:
                    break
                U_mod, s_mod = _modular_lattice_construction(m, density, deadline)
                if s_mod > best_score:
                    best_score = s_mod
                    best_U = U_mod[:]
        
        # Local search again
        if time.time() < deadline - 10:
            best_U, best_score = _local_search(best_U, max_iters=500, deadline=deadline)
        
    # 4. Final polish
    if time.time() < deadline - 5:
        best_U, best_score = _local_search(best_U, max_iters=300, deadline=deadline)
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
