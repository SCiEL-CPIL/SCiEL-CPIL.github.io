import math
import time
import numpy as np

def _compute_score_fft(u):
    """Compute score using FFT for O(M log M) evaluation."""
    u = sorted(set(u))
    if not u or u[0] != 0:
        u = [0] + list(u)
        u = sorted(set(u))
    
    max_u = max(u)
    m = 2 * max_u + 1
    
    if m <= 1:
        return 1.0, u
    
    # Create indicator vector
    indicator = np.zeros(2 * max_u + 1, dtype=np.float64)
    for x in u:
        indicator[x] = 1.0
    
    # Sumset: convolution of indicator with itself
    # Result has length 2*(2*max_u+1)-1 = 4*max_u+1, but we only care about [0, 2*max_u]
    fft_size = 1
    while fft_size < 2 * (2 * max_u + 1):
        fft_size *= 2
    
    fft_ind = np.fft.fft(indicator, n=fft_size)
    fft_sum = fft_ind * fft_ind
    sumset_counts = np.fft.ifft(fft_sum).real
    
    # Sumset size: count positions where count > 0.5 (in range [0, 2*max_u])
    sumset_size = int(np.sum(sumset_counts[:2 * max_u + 1] > 0.5))
    
    # Diffset: correlation of indicator with itself
    # diff = a - b, range [-max_u, max_u]
    # Use FFT: correlation via fft(ind) * conj(fft(ind))
    fft_corr = fft_ind * np.conj(fft_ind)
    diffset_counts = np.fft.ifft(fft_corr).real
    
    # Diffset: positions where count > 0.5, mapped to [-max_u, max_u]
    # The ifft gives values at positions 0, 1, ..., fft_size-1
    # Position k corresponds to lag k if k <= fft_size//2, or k - fft_size if k > fft_size//2
    # We need lags in [-max_u, max_u]
    diffset_size = 0
    for lag in range(-max_u, max_u + 1):
        pos = lag % fft_size
        if diffset_counts[pos] > 0.5:
            diffset_size += 1
    
    r = diffset_size / sumset_size if sumset_size > 0 else 0.0
    score = 1 + math.log(r) / math.log(m)
    
    return score, u

def _local_search(u, max_steps=5000, deadline=None):
    """Local search to improve the set by adding/removing/moving elements."""
    u = sorted(set(u))
    if not u or u[0] != 0:
        u = [0] + list(u)
        u = sorted(set(u))
    
    current_score, _ = _compute_score_fft(u)
    best_score = current_score
    best_u = list(u)
    
    max_u = max(u)
    
    # Try various mutations
    step = 0
    while step < max_steps:
        if deadline is not None and time.time() > deadline:
            break
        
        mutation_type = step % 3
        
        if mutation_type == 0:
            # Try adding a new element
            # Candidate positions: just above max_u, or in gaps
            candidates = set()
            # Add near max
            candidates.add(max_u + 1)
            candidates.add(max_u + 2)
            candidates.add(max_u + 3)
            # Add in gaps
            for i in range(len(u) - 1):
                gap_start = u[i] + 1
                gap_end = u[i + 1] - 1
                if gap_end >= gap_start:
                    mid = (gap_start + gap_end) // 2
                    candidates.add(mid)
                    candidates.add(gap_start)
                    candidates.add(gap_end)
            
            if not candidates:
                step += 1
                continue
            
            best_candidate = None
            best_candidate_score = current_score
            
            for c in sorted(candidates):
                if c in u:
                    continue
                if c < 0:
                    continue
                new_u = sorted(u + [c])
                s, _ = _compute_score_fft(new_u)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = new_u
            
            if best_candidate is not None:
                u = best_candidate
                current_score = best_candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_u = list(u)
                max_u = max(u)
            
        elif mutation_type == 1:
            # Try removing an element (except 0)
            for i in range(1, len(u)):
                if u[i] != 0:
                    new_u = [x for j, x in enumerate(u) if j != i]
                    s, _ = _compute_score_fft(new_u)
                    if s > current_score:
                        u = new_u
                        current_score = s
                        if current_score > best_score:
                            best_score = current_score
                            best_u = list(u)
                        max_u = max(u)
                        break
        
        else:
            # Try moving an element
            for i in range(1, len(u)):
                if u[i] != 0:
                    old_val = u[i]
                    # Try moving by small amounts
                    for delta in [-3, -2, -1, 1, 2, 3]:
                        new_val = old_val + delta
                        if new_val <= 0:
                            continue
                        # Check if new_val is already in u (excluding position i)
                        if any(u[j] == new_val for j in range(len(u)) if j != i):
                            continue
                        new_u = list(u)
                        new_u[i] = new_val
                        new_u = sorted(set(new_u))
                        if 0 not in new_u:
                            new_u = [0] + new_u
                            new_u = sorted(set(new_u))
                        s, _ = _compute_score_fft(new_u)
                        if s > current_score:
                            u = new_u
                            current_score = s
                            if current_score > best_score:
                                best_score = current_score
                                best_u = list(u)
                            max_u = max(u)
                            break
        
        step += 1
    
    return best_u, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with known good Golomb rulers
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
    ]
    
    # Also try some non-Golomb structures that might have better ratio
    # Sets with more uniform spacing might have better |U-U|/|U+U| ratio
    for n in [10, 15, 20, 30, 50, 80, 120, 200]:
        # Arithmetic progression-like but with some perturbation
        base = list(range(0, n * 3, 3))
        if 0 not in base:
            base = [0] + base
        initial_candidates.append(sorted(set(base)))
        
        # Random-like set with good coverage
        import random
        random.seed(42)
        random_set = {0}
        for _ in range(n):
            random_set.add(random.randint(1, n * 4))
        initial_candidates.append(sorted(random_set))
    
    best_set = [0, 1, 3]
    best_score = 0.0
    
    # First pass: evaluate all initial candidates
    for u in initial_candidates:
        if time.time() > deadline:
            break
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
            u = sorted(set(u))
        score, _ = _compute_score_fft(u)
        if score > best_score:
            best_score = score
            best_set = list(u)
    
    # Second pass: local search from the best candidate
    if time.time() < deadline:
        improved_set, improved_score = _local_search(best_set, max_steps=10000, deadline=deadline)
        if improved_score > best_score:
            best_score = improved_score
            best_set = improved_set
    
    # Third pass: try local search from a few other good candidates
    if time.time() < deadline:
        for u in initial_candidates[:5]:
            if time.time() > deadline:
                break
            u = sorted(set(u))
            if 0 not in u:
                u = [0] + u
                u = sorted(set(u))
            s, _ = _compute_score_fft(u)
            if s > best_score * 0.95:  # Only if reasonably good
                improved_set, improved_score = _local_search(u, max_steps=3000, deadline=deadline)
                if improved_score > best_score:
                    best_score = improved_score
                    best_set = improved_set
    
    # Fourth pass: try building larger sets from scratch
    if time.time() < deadline:
        # Start with a small set and grow it
        current = [0, 1, 3]
        for _ in range(50):
            if time.time() > deadline:
                break
            max_u = max(current)
            # Try adding elements at various positions
            candidates = set()
            candidates.add(max_u + 1)
            candidates.add(max_u + 2)
            candidates.add(max_u + 3)
            candidates.add(max_u + 4)
            # Try filling gaps
            for i in range(len(current) - 1):
                gap_start = current[i] + 1
                gap_end = current[i + 1] - 1
                if gap_end >= gap_start:
                    candidates.add((gap_start + gap_end) // 2)
            
            best_add = None
            best_add_score = 0.0
            current_score, _ = _compute_score_fft(current)
            
            for c in sorted(candidates):
                if c in current or c <= 0:
                    continue
                new_u = sorted(current + [c])
                s, _ = _compute_score_fft(new_u)
                if s > best_add_score:
                    best_add_score = s
                    best_add = new_u
            
            if best_add is not None and best_add_score > current_score:
                current = best_add
                if best_add_score > best_score:
                    best_score = best_add_score
                    best_set = list(current)
            else:
                break
        
        # Local search on the grown set
        if time.time() < deadline:
            improved_set, improved_score = _local_search(current, max_steps=5000, deadline=deadline)
            if improved_score > best_score:
                best_score = improved_score
                best_set = improved_set
    
    return best_set
