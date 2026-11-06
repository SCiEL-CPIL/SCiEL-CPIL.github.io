import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85
    rng = random.Random(42)
    
    def compute_m_vec(A_set):
        """Vectorized sum-multiplicity computation via numpy convolution."""
        if not A_set:
            return np.zeros(257, dtype=np.int32)
        ind = np.zeros(129, dtype=np.float64)
        ind[list(A_set)] = 1.0
        conv = np.convolve(ind, ind, mode='full')  # length 257
        # For unordered pairs with equal summands:
        # m[s] = floor((conv[s] + self_pair[s]) / 2)
        # where self_pair[s] = 1 if s even and s/2 in A
        target = np.arange(128, 257)
        c = conv[128:257].copy()
        even_mask = (target % 2 == 0)
        halves = target // 2
        # halves ranges from 64 to 128
        self_pairs = np.zeros(129, dtype=np.float64)
        self_pairs[list(A_set)] = 1.0
        c[even_mask] += self_pairs[halves[even_mask]]
        m = (c + 0.5).astype(np.int32)  # ceil(c/2) for positive c
        # Actually: m[s] = (conv[s] + sp[s]) // 2
        m = ((c) // 2).astype(np.int32)
        return m
    
    def evaluate_fast(A_set):
        """Return (peak, valid) using vectorized computation."""
        if not A_set:
            return float('inf'), False
        m = compute_m_vec(A_set)
        target_m = m[128:257]
        if np.any(target_m == 0):
            return float('inf'), False
        peak = int(np.max(target_m))
        return peak, True
    
    def local_search(A_set, initial_peak, end_time):
        A = set(A_set)
        best_peak = initial_peak
        current_peak = initial_peak
        stall_count = 0
        
        while time.time() < end_time:
            improved = False
            m_cur = compute_m_vec(A)
            target_m = m_cur[128:257]
            
            # Identify sums at peak
            peak_sums = set(np.where(target_m == current_peak)[0] + 128)
            # Identify sums at low multiplicity (1 or 2)
            low_sums = set(np.where(target_m <= 2)[0] + 128)
            
            # Try targeted moves
            for attempt in range(50):
                if time.time() > end_time:
                    break
                op = rng.choice(['remove', 'add', 'swap', 'swap', 'swap'])
                candidate = A.copy()
                
                if op == 'remove' and len(A) > 2:
                    # Prefer removing elements that contribute to peak sums
                    peak_elements = set()
                    for s in peak_sums:
                        for a in A:
                            b = s - a
                            if a <= b and b in A:
                                peak_elements.add(a)
                                peak_elements.add(b)
                    if peak_elements:
                        el = rng.choice(list(peak_elements))
                    else:
                        el = rng.choice(list(A))
                    candidate.discard(el)
                elif op == 'add':
                    available = [i for i in range(129) if i not in A]
                    if not available:
                        continue
                    # Prefer elements that help cover low-multiplicity sums
                    helpful = []
                    for x in available:
                        for s in low_sums:
                            other = s - x
                            if 0 <= other <= 128 and other in A and x <= other:
                                helpful.append(x)
                                break
                    if helpful:
                        new_el = rng.choice(helpful)
                    else:
                        new_el = rng.choice(available)
                    candidate.add(new_el)
                elif op == 'swap' and len(A) > 2:
                    # Remove from peak, add to help low
                    peak_elements = set()
                    for s in peak_sums:
                        for a in A:
                            b = s - a
                            if a <= b and b in A:
                                peak_elements.add(a)
                                peak_elements.add(b)
                    if peak_elements:
                        el = rng.choice(list(peak_elements))
                    else:
                        el = rng.choice(list(A))
                    candidate.discard(el)
                    avail2 = [i for i in range(129) if i not in candidate]
                    if not avail2:
                        continue
                    # Prefer elements helping low sums
                    helpful = []
                    for x in avail2:
                        for s in low_sums:
                            other = s - x
                            if 0 <= other <= 128 and other in candidate and x <= other:
                                helpful.append(x)
                                break
                    if helpful:
                        new_el = rng.choice(helpful)
                    else:
                        new_el = rng.choice(avail2)
                    candidate.add(new_el)
                else:
                    continue
                
                peak_c, valid_c = evaluate_fast(candidate)
                if valid_c and peak_c < current_peak:
                    A = candidate
                    current_peak = peak_c
                    if peak_c < best_peak:
                        best_peak = peak_c
                    improved = True
                    stall_count = 0
                    break
            
            if not improved:
                stall_count += 1
                # Multi-move perturbation
                candidate = A.copy()
                k_remove = rng.randint(1, 3)
                k_add = rng.randint(1, 3)
                
                if len(candidate) > k_remove + 2:
                    to_remove = rng.sample(list(candidate), k_remove)
                    for el in to_remove:
                        candidate.discard(el)
                
                available = [i for i in range(129) if i not in candidate]
                if available:
                    k_add_actual = min(k_add, len(available))
                    to_add = rng.sample(available, k_add_actual)
                    for el in to_add:
                        candidate.add(el)
                
                peak_c, valid_c = evaluate_fast(candidate)
                if valid_c and peak_c < current_peak:
                    A = candidate
                    current_peak = peak_c
                    if peak_c < best_peak:
                        best_peak = peak_c
                    improved = True
                    stall_count = 0
                
                if not improved and stall_count > 8:
                    break
        
        return A, best_peak
    
    best_A = None
    best_peak = float('inf')
    
    # Crafted initializations targeting balanced coverage
    inits = [
        set(range(64, 129)),
        set(range(0, 32)) | set(range(96, 129)),
        set(range(0, 24)) | set(range(80, 129)),
        set(range(16, 48)) | set(range(80, 112)),
        set(range(0, 16)) | set(range(32, 48)) | set(range(96, 129)),
        set(range(0, 16)) | set(range(48, 64)) | set(range(96, 129)),
        set(range(32, 64)) | set(range(64, 96)),
        set(range(0, 20)) | set(range(60, 80)) | set(range(100, 129)),
        set(range(0, 12)) | set(range(24, 36)) | set(range(48, 60)) | set(range(96, 129)),
        set(range(0, 12)) | set(range(36, 48)) | set(range(72, 84)) | set(range(108, 129)),
        # New structured inits
        set(range(0, 21)) | set(range(42, 63)) | set(range(84, 105)) | set(range(105, 129)),
        set(range(0, 18)) | set(range(36, 54)) | set(range(72, 90)) | set(range(108, 129)),
        set(range(0, 15)) | set(range(30, 45)) | set(range(60, 75)) | set(range(90, 105)) | set(range(105, 129)),
    ]
    
    # Add random initializations with varied sizes
    for k in [28, 35, 42, 50, 56, 64]:
        inits.append(set(rng.sample(range(129), k)))
    
    for init_A in inits:
        if time.time() > deadline - 5:
            break
        peak, valid = evaluate_fast(init_A)
        if not valid:
            continue
        if peak < best_peak:
            best_peak = peak
            best_A = set(init_A)
        
        # Local search with time limit
        remaining = deadline - 3 - time.time()
        if remaining > 2:
            alloc = min(remaining, 8)
            A_local, peak_local = local_search(init_A, peak, time.time() + alloc)
            if peak_local < best_peak:
                best_peak = peak_local
                best_A = set(A_local)
    
    # Final polish if time permits
    if best_A is not None and time.time() < deadline - 4:
        remaining = deadline - 4 - time.time()
        if remaining > 2:
            A_final, peak_final = local_search(best_A, best_peak, time.time() + remaining)
            if peak_final < best_peak:
                best_peak = peak_final
                best_A = set(A_final)
    
    if best_A is None:
        return list(range(64, 129))
    
    return list(best_A)
