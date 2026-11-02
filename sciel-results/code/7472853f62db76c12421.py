import time
import numpy as np
from itertools import combinations
import sys

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def _atom_stats(D):
    """Compute (r, s, log_r, log_2s_plus1) for atom D."""
    D = sorted(D)
    sumset = set()
    diffset = set()
    for a in D:
        for b in D:
            sumset.add(a + b)
            diffset.add(a - b)
    r = len(diffset) / len(sumset)
    s = max(D)
    if r <= 1.0 or s == 0:
        return None
    return (r, s, np.log(r), np.log(2 * s + 1))

def _mixed_atom_stats(D1, D2):
    """Compute stats for mixed-radix product D1 x D2."""
    sumset = set()
    diffset = set()
    for a in D1:
        for b in D2:
            for c in D1:
                for d in D2:
                    sumset.add(a * len(D2) + b + c * len(D2) + d)
                    diffset.add(a * len(D2) + b - c * len(D2) - d)
    r = len(diffset) / len(sumset)
    s = max(D1) * len(D2) + max(D2)
    if r <= 1.0 or s == 0:
        return None
    return (r, s, np.log(r), np.log(2 * s + 1))

def _search_atoms(deadline):
    """Search for efficient atoms using exhaustive + random methods."""
    atoms = []
    
    # Exhaustive search for small atoms
    for size in range(2, 5):
        if size == 2:
            span_limit = 50
        elif size == 3:
            span_limit = 40
        elif size == 4:
            span_limit = 30
        else:
            break
        
        for comb in combinations(range(1, span_limit + 1), size - 1):
            D = [0] + list(comb)
            stats = _atom_stats(D)
            if stats:
                r, s, log_r, log_2sp1 = stats
                eff = log_r / log_2sp1
                if eff > 0:
                    atoms.append((eff, D, stats))
            
            if time.time() > deadline - 60:
                break
        if time.time() > deadline - 60:
            break
    
    # Random search for larger atoms
    np.random.seed(42)
    for size in range(2, 8):
        span_limit = max(20, 60 - size * 5)
        n_trials = 3000 if size <= 4 else 1000
        
        for _ in range(n_trials):
            if time.time() > deadline - 60:
                break
            D = [0] + sorted(np.random.choice(range(1, span_limit + 1), size=size-1, replace=False).tolist())
            stats = _atom_stats(D)
            if stats:
                r, s, log_r, log_2sp1 = stats
                eff = log_r / log_2sp1
                if eff > 0:
                    atoms.append((eff, D, stats))
    
    # Sort by efficiency and keep unique
    atoms.sort(key=lambda x: -x[0])
    unique_atoms = []
    seen = set()
    for eff, D, stats in atoms:
        key = tuple(D)
        if key not in seen:
            seen.add(key)
            unique_atoms.append((eff, D, stats))
            if len(unique_atoms) >= 50:
                break
    
    return unique_atoms

def _build_mixed_radix(atom_list, max_u_limit=10**7):
    """Build U using mixed-radix construction with given atom list."""
    U = [0]
    M = 1
    
    for D in atom_list:
        new_U = []
        for u in U:
            for d in D:
                new_U.append(u + d * M)
        U = new_U
        M = 2 * max(U) + 1
        
        if max(U) > max_u_limit:
            return U
    
    return U

def _analytic_score_mixed(atom_list):
    """Compute analytic score for mixed-radix construction."""
    total_log_r = 0
    max_u = 0
    M = 1
    
    for D in atom_list:
        D_sorted = sorted(D)
        sumset = set()
        diffset = set()
        for a in D_sorted:
            for b in D_sorted:
                sumset.add(a + b)
                diffset.add(a - b)
        r = len(diffset) / len(sumset)
        if r <= 1:
            return 0.0
        total_log_r += np.log(r)
        max_u += max(D_sorted) * M
        M = 2 * max_u + 1
    
    if max_u == 0:
        return 0.0
    
    denom = np.log(2 * max_u + 1)
    return 1.0 + total_log_r / denom

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Step 1: Search for good atoms
    atoms = _search_atoms(deadline)
    
    if not atoms:
        return [0, 1]
    
    # Step 2: Try different combinations of atoms
    best_U = [0, 1]
    best_score = 0.0
    
    # Try using just the best atom multiple times
    for num_copies in range(1, 15):
        if time.time() > deadline - 20:
            break
        
        atom_list = [atoms[0][1]] * num_copies
        score = _analytic_score_mixed(atom_list)
        
        if score > best_score:
            U = _build_mixed_radix(atom_list)
            if U and len(U) > 1:
                real_score = _score_set(U)
                if real_score > best_score:
                    best_score = real_score
                    best_U = U
    
    # Step 3: Try mixed atoms (different atoms at different levels)
    top_atoms = [a[1] for a in atoms[:10]]
    
    # Try various combinations
    for i in range(min(5, len(top_atoms))):
        for j in range(min(5, len(top_atoms))):
            if time.time() > deadline - 15:
                break
            
            for K in range(1, 10):
                # Use atom i for first K-1 levels, atom j for last level
                atom_list = [top_atoms[i]] * (K - 1) + [top_atoms[j]]
                score = _analytic_score_mixed(atom_list)
                
                if score > best_score:
                    U = _build_mixed_radix(atom_list)
                    if U and len(U) > 1:
                        real_score = _score_set(U)
                        if real_score > best_score:
                            best_score = real_score
                            best_U = U
    
    # Step 4: Local search on the last element of best_U
    if time.time() < deadline - 5:
        if best_U and len(best_U) > 2:
            # Try replacing last few elements
            base = best_U[:-1] if len(best_U) > 1 else best_U
            
            # Try adding different last elements
            max_base = max(base) if base else 0
            candidate_last = []
            
            # Try values around the current last element
            if best_U:
                current_last = best_U[-1]
                for delta in range(-5, 6):
                    candidate_last.append(current_last + delta)
            
            # Try some other candidates
            candidate_last.extend([max_base + 1, max_base + 2, max_base + 3, max_base + 5])
            candidate_last = [x for x in candidate_last if x >= 0]
            
            for last_val in candidate_last:
                if time.time() > deadline - 3:
                    break
                
                U_candidate = base + [last_val]
                U_candidate = sorted(set(U_candidate))
                
                if len(U_candidate) < 2:
                    continue
                
                score = _score_set(U_candidate)
                if score > best_score:
                    best_score = score
                    best_U = U_candidate
    
    # Final check: ensure 0 is in the set
    if 0 not in best_U:
        best_U = sorted([0] + best_U)
    
    return best_U

if __name__ == "__main__":
    s = construct()
    print(f"Result: Size={len(s)}, Score={_score_set(s):.6f}")
    print(f"First 20 elements: {s[:20]}")
