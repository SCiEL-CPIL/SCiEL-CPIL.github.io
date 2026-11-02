import time
import numpy as np

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
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

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    def compute_score_from_gaps(gaps):
        """Compute score for a set defined by gaps (gaps[i] = x_{i+1} - x_i)."""
        if not gaps:
            return 0.0
        U = [0]
        for g in gaps:
            if g <= 0:
                return 0.0
            U.append(U[-1] + g)
        U = np.array(U, dtype=np.int64)
        n = len(U)
        if n < 2:
            return 0.0
        mx = U[-1]
        if mx == 0:
            return 0.0
        
        # Use sets for speed if mx is small
        if mx < 5000:
            sumset = set()
            diffset = set()
            for i in range(n):
                for j in range(n):
                    sumset.add(int(U[i]) + int(U[j]))
                    diffset.add(int(U[i]) - int(U[j]))
            s_size = len(sumset)
            d_size = len(diffset)
            if s_size == 0:
                return 0.0
            ratio = d_size / s_size
            if ratio <= 1:
                return 0.0
            return 1.0 + np.log(ratio) / np.log(2 * mx + 1)
        else:
            return _score_set(U)

    # Modular lattice strategy:
    # Pick a modulus m, choose a subset R of residues mod m.
    # The set U is formed by taking multiple "layers" scaled by m:
    #   U = { r + k * m : r in R, k = 0, 1, ..., K-1 }  (with some structure)
    # This creates a lattice-like structure where sums wrap around modulo m,
    # potentially collapsing |U+U| while keeping |U-U| large.
    
    best_score = 0.0
    best_gaps = []
    
    np.random.seed(12345)
    
    # Try various modular lattice configurations
    def try_modular_lattice(m, residues, K):
        """
        Build U from residues mod m with K layers.
        U = { r + k*m for r in residues, k in range(K) }
        But we need U to be a set of nonnegative integers containing 0.
        If 0 is not in residues, add it.
        """
        elements = set()
        for r in residues:
            for k in range(K):
                elements.add(r + k * m)
        elements.add(0)
        if 0 not in elements:
            elements.add(0)
        U = sorted(elements)
        # Convert to gaps
        if not U:
            return None, 0.0
        gaps = [U[i+1] - U[i] for i in range(len(U)-1)]
        s = compute_score_from_gaps(gaps)
        return gaps, s
    
    # Explore different moduli and residue sets
    for m in [5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37, 41, 43, 49, 50]:
        if time.time() > deadline:
            break
        
        # Try various residue sets
        # 1. All residues
        # 2. Half the residues (evens/odds)
        # 3. Small subset
        # 4. Random subset
        
        for K in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
            if time.time() > deadline:
                break
            
            # All residues
            residues = list(range(m))
            gaps, s = try_modular_lattice(m, residues, K)
            if gaps and s > best_score:
                best_score = s
                best_gaps = gaps[:]
            
            # Even residues
            if m >= 4:
                residues = list(range(0, m, 2))
                gaps, s = try_modular_lattice(m, residues, K)
                if gaps and s > best_score:
                    best_score = s
                    best_gaps = gaps[:]
            
            # Small subset: first few residues
            for ns in [2, 3, 4]:
                if ns > m:
                    continue
                residues = list(range(ns))
                gaps, s = try_modular_lattice(m, residues, K)
                if gaps and s > best_score:
                    best_score = s
                    best_gaps = gaps[:]
            
            # Random subset of residues
            for trial in range(3):
                if time.time() > deadline:
                    break
                ns = np.random.randint(2, min(m, 8))
                residues = np.random.choice(m, size=ns, replace=False).tolist()
                if 0 not in residues:
                    residues.append(0)
                residues = sorted(set(residues))
                gaps, s = try_modular_lattice(m, residues, K)
                if gaps and s > best_score:
                    best_score = s
                    best_gaps = gaps[:]
        
        # Also try "staggered" modular lattices where layers are offset
        for offset_pattern in [[0, 1, 0, 1], [0, 1, 2, 0], [0, 1, 2, 3]]:
            if time.time() > deadline:
                break
            K = len(offset_pattern) * 2
            elements = set()
            for k in range(K):
                offset = offset_pattern[k % len(offset_pattern)]
                for r in range(0, m, 2):  # even residues
                    elements.add((r + offset) % m + (k // 2) * m)
            elements.add(0)
            U = sorted(elements)
            if U and len(U) > 1:
                gaps = [U[i+1] - U[i] for i in range(len(U)-1)]
                s = compute_score_from_gaps(gaps)
                if s > best_score:
                    best_score = s
                    best_gaps = gaps[:]
    
    # Also try the original gap-based local search with some good initial candidates
    candidates = []
    
    # Geometric-like
    g = [1]
    for i in range(1, 8):
        g.append(g[-1] * 2)
    candidates.append(g)
    
    # Modular-inspired initial gaps
    for m in [7, 11, 13, 17, 19]:
        # Gaps that reflect the modular structure
        g = [1] * (m - 1) + [m - (m - 1)]
        # Simplify: gaps are mostly 1 with occasional larger gap
        g2 = []
        for i in range(m):
            if i < m - 1:
                g2.append(1)
            else:
                g2.append(m - (m - 1))
        candidates.append(g2)
    
    # Random
    for _ in range(5):
        g = np.random.randint(1, 6, size=np.random.randint(8, 20)).tolist()
        candidates.append(g)
    
    for g in candidates:
        if time.time() > deadline:
            break
        s = compute_score_from_gaps(g)
        if s > best_score:
            best_score = s
            best_gaps = g[:]
    
    # Local search on the best gaps found
    if best_gaps:
        current_gaps = best_gaps[:]
        current_score = best_score
        no_improve = 0
        max_no_improve = 150
        
        while time.time() < deadline and no_improve < max_no_improve:
            new_gaps = current_gaps[:]
            move_type = np.random.randint(0, 5)
            
            if move_type == 0 and len(new_gaps) > 0:  # Modify gap
                idx = np.random.randint(0, len(new_gaps))
                delta = np.random.choice([-3, -2, -1, 1, 2, 3, 4, 5])
                new_val = new_gaps[idx] + delta
                if new_val >= 1:
                    new_gaps[idx] = new_val
            
            elif move_type == 1 and len(new_gaps) < 35:  # Insert gap
                idx = np.random.randint(0, len(new_gaps) + 1)
                new_val = np.random.randint(1, 6)
                new_gaps.insert(idx, new_val)
            
            elif move_type == 2 and len(new_gaps) > 5:  # Remove gap
                idx = np.random.randint(0, len(new_gaps))
                new_gaps.pop(idx)
            
            elif move_type == 3 and len(new_gaps) > 1:  # Swap
                i, j = np.random.choice(len(new_gaps), 2, replace=False)
                new_gaps[i], new_gaps[j] = new_gaps[j], new_gaps[i]
            
            elif move_type == 4 and len(new_gaps) > 2:  # Duplicate a gap pattern
                idx = np.random.randint(0, len(new_gaps) - 1)
                new_gaps.insert(idx + 1, new_gaps[idx])
            
            if new_gaps != current_gaps:
                new_score = compute_score_from_gaps(new_gaps)
                if new_score > current_score:
                    current_gaps = new_gaps
                    current_score = new_score
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_gaps = current_gaps[:]
                else:
                    no_improve += 1
        
        # Final polish: try small perturbations on the best
        for _ in range(50):
            if time.time() > deadline:
                break
            new_gaps = best_gaps[:]
            if len(new_gaps) > 0:
                idx = np.random.randint(0, len(new_gaps))
                delta = np.random.choice([-1, 1])
                new_val = new_gaps[idx] + delta
                if new_val >= 1:
                    new_gaps[idx] = new_val
                    s = compute_score_from_gaps(new_gaps)
                    if s > best_score:
                        best_score = s
                        best_gaps = new_gaps[:]
    
    # Construct final set
    if not best_gaps:
        return [0, 1]
    
    U = [0]
    for g in best_gaps:
        U.append(U[-1] + g)
    
    return U

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
