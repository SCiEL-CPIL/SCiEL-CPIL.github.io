import time
import numpy as np
import itertools

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
    
    N = 2 * mx + 1
    
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6)
    
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
    deadline = start_time + 115.0
    
    # Generate atoms: small sets S containing 0 with ratio r = |S-S|/|S+S| > 1
    # and span = max(S).
    # We want to maximize the "efficiency" = log(r) / log(span+1) roughly.
    # But actually, in the mixed-radix construction:
    # U = sum d_i M_i, M_{i+1} = 2*Max(U_i)+1
    # |U+U| = prod |D_i+D_i|, |U-U| = prod |D_i-D_i|
    # Max(U) = sum d_i M_i (where d_i = max(D_i))
    # 2*Max(U)+1 is roughly 2 * sum d_i M_i
    # log(2*Max(U)+1) = sum log(M_i) + log(2*...)
    # Actually, M_i grows exponentially.
    # Let's just precompute atoms and their properties.
    
    atoms = []
    max_span = 30
    for size in range(2, 6):
        for comb in itertools.combinations(range(1, max_span + 1), size - 1):
            S = [0] + list(comb)
            S.sort()
            sumset = set()
            diffset = set()
            for a in S:
                for b in S:
                    sumset.add(a + b)
                    diffset.add(a - b)
            r = len(diffset) / len(sumset)
            if r > 1.001:
                span = max(S)
                atoms.append({
                    'set': S,
                    'r': r,
                    'span': span,
                    'log_r': np.log(r),
                    'log_span': np.log(span + 1)
                })
                
    # Sort by efficiency
    atoms.sort(key=lambda x: x['log_r'] / x['log_span'], reverse=True)
    
    def calc_analytic_score(atom_indices):
        if not atom_indices:
            return 0.0
        log_num = 0.0
        max_u = 0
        m = 1
        for idx in atom_indices:
            S = atoms[idx]['set']
            log_num += atoms[idx]['log_r']
            new_max_u = max_u + atoms[idx]['span'] * m
            max_u = new_max_u
            m = 2 * max_u + 1
            if max_u > 10**18:
                return -np.inf
        if max_u == 0:
            return 0.0
        log_denom = np.log(2 * max_u + 1)
        if log_denom == 0:
            return 0.0
        return 1.0 + log_num / log_denom

    # Greedy initialization: pick atoms that maximize incremental score
    # Start with empty, then add the atom that gives the best score.
    # Then keep adding.
    
    best_score = 0.0
    best_seq = []
    
    # Try all single atoms
    for i, a in enumerate(atoms[:100]):
        s = calc_analytic_score([i])
        if s > best_score:
            best_score = s
            best_seq = [i]
            
    # Greedy expansion
    current_seq = best_seq[:]
    current_score = best_score
    
    # Keep adding the best atom until score stops improving or time runs out
    while time.time() < deadline - 20:
        best_add_score = current_score
        best_add_idx = -1
        best_add_pos = len(current_seq)  # default: append
        
        # Try adding at each position
        for pos in range(len(current_seq) + 1):
            for a_idx in range(min(50, len(atoms))):
                new_seq = current_seq[:pos] + [a_idx] + current_seq[pos:]
                s = calc_analytic_score(new_seq)
                if s > best_add_score + 1e-10:
                    best_add_score = s
                    best_add_idx = a_idx
                    best_add_pos = pos
                    
        if best_add_idx == -1:
            break
            
        current_seq = current_seq[:best_add_pos] + [best_add_idx] + current_seq[best_add_pos:]
        current_score = best_add_score
        if current_score > best_score:
            best_score = current_score
            best_seq = current_seq[:]
            
        if len(current_seq) > 50:
            break

    # Local search on the sequence
    # Moves: replace, swap, insert, delete
    search_pool = list(range(min(30, len(atoms))))
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for _ in range(100):  # multiple passes
            if time.time() > deadline - 10:
                break
            move_type = np.random.choice(['replace', 'swap', 'insert', 'delete'])
            
            if move_type == 'replace':
                if not current_seq:
                    continue
                pos = np.random.randint(0, len(current_seq))
                new_idx = search_pool[np.random.randint(0, len(search_pool))]
                new_seq = current_seq[:]
                new_seq[pos] = new_idx
                s = calc_analytic_score(new_seq)
                if s > current_score + 1e-12:
                    current_seq = new_seq
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_seq = current_seq[:]
                        
            elif move_type == 'swap':
                if len(current_seq) < 2:
                    continue
                p1, p2 = np.random.choice(len(current_seq), 2, replace=False)
                new_seq = current_seq[:]
                new_seq[p1], new_seq[p2] = new_seq[p2], new_seq[p1]
                s = calc_analytic_score(new_seq)
                if s > current_score + 1e-12:
                    current_seq = new_seq
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_seq = current_seq[:]
                        
            elif move_type == 'insert':
                if len(current_seq) > 50:
                    continue
                pos = np.random.randint(0, len(current_seq) + 1)
                new_idx = search_pool[np.random.randint(0, len(search_pool))]
                new_seq = current_seq[:pos] + [new_idx] + current_seq[pos:]
                s = calc_analytic_score(new_seq)
                if s > current_score + 1e-12:
                    current_seq = new_seq
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_seq = current_seq[:]
                        
            elif move_type == 'delete':
                if len(current_seq) < 2:
                    continue
                pos = np.random.randint(0, len(current_seq))
                new_seq = current_seq[:pos] + current_seq[pos+1:]
                s = calc_analytic_score(new_seq)
                if s > current_score + 1e-12:
                    current_seq = new_seq
                    current_score = s
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_seq = current_seq[:]

    # Construct the final set
    if not best_seq:
        return [0, 1]
        
    U = [0]
    M = 1
    for idx in best_seq:
        S = atoms[idx]['set']
        new_U = []
        for u in U:
            for d in S:
                new_U.append(u + d * M)
        U = new_U
        if not U:
            break
        M = 2 * max(U) + 1
        if len(U) > 100000:
            break
        if time.time() > deadline - 5:
            break
            
    return U

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
