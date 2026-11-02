import time
import numpy as np
from itertools import combinations

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
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    # Strategy:
    # 1. Generate a pool of "Atoms" (small sets S containing 0) with high ratio |S-S|/|S+S|.
    # 2. Use the mixed-radix factorization property:
    #    If U = D_0 + M_1 D_1 + ... + M_K D_K with M_{i+1} > 2*max(U_i),
    #    then |U+U| = prod |D_i+D_i| and |U-U| = prod |D_i-D_i|.
    #    This allows an O(K) analytic score calculation.
    # 3. Perform a local search on the sequence of atoms to maximize the score.
    # 4. Use the saved best sequence to construct the final set U.
    
    # Step 1: Generate Atoms
    # We look for atoms of size 2 to 8 with span up to 40.
    # We want to maximize log(ratio) / log(span) roughly.
    
    atoms = []
    max_span_atom = 40
    
    # We will store:
    # 'set': the list of integers
    # 'r': |S-S|/|S+S|
    # 'span': max(S)
    # 'log_r': log(r)
    # 'log_span': log(span)
    # 'efficiency': log_r / log_span
    
    for size in range(2, 9): # 2 to 8 elements total
        # We need size-1 positive integers
        for comb in combinations(range(1, max_span_atom + 1), size - 1):
            S = [0] + list(comb)
            S.sort()
            
            # Calculate ratio
            sumset = set()
            diffset = set()
            for a in S:
                for b in S:
                    sumset.add(a + b)
                    diffset.add(a - b)
            
            r = len(diffset) / len(sumset)
            if r > 1.0:
                span = max(S)
                log_r = np.log(r)
                log_span = np.log(span)
                atoms.append({
                    'set': S,
                    'r': r,
                    'span': span,
                    'log_r': log_r,
                    'log_span': log_span,
                    'efficiency': log_r / log_span
                })
                
    # Sort by efficiency to have a good starting pool
    atoms.sort(key=lambda x: x['efficiency'], reverse=True)
    
    # Limit the pool to the top 150 most efficient atoms
    search_pool_indices = list(range(min(150, len(atoms))))
    
    # Step 2: Analytic Score Function
    def calc_analytic_score(atom_indices):
        if not atom_indices:
            return 0.0
        
        log_num = 0.0
        max_u = 0
        m = 1
        
        for idx in atom_indices:
            S = atoms[idx]['set']
            log_num += atoms[idx]['log_r']
            
            span = atoms[idx]['span']
            
            if max_u == 0:
                max_u = span
            else:
                max_u = max_u + m * span
                
            m = 2 * max_u + 1
            
            if max_u > 10**18:
                return -np.inf
                
        if max_u == 0:
            return 0.0
            
        log_denom = np.log(2 * max_u + 1)
        if log_denom == 0:
            return 0.0
            
        return 1.0 + log_num / log_denom

    # Step 3: Local Search with Simulated Annealing
    best_score = 0.0
    best_seq = []
    
    for idx in search_pool_indices:
        s = calc_analytic_score([idx])
        if s > best_score:
            best_score = s
            best_seq = [idx]
            
    current_seq = best_seq[:]
    current_score = best_score
    
    # Simulated annealing parameters
    T_start = 0.05
    T_end = 0.001
    alpha = 0.995
    T = T_start
    no_improve_count = 0
    max_no_improve = 50
    
    step = 0
    while time.time() < deadline:
        step += 1
        no_improve_count += 1
        improved = False
        
        # Choose a move
        move_type = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.3, 0.1])
        
        new_seq = current_seq[:]
        
        if move_type == 0: # Add atom
            if len(new_seq) < 25:
                idx = search_pool_indices[np.random.randint(0, len(search_pool_indices))]
                pos = np.random.randint(0, len(new_seq) + 1)
                new_seq.insert(pos, idx)
                
        elif move_type == 1: # Remove atom
            if len(new_seq) > 1:
                pos = np.random.randint(0, len(new_seq))
                new_seq.pop(pos)
                
        elif move_type == 2: # Replace atom
            if len(new_seq) > 0:
                pos = np.random.randint(0, len(new_seq))
                idx = search_pool_indices[np.random.randint(0, len(search_pool_indices))]
                new_seq[pos] = idx
                
        elif move_type == 3: # Swap
            if len(new_seq) > 1:
                i, j = np.random.choice(len(new_seq), 2, replace=False)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                
        if new_seq != current_seq:
            new_score = calc_analytic_score(new_seq)
            delta_score = new_score - current_score
            
            # Metropolis acceptance
            accept = False
            if delta_score > 0:
                accept = True
            else:
                if T > T_end:
                    p = np.exp(delta_score / T)
                    if np.random.rand() < p:
                        accept = True
            
            if accept:
                current_seq = new_seq
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_seq = current_seq[:]
                    no_improve_count = 0
                    improved = True
        
        # Cool down
        T *= alpha
        
        # Random restart if stuck for too long
        if no_improve_count > max_no_improve:
            # Generate a random sequence
            size = np.random.randint(1, 8)
            new_seq = []
            for _ in range(size):
                idx = search_pool_indices[np.random.randint(0, len(search_pool_indices))]
                new_seq.append(idx)
            current_seq = new_seq
            current_score = calc_analytic_score(current_seq)
            no_improve_count = 0
            T = max(T, T_start * 0.5)
            
            # Also try a greedy build from scratch?
            g_seq = []
            g_score = 0
            for _ in range(15):
                best_add_idx = -1
                best_add_score = g_score
                for idx in search_pool_indices[:30]:
                    test_seq = g_seq + [idx]
                    s = calc_analytic_score(test_seq)
                    if s > best_add_score:
                        best_add_score = s
                        best_add_idx = idx
                if best_add_idx != -1:
                    g_seq.append(best_add_idx)
                    g_score = best_add_score
                else:
                    break
                    
            if g_score > current_score:
                current_seq = g_seq
                current_score = g_score
                if g_score > best_score:
                    best_score = g_score
                    best_seq = g_seq[:]

    # Step 4: Construct the final set U from best_seq
    if not best_seq:
        return [0, 1]
        
    U = set([0])
    max_u = 0
    m = 1
    
    for idx in best_seq:
        S = atoms[idx]['set']
        span = atoms[idx]['span']
        
        new_U = set()
        for u in U:
            for s in S:
                new_U.add(u + m * s)
        U = new_U
        
        if max_u == 0:
            max_u = span
        else:
            max_u = max_u + m * span
            
        m = 2 * max_u + 1
        
        if max_u > 10**18:
            break
            
    U_list = sorted(list(U))
    
    if 0 not in U_list:
        U_list.insert(0, 0)
        
    return U_list

if __name__ == "__main__":
    s = construct()
    print(f"Result Size: {len(s)}")
    if len(s) < 20:
        print(f"Result: {s}")
    else:
        print(f"Result: {s[:10]}... {s[-5:]}")
    print(f"Score: {_score_set(s)}")
