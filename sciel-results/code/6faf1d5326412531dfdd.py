import time
import numpy as np
from itertools import combinations

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
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
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
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Strategy:
    # 1. Generate a pool of "Atoms" (small sets S containing 0) with high ratio |S-S|/|S+S|.
    # 2. Use the mixed-radix factorization property:
    #    If U = D_0 + M_1 D_1 + ... + M_K D_K with M_{i+1} > 2*max(U_i),
    #    then |U+U| = prod |D_i+D_i| and |U-U| = prod |D_i-D_i|.
    #    This allows an O(K) analytic score calculation, which is extremely fast.
    # 3. Perform a local search on the sequence of atoms to maximize the score.
    # 4. Use the saved best sequence to construct the final set U.
    
    # Step 1: Generate Atoms
    # We look for atoms of size 2 to 6 with span up to 30.
    # We want to maximize log(ratio) / log(span) roughly, but also consider absolute ratio.
    atoms = []
    max_span_atom = 30
    
    # Precompute sumset/diffset sizes for atoms
    # To speed up generation, we can prune based on known bad structures (like APs)
    # But for size < 6 and span < 30, it's fast enough to just compute.
    
    # We will store:
    # 'set': the list of integers
    # 'r': |S-S|/|S+S|
    # 'span': max(S)
    # 'log_r': log(r)
    # 'log_span': log(span)
    # 'efficiency': log_r / log_span (rough heuristic for greedy)
    
    # We also need the actual sizes for the analytic formula?
    # No, the analytic formula uses log(|S-S|) - log(|S+S|) = log(r).
    # So we just need log_r.
    
    # However, we must be careful with the "gating" condition.
    # The condition M_{i+1} > 2 * max(U_i) ensures no carries in the sumset representation.
    # Specifically, if U_i = { sum_{j=0}^{i} d_j M_j }, then max(U_i) is determined by the atoms.
    # The condition is M_{i+1} > 2 * max(U_i).
    # Let's define the construction recursively.
    # U_0 = D_0. Max_0 = max(D_0).
    # M_1 = 2 * Max_0 + 1.
    # U_1 = D_0 + M_1 D_1. Max_1 = Max_0 + M_1 * max(D_1).
    # M_2 = 2 * Max_1 + 1.
    # And so on.
    
    # The total max U = Max_K.
    # The total ratio = prod r_i.
    # Score = 1 + sum(log r_i) / log(2 * Max_K + 1).
    
    # Let's generate atoms.
    # To ensure diversity and good performance, we filter for r > 1.
    # We also want to avoid atoms that are "too regular" like arithmetic progressions,
    # although the search will handle that.
    
    # Optimization: Only consider atoms where 0 is the minimum.
    # We can normalize atoms by shifting so min is 0.
    # But the definition requires 0 in U. If we shift, we must ensure 0 is in the set.
    # So we generate sets containing 0.
    
    # Let's iterate over combinations of positive integers.
    # Size k means k positive integers + 0.
    
    for size in range(2, 7): # 2 to 6 elements total
        # We need size-1 positive integers
        # To avoid duplicates and ensure sorted, we use combinations
        # Range: 1 to max_span_atom
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
    
    # Limit the pool to the top 50 most efficient atoms to keep search fast
    # and avoid too many "bad" atoms.
    # Actually, sometimes a less efficient atom has a better span (smaller),
    # which helps the denominator. So we keep a mix.
    # Let's keep top 100 by efficiency.
    search_pool_indices = list(range(min(100, len(atoms))))
    
    # Step 2: Analytic Score Function
    # Inputs: list of atom indices
    # Returns: score
    
    def calc_analytic_score(atom_indices):
        if not atom_indices:
            return 0.0
        
        log_num = 0.0
        max_u = 0
        m = 1 # M_0 is effectively 1 for the first atom?
              # Let's trace:
              # U = D_0. Max(U) = max(D_0).
              # Next M = 2*max(U)+1.
              # U_new = U + M * D_1.
              # Max(U_new) = max(U) + M * max(D_1).
              # So the first atom is just D_0.
        
        for idx in atom_indices:
            S = atoms[idx]['set']
            log_num += atoms[idx]['log_r']
            
            span = atoms[idx]['span']
            
            # Update Max(U)
            # If this is the first atom, max_u = span.
            # If not, max_u = max_u + m * span.
            if max_u == 0:
                max_u = span
            else:
                max_u = max_u + m * span
                
            # Update m for next step
            # m_next = 2 * max_u + 1
            m = 2 * max_u + 1
            
            # Safety check for overflow
            if max_u > 10**18:
                return -np.inf
                
        if max_u == 0:
            return 0.0
            
        log_denom = np.log(2 * max_u + 1)
        if log_denom == 0:
            return 0.0
            
        return 1.0 + log_num / log_denom

    # Step 3: Local Search
    # Initialize with the best single atom
    best_score = 0.0
    best_seq = []
    
    for idx in search_pool_indices:
        s = calc_analytic_score([idx])
        if s > best_score:
            best_score = s
            best_seq = [idx]
            
    # Greedy expansion
    # Try adding atoms to the sequence.
    # The score is roughly 1 + (sum log_r) / log(2 max_u + 1).
    # Adding an atom increases log_num by log_r_i and increases log_denom.
    # We want to add atoms where the marginal gain is positive.
    
    current_seq = best_seq[:]
    current_score = best_score
    
    # We will perform a local search on the sequence.
    # Moves:
    # 1. Add an atom at the end
    # 2. Remove an atom
    # 3. Replace an atom
    # 4. Swap two atoms (order matters for max_u growth?
    #    Yes, because M depends on the previous max_u.
    #    However, the product of ratios is commutative.
    #    The denominator depends on the final max_u.
    #    Max_u is minimized if we put "small span" atoms first?
    #    Let's check:
    #    U1 = D1. M1 = 2 max(D1) + 1.
    #    U2 = D1 + M1 D2. Max = max(D1) + M1 max(D2).
    #    If we swap:
    #    U1' = D2. M1' = 2 max(D2) + 1.
    #    U2' = D2 + M1' D1. Max' = max(D2) + M1' max(D1).
    #    Generally, we want the atom with larger span to be later?
    #    Or earlier?
    #    Let's let the search figure it out.)
    
    step = 0
    no_improve_count = 0
    max_no_improve = 100
    
    # We will use a "temperature" for simulated annealing?
    # Or just simple hill climbing with random restarts.
    # Given the speed of calc_analytic_score, we can do many steps.
    
    while time.time() < deadline:
        step += 1
        improved = False
        
        # Choose a move
        # We bias towards adding atoms if the sequence is short,
        # and replacing if it's long.
        
        move_type = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.3, 0.1])
        
        new_seq = current_seq[:]
        
        if move_type == 0: # Add atom
            if len(new_seq) < 20: # Cap sequence length to avoid excessive max_u
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
            
            # Simple hill climbing
            if new_score > current_score:
                current_seq = new_seq
                current_score = new_score
                improved = True
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_seq = current_seq[:]
            else:
                no_improve_count += 1
                
        # Random restart if stuck
        if no_improve_count > max_no_improve:
            # Generate a random sequence
            size = np.random.randint(1, 6)
            new_seq = []
            for _ in range(size):
                idx = search_pool_indices[np.random.randint(0, len(search_pool_indices))]
                new_seq.append(idx)
            current_seq = new_seq
            current_score = calc_analytic_score(current_seq)
            no_improve_count = 0
            
            # Also try a scaled version of the best?
            # Or just a greedy build from scratch?
            # Greedy build:
            g_seq = []
            g_score = 0
            for _ in range(10): # Try to build up to 10 atoms greedily
                best_add_idx = -1
                best_add_score = g_score
                for idx in search_pool_indices[:20]: # Top 20 only for speed
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
        # Fallback
        return [0, 1]
        
    U = set([0])
    max_u = 0
    m = 1
    
    for idx in best_seq:
        S = atoms[idx]['set']
        span = atoms[idx]['span']
        
        # Update U
        # U_new = U + m * S
        # This is a Minkowski sum
        new_U = set()
        for u in U:
            for s in S:
                new_U.add(u + m * s)
        U = new_U
        
        # Update max_u
        if max_u == 0:
            max_u = span
        else:
            max_u = max_u + m * span
            
        # Update m
        m = 2 * max_u + 1
        
        # Safety check
        if max_u > 10**18:
            break
            
    # Convert to list and sort
    U_list = sorted(list(U))
    
    # Ensure 0 is in the set
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
