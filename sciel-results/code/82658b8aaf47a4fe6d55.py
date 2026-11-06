import numpy as np
import time
import sys

N = 51975

def get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return [d for d in divisors if d > 1]

def precompute_indices(moduli):
    # indices[m][r] will be a numpy array of indices
    indices = {}
    for m in moduli:
        indices[m] = []
        for r in range(m):
            # x = r + k*m < N
            # k < (N - r) / m
            # indices are r, r+m, r+2m...
            arr = np.arange(r, N, m)
            indices[m].append(arr)
    return indices

def calculate_score(covered):
    return N - covered.sum()

def greedy_construct(moduli, indices, order):
    covered = np.zeros(N, dtype=bool)
    classes = []
    available = set(moduli)
    
    # Order: 'dynamic', 'desc', 'asc'
    while available:
        best_gain = 0
        best_m = None
        best_r = None
        
        if order == 'dynamic':
            for m in list(available):
                for r in range(m):
                    idx = indices[m][r]
                    # Count uncovered
                    gain = (~covered[idx]).sum()
                    if gain > best_gain:
                        best_gain = gain
                        best_m = m
                        best_r = r
        else:
            # Static order
            sorted_mods = sorted(available, reverse=(order == 'desc'))
            for m in sorted_mods:
                best_r_for_m = 0
                max_gain_for_m = 0
                for r in range(m):
                    idx = indices[m][r]
                    gain = (~covered[idx]).sum()
                    if gain > max_gain_for_m:
                        max_gain_for_m = gain
                        best_r_for_m = r
                if max_gain_for_m > best_gain:
                    best_gain = max_gain_for_m
                    best_m = m
                    best_r = best_r_for_m
        
        if best_m is None or best_gain == 0:
            break
            
        classes.append([best_m, best_r])
        available.remove(best_m)
        covered[indices[best_m][best_r]] = True
        
    return classes, covered

def local_search(classes, indices, covered):
    # classes is a list of [m, r]
    # We want to optimize residues and potentially add/remove moduli
    # Since adding a modulus never hurts, we assume we start with a set of moduli
    # and try to optimize residues. We also try to add moduli not in the set.
    
    current_score = calculate_score(covered)
    moduli_in_use = {c[0] for c in classes}
    all_moduli = set(indices.keys())
    
    improved = True
    while improved:
        improved = False
        
        # 1. Optimize residues for existing moduli
        for i, (m, r) in enumerate(classes):
            best_r = r
            best_gain = 0
            # Current contribution of this class is not simply 'gain' because
            # other classes might cover some of its elements.
            # To find the best r for m, we should temporarily remove m,
            # then find the r that maximizes the new coverage.
            
            # Save current state? No, we can just calculate the score if we change r.
            # Score = N - covered.sum()
            # If we change r to r', the new covered array will be different.
            # Efficient way:
            # Let C be the set of elements covered by classes OTHER than m.
            # Let S_r be the set of elements covered by (m, r).
            # Total covered = C U S_r.
            # Size = |C| + |S_r \ C|.
            # |C| is constant for fixed other classes.
            # So we maximize |S_r \ C|.
            # C is everything covered except by (m, r).
            
            # To get C, we can't easily subtract.
            # Alternative:
            # 1. Remove (m, r) from covered.
            # 2. For each candidate r', count how many elements in S_r' are not in covered.
            # 3. Pick r' with max count.
            # 4. Update covered with (m, r').
            
            # Remove current
            idx_current = indices[m][r]
            # We need to know which elements were ONLY covered by this class?
            # No, we just need the state of covered WITHOUT this class.
            # This is tricky with a single boolean array.
            
            # Let's use a different approach for local search step:
            # Recalculate the "background" coverage.
            # This is expensive if done for every m.
            
            # Optimization:
            # Since N is small, we can just recompute the score for each candidate r.
            # Score(r') = N - (covered_with_m_replaced_by_r').sum()
            # To compute covered_with_m_replaced_by_r':
            # We can't easily do this with a single array without copying.
            # Copying a 52k bool array is fast (~microseconds).
            
            # Let's try the copy approach.
            base_covered = covered.copy()
            # Remove current m, r
            # We need to know which elements were covered by m,r and NOT by others.
            # This is hard.
            
            # Alternative:
            # Just evaluate the full score for each candidate r.
            # To do this, we need to construct the covered array for that candidate.
            # This requires summing all other classes + the new one.
            # That's O(K * N) per candidate, where K is number of classes.
            # K ~ 47. N ~ 52k. 47*52k ~ 2.4M ops.
            # Candidates per m ~ m.
            # Total ops per pass ~ sum(m * 2.4M) ~ 12k * 2.4M ~ 28 Billion. Too slow.
            
            # We need a faster local search.
            
            # Let's stick to the "Gain" heuristic but be careful.
            # If we just look at `(~covered[idx]).sum()`, we are assuming
            # that the elements covered by this class are the only ones that matter.
            # But if an element is covered by another class, changing this class
            # doesn't uncover it.
            # So the "gain" of switching to r' is:
            # (Elements covered by r' but not by current r AND not by others)
            # - (Elements covered by current r but not by r' AND not by others)
            
            # Let O be the set of elements covered by OTHER classes.
            # Gain(r') = | (S_r' \ O) \ (S_r \ O) | - | (S_r \ O) \ (S_r' \ O) |
            #           = | (S_r' \ S_r) \ O | - | (S_r \ S_r') \ O |
            
            # We can compute O for a given m by removing m from the coverage.
            # But we don't have O stored.
            
            # Let's use a simpler local search:
            # Randomly pick a modulus m.
            # Try all residues r.
            # For each r, simulate the change:
            # 1. Determine elements that will be UNCOVERED: U = S_r_current \ S_r_new \ O
            # 2. Determine elements that will be COVERED: C = S_r_new \ S_r_current \ O
            # 3. If |C| > |U|, accept.
            
            # To do this, we need O.
            # We can approximate O or compute it on the fly.
            # Computing O for one m:
            # O = Union of S_{m', r'} for all m' != m.
            # We can compute this by taking `covered` and subtracting S_{m, r_current}?
            # No, because some elements in S_{m, r_current} are also in O.
            # So `covered` is O U S_{m, r_current}.
            
            # Let's just use the copy method but optimize it.
            # Actually, for N=52k, copying and summing is very fast in NumPy.
            # Let's benchmark mentally.
            # 52k bytes copy: ~0.1 us.
            # Sum: ~0.1 us.
            # Total per candidate: ~1 us.
            # Candidates per m: ~m.
            # Pass over all m: sum(m) ~ 12k candidates.
            # Time per pass: ~12 ms.
            # This is very fast!
            
            # So:
            # 1. For each m in classes:
            #    a. Save current covered.
            #    b. For each r in range(m):
            #       1. Create new_covered = covered.copy()
            #       2. We need to remove the old r and add the new r.
            #          This is hard because we don't know what was uniquely covered by old r.
            
            # Correct approach for "What if I change m to r'":
            # 1. Compute coverage of all classes EXCEPT m. Let's call this `base_covered`.
            #    We can compute this once per m.
            #    base_covered = zeros(N)
            #    for (m2, r2) in classes:
            #       if m2 != m:
            #          base_covered[indices[m2][r2]] = True
            # 2. For each candidate r':
            #    new_covered = base_covered.copy()
            #    new_covered[indices[m][r']] = True
            #    score = N - new_covered.sum()
            # 3. Pick best r'.
            
            # Cost:
            # Step 1: O(K * N) per m. K~47, N~52k. ~2.4M ops.
            # Step 2: O(m * N) per m.
            # Total per m: (K + m) * N.
            # Total for all m: sum_m (K + m) * N = (K^2 + sum m) * N
            # K^2 ~ 2200. sum m ~ 12k.
            # (2200 + 12000) * 52000 ~ 14000 * 52000 ~ 728 Million ops.
            # In NumPy, 728M simple ops might take ~1-2 seconds.
            # This is acceptable for one pass.
            # We can do a few passes.
            
            # Let's implement this.
            
            base_covered = np.zeros(N, dtype=bool)
            for m2, r2 in classes:
                if m2 != m:
                    base_covered[indices[m2][r2]] = True
            
            best_r = r
            best_score = N # max possible uncovered is N
            
            for r_candidate in range(m):
                new_covered = base_covered.copy()
                new_covered[indices[m][r_candidate]] = True
                score = N - new_covered.sum()
                if score < best_score:
                    best_score = score
                    best_r = r_candidate
            
            if best_r != r:
                classes[i] = [m, best_r]
                # Update global covered
                # We can just recompute global covered at the end of the pass
                # or update it carefully.
                # Let's just recompute global covered after the loop for this m.
                improved = True
                
        # Recompute global covered after optimizing residues
        covered = np.zeros(N, dtype=bool)
        for m, r in classes:
            covered[indices[m][r]] = True
            
        # 2. Try to add new moduli
        # For each m not in classes:
        #    Compute base_covered (which is just current covered)
        #    Find r that minimizes score
        #    If score improves, add m
        
        current_score = N - covered.sum()
        for m in all_moduli - moduli_in_use:
            best_r = None
            best_score = current_score
            
            for r in range(m):
                new_covered = covered.copy()
                new_covered[indices[m][r]] = True
                score = N - new_covered.sum()
                if score < best_score:
                    best_score = score
                    best_r = r
            
            if best_r is not None and best_score < current_score:
                classes.append([m, best_r])
                moduli_in_use.add(m)
                covered = np.zeros(N, dtype=bool)
                for mm, rr in classes:
                    covered[indices[mm][rr]] = True
                current_score = best_score
                improved = True
                
    return classes, covered

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    moduli = get_divisors(N)
    indices = precompute_indices(moduli)
    
    best_classes = []
    best_score = N
    
    # Try different greedy orders
    orders = ['dynamic', 'desc', 'asc']
    
    for order in orders:
        if time.time() > deadline:
            break
            
        classes, covered = greedy_construct(moduli, indices, order)
        score = calculate_score(covered)
        
        if score < best_score:
            best_score = score
            best_classes = classes.copy()
            
        # Local Search
        classes_ls, covered_ls = local_search(classes, indices, covered)
        score_ls = calculate_score(covered_ls)
        
        if score_ls < best_score:
            best_score = score_ls
            best_classes = classes_ls.copy()
            
    return best_classes
