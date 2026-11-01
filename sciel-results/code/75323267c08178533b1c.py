"""Difference triangle set (4,7): spectral clustering on difference conflict graph +
   greedy row assembly + ILS with backtracking polish."""
import time
import random
import math

import numpy as np

_DEADLINE = 145.0  # hard wall-clock kill margin (evaluator kills at ~150s)


def _row_diffs(row):
    """All positive differences in a row."""
    out = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out


def _is_valid(rows):
    seen = set()
    for r in rows:
        for d in _row_diffs(r):
            if d in seen:
                return False
            seen.add(d)
    return True


def _scope(rows):
    return max(r[-1] for r in rows)


def _greedy_build_row(used, k, max_val):
    """Greedily build one row of k+1 increasing ints starting at 0,
       avoiding used differences. Returns row or None."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > max_val:
            return None
        new_diffs = [x - a for a in row]
        if any(d in used for d in new_diffs):
            continue
        for d in new_diffs:
            used.add(d)
        row.append(x)
    return row


def _greedy_baseline(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_build_row(used, k, 10**6)
        if row is None:
            break
        rows.append(row)
    return rows


def _build_conflict_graph(n_rows, k, max_scope):
    """Build a conflict graph over candidate differences [1..max_scope].
    Two differences conflict if they cannot coexist in the same row
    (i.e., they'd create a duplicate difference within a single row).
    Actually, the real constraint is: within one row, all pairwise diffs must be distinct.
    So two diffs d1, d2 conflict in a row if there exist positions where both appear
    AND they'd force a duplicate. This is complex; instead use a simpler model:
    each row needs C(k+1,2) distinct diffs. We want to partition [1..max_scope] into
    n_rows groups of size C(k+1,2) such that each group can form a valid row.
    
    For the conflict graph: diff d1 and d2 are 'compatible' if they can both appear
    in the same row. They're incompatible if... actually every pair of distinct positive
    integers CAN appear in the same row (e.g., row [0, d2, d2+d1] has diffs d2, d1, d2+d1).
    The real constraint is that ALL C(k+1,2) diffs in a row must be mutually consistent
    (form a valid difference set).
    
    Let's use a different approach: spectral clustering on a compatibility matrix
    where entry (d1,d2) = 1 if d1 and d2 can coexist in a valid row segment."""
    # For a (k+1)-element row, we need k+1 numbers a_0=0 < a_1 < ... < a_k
    # such that all C(k+1,2) differences are distinct.
    # Two differences d1, d2 are "compatible" if there's a valid row containing both.
    # This is hard to compute exactly. Instead, use a heuristic:
    # d1 and d2 are compatible if |d1 - d2| is not equal to either d1 or d2
    # (because if d2 = 2*d1, then in a row [0, d1, 2*d1] we'd have diffs d1, d1, 2*d1 -> duplicate!)
    # More generally, if d2 = m * d1 for some m, they might conflict.
    
    # Simpler heuristic: d1 and d2 conflict if d1 == d2 (obvious) or if one is a multiple
    # of the other in a way that creates forced duplicates.
    # Actually the cleanest model: build the graph where nodes are differences 1..max_scope,
    # and edges connect differences that CANNOT be in the same row.
    # Two diffs d1 < d2 cannot be in the same row if d2 = d1 (impossible, distinct) 
    # or if there's a structural conflict. 
    # 
    # Let me use a different, more principled approach:
    # The key insight is that we want to find n_rows subsets of size C(k+1,2) from [1..max_scope]
    # such that each subset can be realized as the difference set of a (k+1)-element set.
    # 
    # For the spectral approach, let's define compatibility:
    # d1 and d2 are compatible if gcd(d1,d2) == 1 (heuristic: coprime diffs are more likely
    # to coexist without creating arithmetic progressions that force duplicates).
    
    # Actually, let me use the most direct model:
    # A valid row with elements 0 = a_0 < a_1 < ... < a_k has exactly C(k+1,2) differences.
    # The difference set must be "Sidon-like" in the sense that no difference repeats.
    # 
    # For spectral clustering, I'll use: two differences d1, d2 are "attractive" (should be
    # in same row) if they are small and coprime, "repulsive" if they create AP conflicts.
    # 
    # Let me just use: compatibility = 1 - (1 if d2 % d1 == 0 and d1 != 1 else 0) * 0.5
    # This penalizes multiples.
    
    n_diffs = max_scope
    if n_diffs > 500:
        n_diffs = 500  # cap for memory
    
    # Build compatibility matrix (sparse-friendly via numpy)
    # compat[d1-1, d2-1] = 1.0 if d1,d2 can likely coexist, 0.0 if not
    # Use: incompatible if d2 is a multiple of d1 (creates AP risk)
    compat = np.ones((n_diffs, n_diffs), dtype=np.float64)
    np.fill_diagonal(compat, 0.0)
    for i in range(1, n_diffs + 1):
        for j in range(i * 2, n_diffs + 1, i):
            compat[i-1, j-1] = 0.0
            compat[j-1, i-1] = 0.0
    
    # Normalize and compute graph Laplacian
    # L = D - W where W is the weight matrix (compatibility)
    # Actually for clustering we want to MINIMIZE the cut, so we use
    # the normalized Laplacian and find the first n_rows eigenvectors.
    
    # Use W = compat (higher = more compatible = should be together)
    # L = D - W
    degree = compat.sum(axis=1)
    L = np.diag(degree) - compat
    
    # Compute eigenvectors for smallest eigenvalues (after the trivial one)
    # For n_rows=4, we need eigenvectors 1, 2, 3 (0-indexed after removing constant)
    # Use numpy's eigh for symmetric matrices
    if n_diffs > 200:
        # Too large for full eigh; use power iteration or truncated SVD
        # Use scipy if available, else fallback
        try:
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csc_matrix
            L_sp = csc_matrix(L)
            # Find n_rows smallest eigenvalues (excluding the trivial 0 eigenvalue)
            vals, vecs = eigsh(L_sp, k=n_rows, sigma=0.0, which='LM')
            # vecs shape: (n_diffs, n_rows)
            eigenvectors = vecs
        except Exception:
            # Fallback: use random projection
            rng = np.random.RandomState(42)
            eigenvectors = rng.randn(n_diffs, n_rows)
    else:
        vals, vecs = np.linalg.eigh(L)
        # Skip the first (constant) eigenvector
        eigenvectors = vecs[:, 1:n_rows+1]
    
    # Cluster using k-means on the eigenvector coordinates
    # Normalize rows
    norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = eigenvectors / norms
    
    # Simple k-means
    rng = np.random.RandomState(123)
    # Initialize with k-means++
    centers = [X[rng.randint(n_diffs)]]
    for _ in range(n_rows - 1):
        dists = np.array([np.sum((X - c)**2, axis=1) for c in centers])
        min_dists = dists.min(axis=0)
        probs = min_dists / min_dists.sum()
        idx = rng.choice(n_diffs, p=probs)
        centers.append(X[idx])
    
    for _ in range(50):
        # Assign
        dists = np.array([np.sum((X - c)**2, axis=1) for c in centers])
        labels = dists.argmin(axis=0)
        # Update
        new_centers = []
        for i in range(n_rows):
            mask = labels == i
            if mask.sum() > 0:
                new_centers.append(X[mask].mean(axis=0))
            else:
                new_centers.append(centers[i])
        if all(np.allclose(c1, c2) for c1, c2 in zip(centers, new_centers)):
            break
        centers = new_centers
    
    # Return the clusters: list of n_rows sets of difference values (1-indexed)
    clusters = [set() for _ in range(n_rows)]
    for i in range(n_diffs):
        clusters[labels[i]].add(i + 1)
    
    return clusters


def _assemble_row_from_diffs(target_diffs, k, max_scope, rng):
    """Given a set of target differences, try to find a valid row that uses exactly
       these differences. Uses backtracking."""
    target = set(target_diffs)
    if len(target) != k * (k + 1) // 2:
        return None
    
    # Backtracking: build row incrementally
    row = [0]
    
    def backtrack(pos, current_max):
        if pos == k + 1:
            # Verify all diffs are in target
            diffs = set()
            for i in range(pos):
                for j in range(i + 1, pos):
                    diffs.add(row[j] - row[i])
            return diffs == target
        
        # Try next value
        for x in range(current_max + 1, max_scope + 1):
            # Check new diffs don't exceed target
            new_diffs = [x - row[i] for i in range(pos)]
            valid = True
            for d in new_diffs:
                if d not in target:
                    valid = False
                    break
            if not valid:
                continue
            # Also check no duplicate within current row
            if len(set(new_diffs)) != len(new_diffs):
                continue
            row.append(x)
            if backtrack(pos + 1, x):
                return True
            row.pop()
        return False
    
    if backtrack(1, 0):
        return row[:]
    return None


def _spectral_approach(n_rows, k, max_scope, deadline, rng):
    """Use spectral clustering to partition differences, then assemble rows."""
    t_start = time.time()
    
    # Try several max_scope values
    for scope in range(max_scope, max_scope - 30, -1):
        if time.time() > deadline:
            break
        
        clusters = _build_conflict_graph(n_rows, k, scope)
        
        # Try to assemble rows from each cluster
        rows = []
        success = True
        for i in range(n_rows):
            # Add some flexibility: allow using diffs from other clusters if needed
            target = set(clusters[i])
            # If cluster size doesn't match, try to adjust
            needed = k * (k + 1) // 2
            if len(target) < needed:
                # Add smallest available diffs from other clusters
                all_other = set()
                for j in range(n_rows):
                    if j != i:
                        all_other.update(clusters[j])
                available = set(range(1, scope + 1)) - all_other
                for d in sorted(available):
                    if len(target) >= needed:
                        break
                    target.add(d)
            elif len(target) > needed:
                # Remove largest diffs (they're less likely to be needed)
                target = set(sorted(target)[:needed])
            
            row = _assemble_row_from_diffs(target, k, scope, rng)
            if row is None:
                success = False
                break
            rows.append(row)
        
        if success and _is_valid(rows):
            return rows
    
    return None


def _backtrack_build_row(used, k, max_val, rng, timeout=5.0):
    """Backtracking search for a single row avoiding used diffs."""
    t0 = time.time()
    row = [0]
    
    def backtrack(pos, current_max):
        if time.time() - t0 > timeout:
            return False
        if pos == k + 1:
            return True
        
        # Heuristic: try values that create "useful" diffs
        candidates = []
        for x in range(current_max + 1, max_val + 1):
            new_diffs = [x - row[i] for i in range(pos)]
            if any(d in used for d in new_diffs):
                continue
            if len(set(new_diffs)) != len(new_diffs):
                continue
            candidates.append(x)
        
        # Shuffle for diversity
        rng.shuffle(candidates)
        
        for x in candidates:
            new_diffs = [x - row[i] for i in range(pos)]
            for d in new_diffs:
                used.add(d)
            row.append(x)
            if backtrack(pos + 1, x):
                return True
            row.pop()
            for d in new_diffs:
                used.discard(d)
        return False
    
    if backtrack(1, 0):
        return row[:]
    return None


def _ils_with_backtrack(n_rows, k, deadline, rng):
    """Iterated Local Search with backtracking row construction."""
    best_rows = None
    best_scope = float('inf')
    
    # Get initial solution
    init = _greedy_baseline(n_rows, k)
    if len(init) == n_rows:
        best_rows = init
        best_scope = _scope(init)
    
    t_start = time.time()
    iterations = 0
    
    while time.time() < deadline - 10.0:
        iterations += 1
        
        # Perturbation: randomly remove one row and rebuild it
        if best_rows is None:
            break
        
        rows = [r[:] for r in best_rows]
        used = set()
        for r in rows:
            for d in _row_diffs(r):
                used.add(d)
        
        # Pick a row to rebuild
        row_idx = rng.randrange(n_rows)
        # Remove its diffs
        for d in _row_diffs(rows[row_idx]):
            used.discard(d)
        rows.pop(row_idx)
        
        # Try to rebuild with backtracking
        new_row = _backtrack_build_row(used, k, best_scope, rng, timeout=3.0)
        if new_row is not None:
            rows.insert(row_idx, new_row)
            new_scope = _scope(rows)
            if new_scope < best_scope and _is_valid(rows):
                best_scope = new_scope
                best_rows = rows
            else:
                # Restore
                for d in _row_diffs(rows[row_idx]):
                    used.add(d)
                rows.pop(row_idx)
        
        # Occasionally try full rebuild with different order
        if iterations % 10 == 0 and time.time() < deadline - 15.0:
            order = list(range(n_rows))
            rng.shuffle(order)
            used2 = set()
            new_rows = []
            success = True
            for idx in order:
                r = _backtrack_build_row(used2, k, best_scope, rng, timeout=2.0)
                if r is None:
                    success = False
                    break
                new_rows.append(r)
            if success and len(new_rows) == n_rows:
                # Reorder back
                final_rows = [None] * n_rows
                for i, idx in enumerate(order):
                    final_rows[idx] = new_rows[i]
                s = _scope(final_rows)
                if s < best_scope and _is_valid(final_rows):
                    best_scope = s
                    best_rows = final_rows
    
    return best_rows


def _polish_max_row(rows, k, used, deadline, rng):
    """Try to reduce the maximum entry by rebuilding the row with the largest value."""
    if rows is None:
        return None
    
    best = rows
    best_scope = _scope(best)
    
    while time.time() < deadline:
        # Find row with max value
        max_idx = max(range(len(best)), key=lambda i: best[i][-1])
        max_val = best[max_idx][-1]
        
        # Try to rebuild this row with a smaller max
        for target_max in range(max_val - 1, max_val - 20, -1):
            if time.time() > deadline:
                break
            
            # Remove current row's diffs
            cur_used = set(used)
            for d in _row_diffs(best[max_idx]):
                cur_used.discard(d)
            
            # Try to build a new row with max <= target_max
            new_row = _backtrack_build_row(cur_used, k, target_max, rng, timeout=2.0)
            if new_row is not None:
                candidate = [r[:] for r in best]
                candidate[max_idx] = new_row
                if _is_valid(candidate):
                    s = _scope(candidate)
                    if s < best_scope:
                        best = candidate
                        best_scope = s
                        # Update used
                        used = set()
                        for r in best:
                            for d in _row_diffs(r):
                                used.add(d)
                        break
    
    return best


def construct(n_rows, k):
    t0 = time.time()
    deadline = t0 + _DEADLINE
    rng = random.Random(42)
    
    # 1. Fast greedy baseline
    base = _greedy_baseline(n_rows, k)
    if len(base) != n_rows:
        # Fallback: simple construction
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = [x - a for a in row]
                if any(d in used for d in new_diffs):
                    continue
                for d in new_diffs:
                    used.add(d)
                row.append(x)
            rows.append(row)
        return rows
    
    best_rows = base
    best_scope = _scope(base)
    
    # 2. Spectral clustering approach
    if time.time() < deadline - 60.0:
        spec_result = _spectral_approach(n_rows, k, best_scope, deadline - 40.0, rng)
        if spec_result is not None and _is_valid(spec_result):
            s = _scope(spec_result)
            if s < best_scope:
                best_scope = s
                best_rows = spec_result
    
    # 3. ILS with backtracking
    if time.time() < deadline - 20.0:
        ils_result = _ils_with_backtrack(n_rows, k, deadline - 15.0, rng)
        if ils_result is not None and _is_valid(ils_result):
            s = _scope(ils_result)
            if s < best_scope:
                best_scope = s
                best_rows = ils_result
    
    # 4. Polish the max row
    if time.time() < deadline - 5.0:
        used = set()
        for r in best_rows:
            for d in _row_diffs(r):
                used.add(d)
        polished = _polish_max_row(best_rows, k, used, deadline - 2.0, rng)
        if polished is not None and _is_valid(polished):
            s = _scope(polished)
            if s < best_scope:
                best_scope = s
                best_rows = polished
    
    return best_rows
