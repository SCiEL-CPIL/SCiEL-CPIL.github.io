"""MILP-based covering code construction with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin


def _idx_to_word(idx, q, n):
    """Convert a linear index to a word over {0,...,q-1} with n digits."""
    digits = [0] * n
    for i in range(n):
        digits[i] = idx % q
        idx //= q
    return digits


def _word_to_idx(word, q, n):
    """Convert a word to a linear index."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _build_distance_matrix(q, n, radius):
    """Build a boolean matrix: dist_matrix[i,j] = True iff word i is within radius of word j.
    Shape: (total, total) where total = q^n."""
    total = q ** n
    # Generate all words as a 2D array
    words_arr = np.array([_idx_to_word(i, q, n) for i in range(total)], dtype=np.int8)
    # Compute pairwise Hamming distances efficiently
    # dist_matrix[i,j] = sum(words_arr[i] != words_arr[j]) <= radius
    # Use broadcasting: diff shape (total, total, n)
    diff = words_arr[:, np.newaxis, :] != words_arr[np.newaxis, :, :]  # (total, total, n)
    dists = diff.sum(axis=2)  # (total, total)
    return (dists <= radius).astype(np.int8)


def _solve_milp(dist_matrix, time_limit):
    """Solve the set-covering ILP using scipy.optimize.milp.
    Minimize sum(x_j) subject to sum_j A[i,j] * x_j >= 1 for all i.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None
    
    m, k = dist_matrix.shape  # m words, k candidates
    
    # Objective: minimize sum of x_j
    c = np.ones(k, dtype=float)
    
    # Constraints: for each word i, sum_j A[i,j] * x_j >= 1
    # LinearConstraint(A, lb=ones(m), ub=inf)
    A = dist_matrix.astype(float)
    lb = np.ones(m, dtype=float)
    ub = np.full(m, np.inf, dtype=float)
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(lb=np.zeros(k), ub=np.ones(k))
    
    # Integer variables
    integrality = np.ones(k, dtype=int)
    
    start = time.time()
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit, "presolve": True}
    )
    
    elapsed = time.time() - start
    if result.status != 0 and result.status != 2:
        # status 0 = optimal, 2 = time limit reached (still has a solution)
        if result.x is None:
            return None
        # Check if we have a feasible integer solution
        if not np.all(np.isfinite(result.x)):
            return None
    
    # Extract selected codeword indices
    selected = np.where(result.x > 0.5)[0]
    return list(selected)


def _greedy_fallback(q, n, radius, deadline):
    """Fast greedy fallback if MILP fails or is too slow."""
    total = q ** n
    words_arr = np.array([_idx_to_word(i, q, n) for i in range(total)], dtype=np.int8)
    
    covered = np.zeros(total, dtype=bool)
    code = []
    
    while not covered.all():
        # Find the word that covers the most uncovered words
        diff = words_arr[:, np.newaxis, :] != words_arr[np.newaxis, :, :]
        dists = diff.sum(axis=2)
        coverable = (dists <= radius)  # (total, total): coverable[i,j] = word i covers word j
        
        # For each candidate j, count newly covered
        new_cover = (coverable & ~covered[np.newaxis, :]).sum(axis=1)
        best_j = int(np.argmax(new_cover))
        
        if new_cover[best_j] == 0:
            break
        
        code.append(best_j)
        covered |= coverable[:, best_j]
        
        if time.time() > deadline:
            break
    
    return code


def _direct_sum_tetracode(q, n, radius):
    """Direct sum of ternary tetracode with all tails (for q=3, n>=4)."""
    if q != 3:
        return [[0] * n]
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n == 4:
        return [list(h) for h in tetracode]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _verify_code(code_words, q, n, radius):
    """Verify that every word is within radius of some codeword."""
    total = q ** n
    if total > 100000:
        # Sample-based for very large spaces
        rng = np.random.default_rng(42)
        for _ in range(5000):
            w = tuple(int(x) for x in rng.integers(0, q, size=n))
            found = any(_hamming(w, cw) <= radius for cw in code_words)
            if not found:
                return False
        return True
    # Full verification
    for idx in range(total):
        w = _idx_to_word(idx, q, n)
        found = False
        for cw in code_words:
            if _hamming(w, cw) <= radius:
                found = True
                break
        if not found:
            return False
    return True


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    deadline = start_time + _DEADLINE - 10.0  # 10s safety margin
    
    total = q ** n
    
    # Fallback: direct sum construction
    fallback_code = _direct_sum_tetracode(q, n, radius)
    
    # For small enough spaces, use MILP
    if total <= 10000:
        try:
            # Build distance matrix
            dist_matrix = _build_distance_matrix(q, n, radius)
            
            # Time remaining for MILP
            elapsed = time.time() - start_time
            time_for_milp = max(5.0, _DEADLINE - 15.0 - elapsed)
            
            selected = _solve_milp(dist_matrix, time_for_milp)
            
            if selected is not None and len(selected) > 0:
                code_words = [_idx_to_word(i, q, n) for i in selected]
                # Verify
                if _verify_code(code_words, q, n, radius):
                    return code_words
        except Exception:
            pass
        
        # Greedy fallback
        try:
            greedy_code = _greedy_fallback(q, n, radius, deadline)
            if greedy_code:
                code_words = [_idx_to_word(i, q, n) for i in greedy_code]
                if _verify_code(code_words, q, n, radius) and len(code_words) < len(fallback_code):
                    return code_words
        except Exception:
            pass
    
    # Return fallback
    return fallback_code

# EVOLVE-BLOCK-END
