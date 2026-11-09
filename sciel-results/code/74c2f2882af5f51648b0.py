"""Covering code search for K_q(n, radius) using MILP + local search polish."""
import time
import numpy as np
import random
import itertools
from scipy.optimize import milp, LinearConstraint, Bounds

_DEADLINE = 120.0  # seconds; evaluator kills at 150s, keep safety margin


def _idx_to_word(idx, n, q):
    """Convert a linear index to a word over {0,1,...} with n digits (base q)."""
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


def _build_coverage_matrix(q, n, radius):
    """
    Build a coverage matrix M of shape (total, total) where
    M[i, j] = 1 if word i covers word j (i.e., hamming(word_i, word_j) <= radius).
    """
    total = q ** n
    all_words = np.array([_idx_to_word(i, n, q) for i in range(total)], dtype=np.int8)
    
    # Compute pairwise Hamming distances
    # diff[i, j, k] = 1 if all_words[i][k] != all_words[j][k]
    # For efficiency, compute in chunks if needed, but for 729x729 it's fine
    diff = all_words[:, np.newaxis, :] != all_words[np.newaxis, :, :]
    dists = diff.sum(axis=2)
    
    M = (dists <= radius).astype(np.int8)
    return M, all_words


def _solve_milp(M, total, max_time=30.0):
    """
    Solve the set cover ILP: minimize sum(x_i) subject to M @ x >= 1.
    Returns a list of selected indices.
    """
    c = np.ones(total)
    A = M.astype(float)
    # Constraints: A @ x >= 1  =>  -A @ x <= -1
    # scipy.optimize.milp uses constraints in form: A_ub @ x <= b_ub
    # But we have lower bounds on coverage.
    # LinearConstraint(A, lb, ub) means lb <= A @ x <= ub
    lb = np.ones(total)
    ub = np.full(total, total)  # large enough
    
    constraint = LinearConstraint(A, lb, ub)
    bounds = Bounds(0, 1)
    
    start_time = time.time()
    try:
        res = milp(c=c, constraints=constraint, integrality=np.ones(total), bounds=bounds,
                   options={"time_limit": max_time, "mip_gap": 0.0})
        
        if res.status == 0 and res.x is not None:
            x = res.x
            selected = [i for i in range(total) if x[i] > 0.5]
            return selected
        else:
            return None
    except Exception:
        return None


def _greedy_construct(M, all_words, total):
    """
    Greedy set cover: at each step, pick the word that covers the most uncovered words.
    """
    uncovered = np.ones(total, dtype=bool)
    selected = []
    
    while uncovered.any():
        # Count how many uncovered words each candidate covers
        counts = M[:, uncovered].sum(axis=1)
        # Pick the one with max count
        best_idx = np.argmax(counts)
        if counts[best_idx] == 0:
            break
        selected.append(best_idx)
        uncovered[M[best_idx] == 1] = False
    
    return selected


def _verify_code(code_indices, M, total):
    """Verify that every word is covered."""
    if len(code_indices) == 0:
        return total == 0
    selected = np.zeros(total, dtype=bool)
    selected[code_indices] = True
    # Check if M @ selected >= 1 for all rows
    covered = (M @ selected) >= 1
    return covered.all()


def _local_search_polish(code_indices, M, all_words, total, time_limit=30.0):
    """
    Local search: try to remove redundant codewords, then try swaps.
    """
    code_indices = list(code_indices)
    start_time = time.time()
    
    # First, greedy deletion
    changed = True
    while changed:
        changed = False
        # Compute coverage counts
        counts = np.zeros(total, dtype=np.int32)
        for idx in code_indices:
            counts[M[idx] == 1] += 1
        
        for i in range(len(code_indices) - 1, -1, -1):
            idx = code_indices[i]
            cov = M[idx] == 1
            # Check if any word is covered only by this codeword
            if (counts[cov] == 1).any():
                continue
            # Remove it
            counts[cov] -= 1
            code_indices.pop(i)
            changed = True
            break
    
    # Now try swaps
    best_size = len(code_indices)
    
    # Precompute covers as boolean arrays for speed
    covers = [M[idx] == 1 for idx in code_indices]
    
    # Maintain counts
    counts = np.zeros(total, dtype=np.int32)
    for cov in covers:
        counts[cov] += 1
    
    code_set = set(code_indices)
    
    # Random swaps
    rng = random.Random(42)
    
    while time.time() - start_time < time_limit:
        if len(code_indices) <= best_size:
            best_size = len(code_indices)
        
        # Pick a random codeword to replace
        i = rng.randint(0, len(code_indices) - 1)
        idx_out = code_indices[i]
        
        # Pick a random word to swap in
        idx_in = rng.randint(0, total - 1)
        if idx_in in code_set:
            continue
        
        cov_out = covers[i]
        cov_in = M[idx_in] == 1
        
        # Check validity: after removing idx_out and adding idx_in, all counts >= 1
        # Risky words are those in cov_out
        # For w in cov_out: new_count = counts[w] - 1 + (1 if w in cov_in else 0)
        # We need new_count >= 1 for all w
        
        # Efficient check:
        # counts[cov_out] - 1 + cov_in[cov_out] >= 1
        # i.e., counts[cov_out] + cov_in[cov_out] >= 2
        
        risky = cov_out
        if not np.all(counts[risky] + cov_in[risky] >= 2):
            continue
        
        # Swap is valid
        # Update counts
        counts[cov_out] -= 1
        counts[cov_in] += 1
        
        # Update code
        code_indices[i] = idx_in
        covers[i] = cov_in
        code_set.remove(idx_out)
        code_set.add(idx_in)
        
        # Greedy deletion
        changed = True
        while changed:
            changed = False
            for k in range(len(code_indices) - 1, -1, -1):
                cov_k = covers[k]
                if (counts[cov_k] == 1).any():
                    continue
                # Remove k
                counts[cov_k] -= 1
                removed_idx = code_indices[k]
                code_indices.pop(k)
                covers.pop(k)
                code_set.remove(removed_idx)
                changed = True
                break
    
    return code_indices


def construct(q, n, radius):
    """Construct a covering code: choose as few words as possible so every word
    is within Hamming distance radius of some chosen word."""
    start_time = time.time()
    
    total = q ** n
    
    # Build coverage matrix
    M, all_words = _build_coverage_matrix(q, n, radius)
    
    # 1. Greedy construction as a starting point
    greedy_code = _greedy_construct(M, all_words, total)
    
    # 2. Local search polish on greedy solution
    polished_code = _local_search_polish(greedy_code, M, all_words, total, time_limit=10.0)
    
    best_code = polished_code
    best_size = len(best_code)
    
    # 3. Try MILP with remaining time
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 15  # leave 15s safety margin
    
    if remaining > 5:
        milp_code = _solve_milp(M, total, max_time=min(remaining, 60.0))
        if milp_code is not None:
            # Verify MILP solution
            if _verify_code(milp_code, M, total):
                # Polish MILP solution
                polished_milp = _local_search_polish(milp_code, M, all_words, total, time_limit=5.0)
                if len(polished_milp) < best_size:
                    best_code = polished_milp
                    best_size = len(best_code)
    
    # 4. Additional local search if time permits
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed - 10
    if remaining > 5 and best_size > 70:  # only if we haven't found a very good solution
        further_polished = _local_search_polish(best_code, M, all_words, total, time_limit=remaining)
        if len(further_polished) < best_size:
            best_code = further_polished
            best_size = len(best_code)
    
    # Convert indices back to words
    best_code_words = [all_words[i].tolist() for i in best_code]
    
    # Final verification
    if not _verify_code(best_code, M, total):
        # Fallback: use greedy solution which should be valid
        return [all_words[i].tolist() for i in greedy_code]
    
    return best_code_words
