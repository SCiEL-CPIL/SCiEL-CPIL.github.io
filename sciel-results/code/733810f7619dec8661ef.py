import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other parameters
    import itertools
    total = q ** n
    if total <= 10000:
        all_words = [list(c) for c in itertools.product(range(q), repeat=n)]
        uncovered = set(range(len(all_words)))
        codewords = []

        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)

        while uncovered:
            if time.time() > deadline - 5.0:
                break
            best_idx = -1
            best_count = -1
            for i in range(len(all_words)):
                count = 0
                for j in uncovered:
                    if hamming_dist(all_words[i], all_words[j]) <= radius:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            codewords.append(all_words[best_idx])
            new_uncovered = set()
            for j in uncovered:
                if hamming_dist(all_words[best_idx], all_words[j]) > radius:
                    new_uncovered.add(j)
            uncovered = new_uncovered

        return codewords
    else:
        return [list(c) for c in itertools.product(range(q), repeat=n)]


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute base-3 weights
    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute balls: balls[w, 0..16] are the 17 words within Hamming distance 1 of w
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Helper: convert index to digits
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits

    # Try MILP first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp

        # Constraint matrix A: A[i, j] = 1 if word i is covered by codeword j
        # Since distance is symmetric, A[i, j] = 1 if j in balls[i]
        # We can build the sparse matrix using the balls array
        rows = []
        cols = []
        for i in range(total):
            for j in balls[i]:
                rows.append(i)
                cols.append(j)
        
        data = np.ones(len(rows))
        A = sp.csr_matrix((data, (rows, cols)), shape=(total, total))
        
        # Minimize sum(x)
        c = np.ones(total)
        
        # Constraints: A @ x >= 1
        # milp requires A @ x <= b or A @ x >= b via LinearConstraint
        # LinearConstraint(A, lb=1, ub=np.inf)
        
        constraint = LinearConstraint(A, lb=np.ones(total), ub=np.inf)
        bounds = Bounds(lb=0, ub=1)
        
        # Time limit: leave 5s for conversion and safety
        time_limit = max(1, int(deadline - time.time() - 5))
        
        res = milp(c=c, constraints=constraint, integrality=np.ones(total), bounds=bounds, options={'time_limit': time_limit})
        
        if res.status == 0 or res.status == 1: # Optimal or Time Limit exceeded with feasible solution
            # res.x is the solution
            # Round to nearest integer if necessary (should be binary)
            x = np.round(res.x).astype(int)
            selected_indices = np.where(x == 1)[0]
            
            if len(selected_indices) > 0:
                # Verify validity
                cover_count = np.zeros(total, dtype=np.int32)
                for c_idx in selected_indices:
                    cover_count[balls[c_idx]] += 1
                if np.all(cover_count > 0):
                    return [idx_to_digits(cw) for cw in selected_indices]
                else:
                    # If invalid, fall through to heuristic
                    pass
            else:
                # No solution found, fall through
                pass
    except Exception:
        pass

    # Fallback: Heuristic approach (Improved from original)
    # ... (Keep the original heuristic logic but maybe speed it up or improve initialization)
    
    # Original heuristic logic
    ball_membership = np.zeros((total, total), dtype=np.int8)
    for w in range(total):
        ball_membership[w, balls[w]] = 1

    best_size = total
    best_set = None

    for iteration in range(50):
        if time.time() > deadline - 10.0:
            break

        np.random.seed(iteration * 1000 + 42)
        
        m_init = 200
        selected = []
        if m_init > 0:
            first = np.random.randint(total)
            selected.append(first)
            
            for _ in range(m_init - 1):
                if time.time() > deadline - 10.0:
                    break
                best_cand = -1
                best_min_dist = -1
                for _ in range(50):
                    cand = np.random.randint(total)
                    if cand in selected:
                        continue
                    min_d = ball_size + 1
                    for s in selected:
                        d = 0
                        for pos in range(n):
                            dw = weights[pos]
                            dc = (cand // dw) % q
                            ds = (s // dw) % q
                            if dc != ds:
                                d += 1
                                if d >= min_d:
                                    break
                        if d > best_min_dist:
                            best_min_dist = d
                            best_cand = cand
                if best_cand != -1:
                    selected.append(best_cand)
        
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[balls[c]] += 1
        
        uncovered_mask = (cover_count == 0)
        
        if np.any(uncovered_mask):
            while np.any(uncovered_mask):
                if time.time() > deadline - 10.0:
                    break
                
                hardness = np.zeros(total, dtype=np.float64)
                nonzero = cover_count > 0
                hardness[nonzero] = 1.0 / cover_count[nonzero].astype(np.float64)
                hardness[~nonzero] = 1.0
                
                scores = hardness[balls].sum(axis=1)
                uncov_in_ball = uncovered_mask[balls].sum(axis=1)
                scores += uncov_in_ball * 100.0
                
                max_score = np.max(scores)
                candidates = np.where(scores == max_score)[0]
                best_idx = candidates[np.random.randint(len(candidates))]
                
                selected.append(best_idx)
                cover_count[balls[best_idx]] += 1
                uncovered_mask = (cover_count == 0)
        
        if not np.any(uncovered_mask):
            # Local Search
            changed = True
            while changed and time.time() < deadline - 5.0:
                changed = False
                new_selected = []
                for c in selected:
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        changed = True
                    else:
                        new_selected.append(c)
                selected = new_selected
            
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                new_selected = []
                for c in selected:
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        improved = True
                        continue
                    
                    critical_mask = (cover_count[balls[c]] == 1)
                    critical_words = balls[c][critical_mask]
                    
                    if len(critical_words) == 0:
                        cover_count[balls[c]] -= 1
                        improved = True
                        continue
                    
                    candidate_mask = np.ones(total, dtype=bool)
                    for cw in critical_words:
                        candidate_mask &= (ball_membership[:, cw] == 1)
                    
                    valid_candidates = np.where(candidate_mask)[0]
                    valid_candidates = valid_candidates[valid_candidates != c]
                    
                    if len(valid_candidates) > 0:
                        c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                        cover_count[balls[c]] -= 1
                        cover_count[balls[c_prime]] += 1
                        new_selected.append(c_prime)
                        improved = True
                    else:
                        new_selected.append(c)
                
                selected = new_selected
            
            if len(selected) < best_size:
                best_size = len(selected)
                best_set = selected[:]
                if best_size <= 486:
                    break

    if best_set is None:
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)
        
        while np.any(uncovered_mask) and time.time() < deadline - 5.0:
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        
        best_set = selected
    
    result = [idx_to_digits(cw) for cw in best_set]
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    
    return unique_result
