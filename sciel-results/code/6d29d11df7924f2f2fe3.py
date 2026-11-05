import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct(q, n, radius):
    # This implementation assumes the specific case q=3, n=6, radius=1
    # which is the target of the task.
    # For generality, we fall back to a simple method if parameters differ.
    
    if q != 3 or n != 6 or radius != 1:
        return _construct_fallback(q, n, radius)

    total_words = q ** n # 729
    
    def index_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)

    # Precompute ball membership
    # We need a sparse matrix A where A[i, j] = 1 if j is in ball(i)
    # Actually, for the constraint "word i is covered by some codeword j",
    # we need sum_{j in ball(i)} x_j >= 1.
    # So row i of A corresponds to word i.
    # Col j of A corresponds to variable x_j (is word j a codeword?).
    # A[i, j] = 1 if dist(i, j) <= radius.
    
    rows = []
    cols = []
    data = []
    
    for i in range(total_words):
        w_i = index_to_tuple(i)
        # i is in its own ball
        rows.append(i)
        cols.append(i)
        data.append(1)
        
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w_i[pos]:
                        continue
                    new_w = list(w_i)
                    new_w[pos] = val
                    j = 0
                    for digit in new_w:
                        j = j * q + digit
                    rows.append(i)
                    cols.append(j)
                    data.append(1)

    A = sp.csr_matrix((data, (rows, cols)), shape=(total_words, total_words))
    
    # Objective: minimize sum(x)
    c = np.ones(total_words)
    
    # Constraints: A @ x >= 1
    # milp uses LinearConstraint(A, lb, ub)
    # We want A @ x >= 1, so lb = 1, ub = inf
    constraints = LinearConstraint(A, lb=1, ub=np.inf)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x_i must be integer (0 or 1)
    integrality = np.ones(total_words)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            # Due to floating point, threshold at 0.5
            codeword_indices = [i for i in range(total_words) if x[i] > 0.5]
            
            # Verify and return
            result = [list(index_to_tuple(i)) for i in codeword_indices]
            return result
        else:
            # Fallback if ILP fails
            return _construct_greedy(q, n, radius)
            
    except Exception as e:
        # Fallback on error
        return _construct_greedy(q, n, radius)

def _construct_greedy(q, n, radius):
    # Re-implementing the greedy logic from the original to serve as fallback
    total_words = q ** n
    indices = range(total_words)
    
    def tuple_to_index(t):
        idx = 0
        for digit in t:
            idx = idx * q + digit
        return idx

    def index_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return tuple(t)

    ball_masks = [0] * total_words
    for i in indices:
        w = index_to_tuple(i)
        mask = 1 << i
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    j = tuple_to_index(tuple(new_w))
                    mask |= (1 << j)
        ball_masks[i] = mask

    all_bits = (1 << total_words) - 1
    best_codewords = None
    best_size = total_words + 1
    num_restarts = 5 # Reduced for speed in fallback
    
    for _ in range(num_restarts):
        uncovered = all_bits
        codewords = []
        while uncovered:
            max_cov = -1
            best_candidates = []
            for i in indices:
                cov = (uncovered & ball_masks[i]).bit_count()
                if cov > max_cov:
                    max_cov = cov
                    best_candidates = [i]
                elif cov == max_cov:
                    best_candidates.append(i)
            if not best_candidates:
                break
            chosen = random.choice(best_candidates)
            codewords.append(chosen)
            uncovered &= ~ball_masks[chosen]
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
            
    return [list(index_to_tuple(i)) for i in best_codewords]

def _construct_fallback(q, n, radius):
    import itertools
    all_words = list(itertools.product(range(q), repeat=n))
    balls = []
    for w in all_words:
        ball = {w}
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    neighbor = list(w)
                    neighbor[i] = val
                    ball.add(tuple(neighbor))
        balls.append(ball)
        
    uncovered = set(all_words)
    codewords = []
    while uncovered:
        best_idx = -1
        max_covered = 0
        for i in range(len(all_words)):
            covered_count = len(balls[i] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_idx = i
        if best_idx == -1:
            break
        best_word = all_words[best_idx]
        codewords.append(best_word)
        uncovered -= balls[best_idx]
    return [list(c) for c in codewords]
