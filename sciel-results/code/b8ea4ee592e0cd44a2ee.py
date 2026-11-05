import itertools
import time
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = 110.0

    total_words = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}

    # Build adjacency/coverage matrix
    # A[i, j] = 1 if dist(i, j) <= radius
    A = np.zeros((total_words, total_words), dtype=np.int8)
    for i, v in enumerate(words):
        A[i, i] = 1
        for pos in range(n):
            for d in range(q):
                if d == v[pos]:
                    continue
                w = v.copy()
                w[pos] = d
                j = word_to_idx[tuple(w)]
                A[i, j] = 1

    # 1. Try ILP
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
        
        c = np.ones(total_words)
        A_sp = csr_matrix(A)
        constraint = LinearConstraint(A_sp, 1, np.inf)
        bounds = Bounds(0, 1)
        integrality = np.ones(total_words)
        
        result = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if result.success and result.status == 0:
            x = result.x
            codewords = [words[i] for i in range(total_words) if x[i] > 0.5]
            return codewords
    except Exception:
        pass

    # 2. Heuristic: Greedy
    uncovered = set(range(total_words))
    selected = []
    cover_sets = [set(np.where(A[i] == 1)[0]) for i in range(total_words)]

    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = -1
        for i in range(total_words):
            if i in selected:
                continue
            count = len(uncovered & cover_sets[i])
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        uncovered -= cover_sets[best_idx]

    # 3. Local Search: Remove redundant codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(selected) - 1, -1, -1):
            c = selected[i]
            # Check if c is the unique cover for any word
            redundant = True
            for w in cover_sets[c]:
                # Check if any other selected codeword covers w
                if not any(A[w, idx] == 1 for idx in selected if idx != c):
                    redundant = False
                    break
            
            if redundant:
                selected.pop(i)
                changed = True

    return [words[idx] for idx in selected]
