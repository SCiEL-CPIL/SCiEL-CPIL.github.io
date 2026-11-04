"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 140  # safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank using randomized search
    # Target: find rank r < 48
    # We'll search for r in a reasonable range
    target_rank = 44  # Try to beat 48
    
    if time.time() - start_time < timeout - 20:
        result = _search_rank(n, m, p, target_rank, timeout - 20 - (time.time() - start_time))
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _search_rank(n, m, p, target_rank, timeout):
    """Try to find a decomposition of given target rank."""
    start = time.time()
    rng = random.Random(42)
    
    num_u = n * m  # 6
    num_v = m * p  # 24
    num_w = p * n  # 16
    
    # Target tensor: T[i,j,k] = 1 for all valid i,j,k
    target = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target.add((i, j, k))
    
    def build_tensor(terms):
        """terms: list of (u, v, w) where u is list of num_u, etc."""
        T = {}
        for u, v, w in terms:
            for a in range(num_u):
                if u[a] == 0:
                    continue
                for b in range(num_v):
                    if v[b] == 0:
                        continue
                    for c in range(num_w):
                        if w[c] == 0:
                            continue
                        key = (a, b, c)
                        T[key] = T.get(key, 0) + u[a] * v[b] * w[c]
        return T
    
    def score(terms):
        T = build_tensor(terms)
        s = 0
        for e in target:
            if T.get(e, 0) != 1:
                s += 1
        for e, val in T.items():
            if val != 0 and e not in target:
                s += 1
        return s
    
    # Initialize with schoolbook then truncate
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * num_u
                v = [0] * num_v
                w = [0] * num_w
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                schoolbook_terms.append((u, v, w))
    
    # Start with random terms
    best_terms = None
    best_sc = float('inf')
    
    max_iter = 50000
    for restart in range(20):
        if time.time() - start > timeout:
            break
        
        # Random initialization
        terms = []
        for _ in range(target_rank):
            u = [rng.choice([-1, 0, 1]) for _ in range(num_u)]
            v = [rng.choice([-1, 0, 1]) for _ in range(num_v)]
            w = [rng.choice([-1, 0, 1]) for _ in range(num_w)]
            terms.append((u, v, w))
        
        sc = score(terms)
        
        # Local search
        for it in range(max_iter):
            if time.time() - start > timeout:
                break
            if sc == 0:
                break
            
            # Pick a random term and mutate
            idx = rng.randrange(target_rank)
            which = rng.choice(['u', 'v', 'w'])
            pos = rng.choice([0, 1, 2])  # which vector
            dim = [num_u, num_v, num_w][pos]
            
            # Mutate: change one entry
            entry = rng.randrange(dim)
            old_terms = [t[:] for t in terms]
            vec = list(terms[idx][pos])
            vec[entry] = rng.choice([-1, 0, 1])
            terms[idx] = (list(terms[idx][0]), list(terms[idx][1]), list(terms[idx][2]))
            terms[idx] = (terms[idx][0], terms[idx][1], terms[idx][2])
            if pos == 0:
                terms[idx] = (vec, terms[idx][1], terms[idx][2])
            elif pos == 1:
                terms[idx] = (terms[idx][0], vec, terms[idx][2])
            else:
                terms[idx] = (terms[idx][0], terms[idx][1], vec)
            
            new_sc = score(terms)
            if new_sc <= sc:
                sc = new_sc
                if sc < best_sc:
                    best_sc = sc
                    best_terms = [t[:] for t in terms]
            else:
                terms = old_terms
        
        if sc == 0:
            break
    
    if best_sc == 0 and best_terms is not None:
        U = [[0] * target_rank for _ in range(num_u)]
        V = [[0] * target_rank for _ in range(num_v)]
        W = [[0] * target_rank for _ in range(num_w)]
        for c, (u, v, w) in enumerate(best_terms):
            for a in range(num_u):
                U[a][c] = u[a]
            for b in range(num_v):
                V[b][c] = v[b]
            for c2 in range(num_w):
                W[c2][c] = w[c2]
        return U, V, W
    
    return None
# EVOLVE-BLOCK-END

