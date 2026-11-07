import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = P['n']
    full = (1 << n) - 1

    # adjacency bitmask
    adj = [0] * n
    es = P['edges']
    for a, b in es:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute for each subset: is it a clique? independent set?
    N = 1 << n
    is_clique = [False] * N
    is_indep = [False] * N
    is_clique[0] = True
    is_indep[0] = True
    for mask in range(1, N):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True

    # Collect all valid sets (non-empty)
    valid_sets = []
    for m in range(1, N):
        if is_clique[m] or is_indep[m]:
            valid_sets.append(m)
            
    K = len(valid_sets)
    
    # If no valid sets found (should not happen), fallback
    if K == 0:
        return [[i] for i in range(n)]

    # Build ILP
    # Variables y_k for k in 0..K-1
    # Minimize sum(y_k)
    c = np.ones(K)
    
    # Constraints: for each vertex v, sum(y_k for k where v in set_k) == 1
    # A_eq is n x K
    A_eq = lil_matrix((n, K), dtype=np.int8)
    
    for k, m in enumerate(valid_sets):
        # Set bits in column k for vertices in m
        mm = m
        while mm:
            lsb = mm & (-mm)
            v = lsb.bit_length() - 1
            A_eq[v, k] = 1
            mm ^= lsb
            
    A_eq = A_eq.tocsr()
    
    # Bounds: 0 <= y_k <= 1
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(K)
    
    try:
        res = milp(c=c, constraints=LinearConstraint(A_eq, lb=np.ones(n), ub=np.ones(n)), integrality=integrality, bounds=bounds)
        if res.status == 0:
            # Extract solution
            y = res.x
            parts_masks = []
            for k in range(K):
                if y[k] > 0.5:
                    parts_masks.append(valid_sets[k])
            
            # Convert to lists
            final_parts = []
            for pm in parts_masks:
                vertices = []
                for i in range(n):
                    if (pm >> i) & 1:
                        vertices.append(i)
                final_parts.append(vertices)
                
            # Verify
            if len(final_parts) > 0:
                # Check coverage
                covered = [0]*n
                for part in final_parts:
                    for v in part:
                        covered[v] += 1
                if all(c == 1 for c in covered):
                    return final_parts
            # If ILP solution is invalid for some reason, fall through
        else:
            # ILP failed, fall through to DP
            pass
    except Exception:
        pass

    # Fallback: DP
    valid_submasks = [[] for _ in range(N)]
    for m in range(N):
        s = m
        while s:
            if is_clique[s] or is_indep[s]:
                valid_submasks[m].append(s)
            s = (s - 1) & m

    INF = 10**9
    f = [INF] * N
    f[0] = 0

    for mask in range(1, N):
        if time.time() - t0 > DEADLINE:
            break
        best = INF
        for s in valid_submasks[mask]:
            val = f[mask ^ s] + 1
            if val < best:
                best = val
        f[mask] = best

    if f[full] != INF:
        parts = []
        mask = full
        while mask:
            found = False
            for s in valid_submasks[mask]:
                if f[mask ^ s] + 1 == f[mask]:
                    parts.append(s)
                    mask ^= s
                    found = True
                    break
            if not found:
                break
            if time.time() - t0 > DEADLINE:
                break
        
        if len(parts) == f[full]:
            final_parts = []
            for p in parts:
                vertices = []
                for i in range(n):
                    if (p >> i) & 1:
                        vertices.append(i)
                final_parts.append(vertices)
            return final_parts

    # Fallback: Greedy
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            part_clique = True
            part_indep = True
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    u, w = part[i], part[j]
                    if (adj[u] >> w) & 1:
                        part_indep = False
                    else:
                        part_clique = False
            if part_clique:
                ok = True
                for u in part:
                    if not ((adj[v] >> u) & 1):
                        ok = False
                        break
                if ok:
                    part.append(v)
                    placed = True
                    break
            elif part_indep:
                ok = True
                for u in part:
                    if (adj[v] >> u) & 1:
                        ok = False
                        break
                if ok:
                    part.append(v)
                    placed = True
                    break
        if not placed:
            parts.append([v])
    
    return parts
