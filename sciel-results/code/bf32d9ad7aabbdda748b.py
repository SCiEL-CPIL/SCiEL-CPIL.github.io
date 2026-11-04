"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    # p[i] = location of facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    n = len(a)
    # b_perm[i,j] = b[p[i], p[j]]
    idx = p
    b_perm = b[np.ix_(idx, idx)]
    return np.int64(np.sum(a * b_perm))


def _swap_cost_delta(a, b, p, i, j):
    """
    Compute the change in QAP cost when swapping positions of facilities i and j
    in permutation p.
    
    p[i] and p[j] are the locations of facilities i and j.
    After swap: facility i goes to p[j], facility j goes to p[i].
    
    The cost change only involves terms where i or j is one of the two indices
    in the sum sum_{u,v} a[u,v] * b[p[u], p[v]].
    
    Old contribution involving i or j:
    - sum_v a[i,v] * b[p[i], p[v]]  +  sum_v a[j,v] * b[p[j], p[v]]
    - sum_u a[u,i] * b[p[u], p[i]]  +  sum_u a[u,j] * b[p[u], p[j]]
    - minus a[i,i]*b[p[i],p[i]] and a[j,j]*b[p[j],p[j]] (counted twice above, subtract once)
    - minus a[i,j]*b[p[i],p[j]] and a[j,i]*b[p[j],p[i]] (these appear in both rows)
    
    Actually let's be more careful. The full cost is:
    C = sum_{u,v} a[u,v] * B[p[u], p[v]] where B = b.
    
    Terms that change when we swap p[i] and p[j]:
    Only terms where u==i or v==i or u==j or v==j change.
    
    Let's define:
    li = p[i] (old location of i)
    lj = p[j] (old location of j)
    
    Old terms involving i or j:
    T_old = sum_v a[i,v]*b[li, p[v]] + sum_v a[j,v]*b[lj, p[v]] 
          + sum_u a[u,i]*b[p[u], li] + sum_u a[u,j]*b[p[u], lj]
          - a[i,i]*b[li,li] - a[j,j]*b[lj,lj]  (diagonal counted twice)
          - a[i,j]*b[li, lj] - a[j,i]*b[lj, li]  (cross terms counted twice)
    
    Wait, this is getting complex. Let me use a cleaner formulation.
    
    The cost can be written as:
    C = sum_{u,v} a[u,v] * b[p[u], p[v]]
    
    When we swap p[i] <-> p[j], the new cost is:
    C' = sum_{u,v} a[u,v] * b'[p'[u], p'[v]]
    where p'[i]=p[j], p'[j]=p[i], p'[k]=p[k] for k!=i,j.
    
    C' - C = sum_{u,v} a[u,v] * (b'[p'[u], p'[v]] - b[p[u], p[v]])
    
    The only terms that differ are those where u in {i,j} or v in {i,j}.
    
    Let's separate:
    delta = C' - C
    
    For terms where neither u nor j is in {i,j}: no change.
    
    Terms where u==i, v!=j:
      Old: a[i,v] * b[p[i], p[v]]
      New: a[i,v] * b[p[j], p[v]]
      Delta: a[i,v] * (b[p[j], p[v]] - b[p[i], p[v]])
    
    Terms where u==j, v!=i:
      Old: a[j,v] * b[p[j], p[v]]
      New: a[j,v] * b[p[i], p[v]]
      Delta: a[j,v] * (b[p[i], p[v]] - b[p[j], p[v]])
    
    Terms where v==i, u!=j:
      Old: a[u,i] * b[p[u], p[i]]
      New: a[u,i] * b[p[u], p[j]]
      Delta: a[u,i] * (b[p[u], p[j]] - b[p[u], p[i]])
    
    Terms where v==j, u!=i:
      Old: a[u,j] * b[p[u], p[j]]
      New: a[u,j] * b[p[u], p[i]]
      Delta: a[u,j] * (b[p[u], p[i]] - b[p[u], p[j]])
    
    Terms where u==i, v==j:
      Old: a[i,j] * b[p[i], p[j]]
      New: a[i,j] * b[p[j], p[i]]
      Delta: a[i,j] * (b[p[j], p[i]] - b[p[i], p[j]])
    
    Terms where u==j, v==i:
      Old: a[j,i] * b[p[j], p[i]]
      New: a[j,i] * b[p[i], p[j]]
      Delta: a[j,i] * (b[p[i], p[j]] - b[p[j], p[i]])
    
    Terms where u==i, v==i:
      Old: a[i,i] * b[p[i], p[i]]
      New: a[i,i] * b[p[j], p[j]]
      Delta: a[i,i] * (b[p[j], p[j]] - b[p[i], p[i]])
    
    Terms where u==j, v==j:
      Old: a[j,j] * b[p[j], p[j]]
      New: a[j,j] * b[p[i], p[i]]
      Delta: a[j,j] * (b[p[i], p[i]] - b[p[j], p[j]])
    
    So total delta:
    delta = sum_{v != j} a[i,v] * (b[p[j], p[v]] - b[p[i], p[v]])
          + sum_{v != i} a[j,v] * (b[p[i], p[v]] - b[p[j], p[v]])
          + sum_{u != j} a[u,i] * (b[p[u], p[j]] - b[p[u], p[i]])
          + sum_{u != i} a[u,j] * (b[p[u], p[i]] - b[p[u], p[j]])
          + a[i,j] * (b[p[j], p[i]] - b[p[i], p[j]])
          + a[j,i] * (b[p[i], p[j]] - b[p[j], p[i]])
          + a[i,i] * (b[p[j], p[j]] - b[p[i], p[i]])
          + a[j,j] * (b[p[i], p[i]] - b[p[j], p[j]])
    
    This is still O(n). Let's precompute row and column sums to make it O(n).
    """
    li = p[i]
    lj = p[j]
    
    # For efficiency, we can precompute some quantities
    # But for now, let's just compute directly - it's O(n) which is acceptable
    
    delta = 0
    
    # sum_v a[i,v] * (b[lj, p[v]] - b[li, p[v]])
    # = sum_v a[i,v] * b[lj, p[v]] - sum_v a[i,v] * b[li, p[v]]
    # We can precompute for each location l: S_row[i, l] = sum_v a[i,v] * b[l, p[v]]
    
    # Actually, let's just compute it. O(n) per swap is fine.
    
    # Precompute arrays for speed
    # row_i = a[i, :]
    # col_i = a[:, i]
    
    # delta contribution from facility i moving from li to lj:
    # Old row i: sum_v a[i,v] * b[li, p[v]]
    # New row i: sum_v a[i,v] * b[lj, p[v]]
    # Delta row i: sum_v a[i,v] * (b[lj, p[v]] - b[li, p[v]])
    
    # Similarly for column i, row j, column j
    
    # Let's compute each part
    row_i = a[i, :]
    row_j = a[j, :]
    col_i = a[:, i]
    col_j = a[:, j]
    
    # Part 1: Row i change
    # sum_v row_i[v] * (b[lj, p[v]] - b[li, p[v]])
    part1 = np.dot(row_i, b[lj, p] - b[li, p])
    
    # Part 2: Row j change  
    # sum_v row_j[v] * (b[li, p[v]] - b[lj, p[v]])
    part2 = np.dot(row_j, b[li, p] - b[lj, p])
    
    # Part 3: Col i change
    # sum_u col_i[u] * (b[p[u], lj] - b[p[u], li])
    part3 = np.dot(col_i, b[p, lj] - b[p, li])
    
    # Part 4: Col j change
    # sum_u col_j[u] * (b[p[u], li] - b[p[u], lj])
    part4 = np.dot(col_j, b[p, li] - b[p, lj])
    
    # Diagonal and cross terms
    part5 = a[i, j] * (b[lj, li] - b[li, lj])
    part6 = a[j, i] * (b[li, lj] - b[lj, li])
    part7 = a[i, i] * (b[lj, lj] - b[li, li])
    part8 = a[j, j] * (b[li, li] - b[lj, lj])
    
    delta = part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8
    
    return np.int64(delta)


def solve(a, b):
    n = len(a)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Initial permutation: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Multi-start local search with first-improvement
    np.random.seed(42)
    
    while time.time() < deadline:
        # Try swaps: first-improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle the order of i for diversification
            order = np.random.permutation(n)
            for i in order:
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    delta = _swap_cost_delta(a, b, p, i, j)
                    if delta < 0:
                        # Accept the swap
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
