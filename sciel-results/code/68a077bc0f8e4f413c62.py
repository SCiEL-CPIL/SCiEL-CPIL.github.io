import itertools
import time
import random
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

N = 16
_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute(all_points):
    n = len(all_points)
    dist_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

def _construct_greedy(all_indices, dist_matrix, rng):
    """
    Fallback greedy construction.
    """
    order = list(all_indices)
    rng.shuffle(order)
    chosen = []
    chosen_set = set()
    
    # Precompute dist_lists for greedy
    n = len(all_points)
    dist_lists = [dict() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            d = dist_matrix[i][j]
            if d not in dist_lists[i]:
                dist_lists[i][d] = []
            dist_lists[i][d].append(j)
            
    def _can_add(x, chosen_set, chosen_list):
        if not chosen_list:
            return True
        seen_dists = set()
        for p in chosen_list:
            d = dist_matrix[x][p]
            if d in seen_dists:
                return False
            seen_dists.add(d)
        for p in chosen_list:
            d_xp = dist_matrix[x][p]
            if d_xp in dist_lists[p]:
                for q in dist_lists[p][d_xp]:
                    if q in chosen_set:
                        return False
        return True

    for p in order:
        if _can_add(p, chosen_set, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen, chosen_set

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    M = len(all_points)
    
    dist_matrix = _precompute(all_points)
    
    # 1. Find all isosceles triples
    # To speed up, we can check conditions.
    # A triple (i,j,k) is isosceles if d(i,j)==d(i,k) or d(i,j)==d(j,k) or d(i,k)==d(j,k).
    
    iso_triples = []
    # Optimization: iterate i, j, k.
    # 2.7M iterations is fine.
    for i in range(M):
        for j in range(i + 1, M):
            dij = dist_matrix[i, j]
            for k in range(j + 1, M):
                dik = dist_matrix[i, k]
                djk = dist_matrix[j, k]
                if dij == dik or dij == djk or dik == djk:
                    iso_triples.append((i, j, k))
                    
    if time.time() > _DEADLINE - 60:
        # Not enough time for MILP setup/solve, use greedy
        chosen, _ = _construct_greedy(all_indices, dist_matrix, random.Random(42))
        return [all_points[i] for i in chosen]

    # 2. Identify pairs P for maximality
    # P is set of {i,j} such that exists u with {i,j,u} isosceles.
    # We can derive this from iso_triples.
    # If (i,j,k) is iso, then pairs (i,j), (i,k), (j,k) are "blocked" by the third point?
    # No, E_u is pairs {i,j} such that {i,j,u} is iso.
    # So for each iso triple (i,j,k):
    #   u=i, pair {j,k}
    #   u=j, pair {i,k}
    #   u=k, pair {i,j}
    
    pair_to_idx = {}
    pairs_list = []
    
    def get_pair_idx(i, j):
        if i > j: i, j = j, i
        key = (i, j)
        if key not in pair_to_idx:
            pair_to_idx[key] = len(pairs_list)
            pairs_list.append(key)
        return pair_to_idx[key]
        
    E_u = [set() for _ in range(M)] # E_u stores indices of pairs in P
    
    for i, j, k in iso_triples:
        # For u=i, pair {j,k}
        idx_jk = get_pair_idx(j, k)
        E_u[i].add(idx_jk)
        # For u=j, pair {i,k}
        idx_ik = get_pair_idx(i, k)
        E_u[j].add(idx_ik)
        # For u=k, pair {i,j}
        idx_ij = get_pair_idx(i, j)
        E_u[k].add(idx_ij)
        
    K = len(pairs_list)
    V = M + K
    
    # 3. Build MILP
    # Variables: x[0..M-1], z[M..M+K-1]
    
    # Constraints
    # 1. Isosceles triple constraints: x_i + x_j + x_k <= 2
    # 2. Maximality: x_u + sum(z_{ij} for {ij} in E_u) >= 1  => -x_u - sum(z) <= -1
    # 3. Linearization for each pair {i,j} in P (index p in 0..K-1, var index M+p):
    #    z_p <= x_i  => -x_i + z_p <= 0
    #    z_p <= x_j  => -x_j + z_p <= 0
    #    z_p >= x_i + x_j - 1 => x_i + x_j - z_p <= 1
    
    rows = []
    cols = []
    vals = []
    lbs = []
    ubs = []
    
    # 1. Triple constraints
    for i, j, k in iso_triples:
        rows.append(len(rows)) # row index
        cols.extend([i, j, k])
        vals.extend([1, 1, 1])
        lbs.append(0)
        ubs.append(2)
        
    # 2. Maximality constraints
    for u in range(M):
        row_idx = len(rows)
        rows.append(row_idx)
        # -x_u
        cols.append(u)
        vals.append(-1)
        # -sum(z)
        for p_idx in E_u[u]:
            cols.append(M + p_idx)
            vals.append(-1)
        lbs.append(-1) # -1 <= -x_u - sum(z)  => x_u + sum(z) >= 1
        ubs.append(0)  # Wait, <= 0?
        # Constraint: -x_u - sum(z) <= -1
        # So lb = -inf, ub = -1?
        # scipy milp LinearConstraint takes (A, lb, ub)
        # Here A is the row.
        # We want A_row * x <= ub and A_row * x >= lb
        # So lb = -inf (or very small), ub = -1.
        lbs[-1] = -np.inf
        ubs[-1] = -1
        
    # 3. Linearization constraints
    for p_idx, (i, j) in enumerate(pairs_list):
        z_var = M + p_idx
        
        # z_p <= x_i  => -x_i + z_p <= 0
        row_idx = len(rows)
        rows.append(row_idx)
        cols.extend([i, z_var])
        vals.extend([-1, 1])
        lbs.append(-np.inf)
        ubs.append(0)
        
        # z_p <= x_j  => -x_j + z_p <= 0
        row_idx = len(rows)
        rows.append(row_idx)
        cols.extend([j, z_var])
        vals.extend([-1, 1])
        lbs.append(-np.inf)
        ubs.append(0)
        
        # x_i + x_j - z_p <= 1
        row_idx = len(rows)
        rows.append(row_idx)
        cols.extend([i, j, z_var])
        vals.extend([1, 1, -1])
        lbs.append(-np.inf)
        ubs.append(1)
        
    # Build sparse matrix
    A = csr_matrix((vals, (rows, cols)), shape=(len(rows), V))
    
    c = np.zeros(V)
    c[:M] = 1.0 # Minimize sum x
    
    integrality = np.zeros(V)
    integrality[:] = 1.0 # All binary
    
    bounds = Bounds(lb=0, ub=1)
    
    constraints = LinearConstraint(A, lb=lbs, ub=ubs)
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": 250})
        if res.status == 0:
            x_sol = res.x[:M]
            chosen = [i for i in range(M) if x_sol[i] > 0.5]
            return [all_points[i] for i in chosen]
    except Exception:
        pass
        
    # Fallback
    chosen, _ = _construct_greedy(all_indices, dist_matrix, random.Random(42))
    return [all_points[i] for i in chosen]

# EVOLVE-BLOCK-END
