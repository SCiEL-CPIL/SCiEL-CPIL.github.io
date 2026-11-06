"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _precompute_triples(n_max=180):
    """
    Pre-compute all triples (a, b, c) with a < b < c <= n_max 
    such that 1/a = 1/b + 1/c.
    Returns:
    - triples: list of tuples (a, b, c)
    - elem_to_triples: dict mapping element -> list of indices in triples
    """
    triples = []
    elem_to_triples = {i: [] for i in range(1, n_max + 1)}
    
    for a in range(1, n_max + 1):
        a_sq = a * a
        for x in range(1, a):
            if a_sq % x == 0:
                y = a_sq // x
                if x >= y:
                    continue
                b = a + x
                c = a + y
                if b <= n_max and c <= n_max:
                    idx = len(triples)
                    triples.append((a, b, c))
                    elem_to_triples[a].append(idx)
                    elem_to_triples[b].append(idx)
                    elem_to_triples[c].append(idx)
                    
    return triples, elem_to_triples

def _is_valid_set(s, triples, n_max=180):
    """Check if set s has no three distinct elements a,b,c with 1/a = 1/b + 1/c."""
    s_set = set(s)
    for (a, b, c) in triples:
        if a in s_set and b in s_set and c in s_set:
            return False
    return True

def _build_conflict_data(S, triples, elem_to_triples, n_max=180):
    """Build conflict tracking data for set S."""
    num_triples = len(triples)
    triple_state = np.zeros(num_triples, dtype=np.int32)
    for idx, (a, b, c) in enumerate(triples):
        cnt = (1 if a in S else 0) + (1 if b in S else 0) + (1 if c in S else 0)
        triple_state[idx] = cnt
    
    conflict_count = np.zeros(n_max + 1, dtype=np.int32)
    total_conflicts = 0
    for idx in range(num_triples):
        if triple_state[idx] == 3:
            total_conflicts += 1
            a, b, c = triples[idx]
            conflict_count[a] += 1
            conflict_count[b] += 1
            conflict_count[c] += 1
    
    return triple_state, conflict_count, total_conflicts

def _greedy_removal(S, triples, elem_to_triples, n_max=180, deadline=None):
    """Greedy removal to eliminate all conflicts."""
    triple_state, conflict_count, total_conflicts = _build_conflict_data(S, triples, elem_to_triples, n_max)
    
    while total_conflicts > 0:
        if deadline is not None and time.time() > deadline:
            break
        
        # Find element with max conflict count in S
        max_conflict = 0
        best_elem = None
        for x in S:
            if conflict_count[x] > max_conflict:
                max_conflict = conflict_count[x]
                best_elem = x
        
        if best_elem is None or max_conflict == 0:
            break
        
        S.remove(best_elem)
        
        for tidx in elem_to_triples[best_elem]:
            old_cnt = triple_state[tidx]
            if old_cnt == 3:
                total_conflicts -= 1
                a, b, c = triples[tidx]
                for x in (a, b, c):
                    if x != best_elem:
                        conflict_count[x] -= 1
            triple_state[tidx] = old_cnt - 1
    
    return S

def _try_add_back(S, triples, n_max=180, deadline=None):
    """Try to add back elements that were removed."""
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for elem in range(1, n_max + 1):
            if elem in S:
                continue
            trial = S | {elem}
            if _is_valid_set(trial, triples, n_max):
                S = trial
                improved = True
                break
    return S

def _local_search_swap(S, triples, n_max=180, deadline=None):
    """Local search: try swapping one element out for another to increase size."""
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for to_add in range(1, n_max + 1):
            if to_add in S:
                continue
            for to_remove in sorted(S, key=lambda x: random.random()):
                if deadline is not None and time.time() > deadline:
                    break
                trial = (S - {to_remove}) | {to_add}
                if _is_valid_set(trial, triples, n_max):
                    S = trial
                    improved = True
                    break
            if improved:
                break
    return S

def _solve_ilp(triples, n_max=180, deadline=None):
    """Solve the ILP for maximum independent set using scipy's MILP solver."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None
    
    num_vars = n_max  # variables x_1, ..., x_180
    num_constraints = len(triples)
    
    if num_constraints == 0:
        return list(range(1, n_max + 1))
    
    # Maximize sum(x_i) subject to x_a + x_b + x_c <= 2 for each triple
    # Convert to minimization: minimize -sum(x_i)
    c = -np.ones(num_vars)
    
    # Constraints: for each triple (a,b,c), x_a + x_b + x_c <= 2
    A = np.zeros((num_constraints, num_vars))
    for idx, (a, b, c) in enumerate(triples):
        A[idx, a-1] = 1
        A[idx, b-1] = 1
        A[idx, c-1] = 1
    
    # 0 <= x_i <= 1, integer
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(num_vars)
    
    # Linear constraint: A @ x <= 2
    constraint = LinearConstraint(A, ub=np.full(num_constraints, 2))
    
    try:
        result = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
        
        if result.status != 0 and result.status != 1:
            return None
        
        x = np.round(result.x).astype(int)
        S = set()
        for i in range(num_vars):
            if x[i] == 1:
                S.add(i + 1)
        
        if not S:
            return None
        
        return S
    except Exception:
        return None

def _multi_restart_greedy(triples, elem_to_triples, n_max=180, deadline=None):
    """Multi-restart greedy with randomization."""
    best_S = set()
    best_size = 0
    
    all_elements = set(range(1, n_max + 1))
    
    for restart in range(50):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        random.seed(42 + restart)
        
        S = set(all_elements)
        
        # Randomize removal order by shuffling conflict resolution
        S = _greedy_removal(S, triples, elem_to_triples, n_max, deadline=deadline)
        S = _try_add_back(S, triples, n_max, deadline=deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n_max = 180
    triples, elem_to_triples = _precompute_triples(n_max)
    
    # Strategy 1: Try ILP for exact/near-exact solution
    best_set = _solve_ilp(triples, n_max, deadline=deadline)
    
    if best_set is None or len(best_set) < 100:
        # Strategy 2: Multi-restart greedy
        greedy_set = _multi_restart_greedy(triples, elem_to_triples, n_max, deadline=deadline)
        if best_set is None or len(greedy_set) > len(best_set):
            best_set = greedy_set
    
    if best_set is None:
        best_set = set(range(91, n_max + 1))
    
    # Polish with local search
    if time.time() < deadline - 15:
        best_set = _try_add_back(best_set, triples, n_max, deadline=deadline)
    
    if time.time() < deadline - 5:
        best_set = _local_search_swap(best_set, triples, n_max, deadline=deadline)
    
    # Final add-back
    if time.time() < deadline - 2:
        best_set = _try_add_back(best_set, triples, n_max, deadline=deadline)
    
    # Verify validity
    if not _is_valid_set(best_set, triples, n_max):
        # Fallback: greedy removal from current set
        best_set = _greedy_removal(set(best_set), triples, elem_to_triples, n_max, deadline=deadline)
    
    result = sorted(best_set)
    if not result:
        result = list(range(91, n_max + 1))
    
    return result
# EVOLVE-BLOCK-END
