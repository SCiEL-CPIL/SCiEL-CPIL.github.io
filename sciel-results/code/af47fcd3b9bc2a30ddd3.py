"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START

# Precompute all K(2,2,2) configurations and which triples belong to each
_K222_CONFIGS = []  # list of frozensets of 8 triples
_TRIPLE_TO_CONFIGS = defaultdict(list)  # triple -> list of config indices

def _precompute_k222():
    """Precompute all K(2,2,2) configurations on 16 points."""
    global _K222_CONFIGS, _TRIPLE_TO_CONFIGS
    _K222_CONFIGS = []
    _TRIPLE_TO_CONFIGS = defaultdict(list)
    
    for six_points in itertools.combinations(range(16), 6):
        a, b, c, d, e, f = six_points
        # All 15 ways to partition 6 points into 3 pairs
        pairings = [
            [(a,b), (c,d), (e,f)],
            [(a,b), (c,e), (d,f)],
            [(a,b), (c,f), (d,e)],
            [(a,c), (b,d), (e,f)],
            [(a,c), (b,e), (d,f)],
            [(a,c), (b,f), (d,e)],
            [(a,d), (b,c), (e,f)],
            [(a,d), (b,e), (c,f)],
            [(a,d), (b,f), (c,e)],
            [(a,e), (b,c), (d,f)],
            [(a,e), (b,d), (c,f)],
            [(a,e), (b,f), (c,d)],
            [(a,f), (b,c), (d,e)],
            [(a,f), (b,d), (c,e)],
            [(a,f), (b,e), (c,d)],
        ]
        
        seen_configs = set()
        for pairing in pairings:
            norm_pairs = tuple(tuple(sorted(p)) for p in sorted(pairing))
            if norm_pairs in seen_configs:
                continue
            seen_configs.add(norm_pairs)
            
            (p1, p2, p3) = norm_pairs
            transversals = set()
            for x in p1:
                for y in p2:
                    for z in p3:
                        transversals.add(tuple(sorted((x, y, z))))
            
            config = frozenset(transversals)
            if len(config) != 8:
                continue
            
            config_idx = len(_K222_CONFIGS)
            _K222_CONFIGS.append(config)
            
            for t in config:
                _TRIPLE_TO_CONFIGS[t].append(config_idx)

# Call once at module load
_precompute_k222()

def _build_constraint_counter(current_set):
    """
    Build a counter: for each K(2,2,2) config, how many of its 8 triples
    are in current_set. Returns a dict {config_idx: count}.
    """
    counter = {}
    for t in current_set:
        for cfg_idx in _TRIPLE_TO_CONFIGS.get(t, []):
            counter[cfg_idx] = counter.get(cfg_idx, 0) + 1
    return counter

def _can_add(triple, current_set, constraint_counter):
    """
    Check if triple can be added without completing any K(2,2,2).
    A config is completed if it has all 8 triples. Adding triple completes
    a config if that config currently has exactly 7 triples in current_set
    and triple is the 8th.
    """
    for cfg_idx in _TRIPLE_TO_CONFIGS.get(triple, []):
        if constraint_counter.get(cfg_idx, 0) >= 7:
            return False
    return True

def _add_triple(triple, current_set, constraint_counter):
    """Add triple and update constraint counter."""
    current_set.add(triple)
    for cfg_idx in _TRIPLE_TO_CONFIGS.get(triple, []):
        constraint_counter[cfg_idx] = constraint_counter.get(cfg_idx, 0) + 1

def _remove_triple(triple, current_set, constraint_counter):
    """Remove triple and update constraint counter."""
    current_set.remove(triple)
    for cfg_idx in _TRIPLE_TO_CONFIGS.get(triple, []):
        constraint_counter[cfg_idx] -= 1

def _greedy_construct(deadline, seed=None):
    """Greedy construction with constraint counting."""
    if seed is not None:
        random.seed(seed)
    
    all_triples = list(itertools.combinations(range(16), 3))
    random.shuffle(all_triples)
    
    current_set = set()
    constraint_counter = {}
    
    for t in all_triples:
        if time.time() > deadline - 3:
            break
        if _can_add(t, current_set, constraint_counter):
            _add_triple(t, current_set, constraint_counter)
    
    return current_set

def _local_search(initial_set, deadline, seed=None):
    """Local search with add/swap moves using constraint counting."""
    if seed is not None:
        random.seed(seed)
    
    current_set = set(initial_set)
    constraint_counter = _build_constraint_counter(current_set)
    all_triples = list(itertools.combinations(range(16), 3))
    best_size = len(current_set)
    
    while time.time() < deadline - 2:
        improved = False
        
        # Try adding missing triples
        missing = [t for t in all_triples if t not in current_set]
        random.shuffle(missing)
        
        for t in missing:
            if _can_add(t, current_set, constraint_counter):
                _add_triple(t, current_set, constraint_counter)
                improved = True
                best_size = len(current_set)
                break
        
        if improved:
            continue
        
        # Try 1-remove, multiple-add swap moves
        if time.time() > deadline - 5:
            break
        
        remove_candidates = list(current_set)
        random.shuffle(remove_candidates)
        
        for t_rem in remove_candidates[:15]:
            # Temporarily remove
            _remove_triple(t_rem, current_set, constraint_counter)
            
            added_triples = []
            missing = [t for t in all_triples if t not in current_set and t != t_rem]
            random.shuffle(missing)
            
            for t_add in missing:
                if _can_add(t_add, current_set, constraint_counter):
                    _add_triple(t_add, current_set, constraint_counter)
                    added_triples.append(t_add)
            
            if len(added_triples) > 1:
                improved = True
                best_size = len(current_set)
                break
            else:
                # Revert
                for t_add in added_triples:
                    _remove_triple(t_add, current_set, constraint_counter)
                _add_triple(t_rem, current_set, constraint_counter)
        
        if not improved:
            # Try 2-remove moves for larger perturbations
            if time.time() > deadline - 8:
                break
            
            pairs_to_remove = list(itertools.combinations(current_set, 2))
            random.shuffle(pairs_to_remove)
            
            for t_rem1, t_rem2 in pairs_to_remove[:10]:
                _remove_triple(t_rem1, current_set, constraint_counter)
                _remove_triple(t_rem2, current_set, constraint_counter)
                
                added_triples = []
                missing = [t for t in all_triples if t not in current_set and t != t_rem1 and t != t_rem2]
                random.shuffle(missing)
                
                for t_add in missing:
                    if _can_add(t_add, current_set, constraint_counter):
                        _add_triple(t_add, current_set, constraint_counter)
                        added_triples.append(t_add)
                
                if len(added_triples) > 2:
                    improved = True
                    best_size = len(current_set)
                    break
                else:
                    for t_add in added_triples:
                        _remove_triple(t_add, current_set, constraint_counter)
                    _add_triple(t_rem1, current_set, constraint_counter)
                    _add_triple(t_rem2, current_set, constraint_counter)
            
            if not improved:
                break
    
    return current_set

def _solve_ilp(time_limit=80.0):
    """
    Solve the MILP to maximize the number of triples.
    Returns a set of triples or None if it fails.
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
    except ImportError:
        return None, 0.0
    
    start_time = time.time()
    
    triple_list = list(itertools.combinations(range(16), 3))
    triple_to_idx = {t: i for i, t in enumerate(triple_list)}
    n_vars = len(triple_list)
    
    # Use precomputed K(2,2,2) configs
    constraints_data = []
    for config in _K222_CONFIGS:
        row_indices = [triple_to_idx[t] for t in config]
        constraints_data.append(row_indices)
    
    n_constraints = len(constraints_data)
    
    row_idx = []
    col_idx = []
    data = []
    
    for i, cols in enumerate(constraints_data):
        for c in cols:
            row_idx.append(i)
            col_idx.append(c)
            data.append(1)
    
    A = csr_matrix((data, (row_idx, col_idx)), shape=(n_constraints, n_vars))
    
    c_obj = -np.ones(n_vars)
    
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    lb = np.zeros(n_constraints)
    ub = np.full(n_constraints, 7)
    lin_cons = LinearConstraint(A, lb=lb, ub=ub)
    
    integrality = np.ones(n_vars)
    
    result = milp(
        c=c_obj,
        constraints=lin_cons,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit, "mip_rel_gap": 0.05}
    )
    
    elapsed = time.time() - start_time
    
    if result.x is not None:
        selected = set()
        for i, val in enumerate(result.x):
            if val > 0.5:
                selected.add(triple_list[i])
        if len(selected) > 0:
            return selected, elapsed
    
    return None, elapsed

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Phase 1: Multiple greedy restarts with different seeds
    best_set = set()
    best_size = 0
    
    greedy_deadline = deadline - 40.0
    n_greedy = 5
    for i in range(n_greedy):
        if time.time() > greedy_deadline:
            break
        g_set = _greedy_construct(deadline, seed=i * 12345 + 678)
        if len(g_set) > best_size:
            best_set = g_set
            best_size = len(g_set)
    
    # Phase 2: Try ILP if enough time remains
    remaining = deadline - time.time() - 15.0
    if remaining > 20.0:
        ilp_set, elapsed = _solve_ilp(time_limit=remaining)
        if ilp_set is not None and len(ilp_set) > best_size:
            best_set = ilp_set
            best_size = len(ilp_set)
    
    # Phase 3: Local search from best solution
    remaining = deadline - time.time() - 3.0
    if remaining > 3.0:
        ls_set = _local_search(best_set, deadline, seed=42)
        if len(ls_set) > best_size:
            best_set = ls_set
            best_size = len(ls_set)
        
        # Additional local search with different seed if time allows
        remaining = deadline - time.time() - 2.0
        if remaining > 5.0 and best_size > 0:
            ls_set2 = _local_search(best_set, deadline, seed=999)
            if len(ls_set2) > best_size:
                best_set = ls_set2
                best_size = len(ls_set2)
    
    return [list(t) for t in best_set]
# EVOLVE-BLOCK-END
