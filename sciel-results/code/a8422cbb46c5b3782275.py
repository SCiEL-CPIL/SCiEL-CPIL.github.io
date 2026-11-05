import time
import random
import numpy as np
from functools import lru_cache
from itertools import combinations

N = 16
ALL_POINTS = list((x, y) for y in range(N) for x in range(N))
N_PTS = len(ALL_POINTS)

# Precompute squared distances
coords = np.array(ALL_POINTS)
diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
dist2 = np.sum(diff**2, axis=2)

# Precompute bad pairs for each point
# BAD_PAIRS[p] is a set of tuples (u, v) with u < v such that {p, u, v} is isosceles.
BAD_PAIRS = [set() for _ in range(N_PTS)]

def _precompute_bad_pairs():
    i_idx, j_idx = np.triu_indices(N_PTS, k=1)
    for p in range(N_PTS):
        dp = dist2[p, :]
        dp_i = dp[i_idx]
        dp_j = dp[j_idx]
        dij = dist2[i_idx, j_idx]
        cond = (dp_i == dp_j) | (dp_i == dij) | (dp_j == dij)
        bad_indices = np.where(cond)[0]
        for k in bad_indices:
            u, v = i_idx[k], j_idx[k]
            if u > v:
                u, v = v, u
            BAD_PAIRS[p].add((u, v))

_start_time = time.time()
_precompute_bad_pairs()

def _can_add(p, S_set):
    """Check if p can be added to S_set (represented as a set of indices)."""
    for u, v in BAD_PAIRS[p]:
        if u in S_set and v in S_set:
            return False
    return True

def _find_addable(S_set, all_points_set):
    """Find all points that can be added to S_set."""
    addable = set()
    for p in all_points_set:
        if p not in S_set:
            if _can_add(p, S_set):
                addable.add(p)
    return addable

def _make_maximal(S_set, all_points_set):
    """Ensure S_set is maximal by adding points until no more can be added."""
    addable = _find_addable(S_set, all_points_set)
    while addable:
        p = random.choice(list(addable))
        S_set.add(p)
        addable = _find_addable(S_set, all_points_set)
    return S_set

def _greedy_initialization(deadline):
    """Greedy construction of a maximal no-isosceles set."""
    S_set = set()
    C = set(range(N_PTS))
    
    while C:
        if time.time() > deadline:
            break
        
        best_p = -1
        best_new_C_size = float('inf')
        best_new_C = None
        
        candidates = list(C)
        random.shuffle(candidates)
        
        for p in candidates:
            new_C = set()
            for q in C:
                if q == p:
                    continue
                # Check if q is valid with S U {p}
                valid = True
                for u in S_set:
                    a, b = (p, u) if p < u else (u, p)
                    if (a, b) in BAD_PAIRS[q]:
                        valid = False
                        break
                if valid:
                    new_C.add(q)
            
            new_C_size = len(new_C)
            if new_C_size < best_new_C_size:
                best_new_C_size = new_C_size
                best_p = p
                best_new_C = new_C
        
        if best_p == -1:
            break
            
        S_set.add(best_p)
        C = best_new_C
        
    return S_set

def _local_search(S_initial, deadline, penalty_weight=10.0):
    """
    Local search to minimize |S| + penalty * N_addable.
    """
    S_set = set(S_initial)
    all_points_set = set(range(N_PTS))
    
    # Calculate initial energy
    addable = _find_addable(S_set, all_points_set)
    current_energy = len(S_set) + penalty_weight * len(addable)
    best_set = set(S_set)
    best_energy = current_energy
    
    temp = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if temp < 0.01:
            temp = 0.01
            
        # Generate a move
        move_type = random.choice(['add', 'remove', 'swap'])
        new_S_set = set(S_set)
        
        if move_type == 'add' and addable:
            p = random.choice(list(addable))
            new_S_set.add(p)
        elif move_type == 'remove' and S_set:
            p = random.choice(list(S_set))
            new_S_set.remove(p)
        elif move_type == 'swap':
            if not S_set or not addable:
                continue
            p_out = random.choice(list(S_set))
            p_in = random.choice(list(addable))
            new_S_set.remove(p_out)
            new_S_set.add(p_in)
        else:
            continue
            
        # Evaluate new state
        new_addable = _find_addable(new_S_set, all_points_set)
        new_energy = len(new_S_set) + penalty_weight * len(new_addable)
        
        # Acceptance criterion (Metropolis)
        delta_e = new_energy - current_energy
        if delta_e < 0 or random.random() < np.exp(-delta_e / temp):
            S_set = new_S_set
            addable = new_addable
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_set = set(S_set)
        
        temp *= cooling
    
    # Ensure the best set is maximal by adding points if necessary
    # This guarantees the output is valid and maximal, even if it increases size
    final_S = set(best_set)
    final_S = _make_maximal(final_S, all_points_set)
    
    return final_S

def construct():
    start_time = time.time()
    deadline = start_time + 300.0 # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts
    while time.time() < deadline - 5.0:
        # 1. Greedy Init
        S_init = _greedy_initialization(deadline - 5.0)
        
        # 2. Local Search
        S_opt = _local_search(S_init, deadline - 2.0)
        
        if len(S_opt) < best_size:
            best_size = len(S_opt)
            best_set = S_opt
            
    if best_set is None:
        # Fallback
        best_set = _greedy_initialization(deadline)
        best_set = _make_maximal(best_set, set(range(N_PTS)))
        
    return [ALL_POINTS[i] for i in best_set]
