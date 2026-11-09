"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

DEADLINE = 320  # seconds, safety margin before 330s kill

# Precomputation
ALL_POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = N * N
POINT_TO_IDX = {p: i for i, p in enumerate(ALL_POINTS)}
IDX_TO_POINT = {i: p for i, p in enumerate(ALL_POINTS)}

# Precompute distance squared between all pairs
def _d2_idx(i, j):
    pi = IDX_TO_POINT[i]
    pj = IDX_TO_POINT[j]
    dx = pi[0] - pj[0]
    dy = pi[1] - pj[1]
    return dx * dx + dy * dy

D2 = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    for j in range(NUM_POINTS):
        if i != j:
            D2[i][j] = _d2_idx(i, j)

# ISO_MASK[i][j] is a bitmask of all k such that (i, j, k) is an isosceles triangle with distinct vertices.
ISO_MASK = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for i in range(NUM_POINTS):
    for j in range(NUM_POINTS):
        if i == j:
            continue
        mask = 0
        d_ij = D2[i][j]
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            d_ik = D2[i][k]
            d_jk = D2[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        ISO_MASK[i][j] = mask

ALL_MASK = (1 << NUM_POINTS) - 1

def _can_add_bitset(p_idx, S_bitset, S_list):
    """
    Check if point p_idx can be added to set S (represented by S_bitset and S_list).
    """
    for a in S_list:
        if (ISO_MASK[p_idx][a] & S_bitset) != 0:
            return False
    return True

def _is_maximal_bitset(S_bitset, S_list):
    """
    Check if S is maximal.
    """
    for p_idx in range(NUM_POINTS):
        if (S_bitset >> p_idx) & 1:
            continue
        if _can_add_bitset(p_idx, S_bitset, S_list):
            return False
    return True

def _greedy_random(all_points_indices, rng):
    """
    Generate a random maximal set using greedy addition.
    """
    order = list(all_points_indices)
    rng.shuffle(order)
    S_list = []
    S_bitset = 0
    for p_idx in order:
        if _can_add_bitset(p_idx, S_bitset, S_list):
            S_list.append(p_idx)
            S_bitset |= (1 << p_idx)
    return S_list, S_bitset

def _local_search_minimize(S_list, S_bitset, rng, max_iters=100, time_limit=None):
    """
    Try to remove points from S while maintaining maximality.
    """
    S_list = list(S_list)
    S_bitset = S_bitset
    improved = True
    iters = 0
    start_time = time.time()
    while improved and iters < max_iters:
        if time_limit and time.time() - start_time > time_limit:
            break
        improved = False
        iters += 1
        indices = list(range(len(S_list)))
        rng.shuffle(indices)
        for idx in indices:
            p_idx = S_list[idx]
            new_S_list = S_list[:idx] + S_list[idx+1:]
            new_S_bitset = S_bitset ^ (1 << p_idx)
            
            if _is_maximal_bitset(new_S_bitset, new_S_list):
                S_list = new_S_list
                S_bitset = new_S_bitset
                improved = True
                break
    return S_list, S_bitset

def construct():
    start_time = time.time()
    all_indices = list(range(NUM_POINTS))
    rng = random.Random(42)
    
    best_S_list = None
    best_S_bitset = None
    best_size = NUM_POINTS + 1
    
    # Phase 1: Generate multiple random maximal sets and minimize them
    while time.time() - start_time < DEADLINE * 0.4:
        S_list, S_bitset = _greedy_random(all_indices, rng)
        # Minimize it
        S_list, S_bitset = _local_search_minimize(S_list, S_bitset, rng, max_iters=50)
        
        if len(S_list) < best_size:
            best_S_list = S_list
            best_S_bitset = S_bitset
            best_size = len(S_list)
            
    # Phase 2: Intensive local search on the best solution found
    if best_S_list is not None:
        while time.time() - start_time < DEADLINE - 5:
            S_list, S_bitset = _local_search_minimize(best_S_list, best_S_bitset, rng, max_iters=200)
            if len(S_list) < best_size:
                best_S_list = S_list
                best_S_bitset = S_bitset
                best_size = len(S_list)
            
            # Occasionally try a fresh random start
            if time.time() - start_time < DEADLINE - 10:
                S_list, S_bitset = _greedy_random(all_indices, rng)
                S_list, S_bitset = _local_search_minimize(S_list, S_bitset, rng, max_iters=100)
                if len(S_list) < best_size:
                    best_S_list = S_list
                    best_S_bitset = S_bitset
                    best_size = len(S_list)

    if best_S_list is None:
        S_list, S_bitset = _greedy_random(all_indices, rng)
        best_S_list = S_list
        best_S_bitset = S_bitset

    result = [IDX_TO_POINT[i] for i in best_S_list]
    return result
# EVOLVE-BLOCK-END
