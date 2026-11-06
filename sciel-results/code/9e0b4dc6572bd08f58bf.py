"""
Minimum maximal no-isosceles set in a 16x16 grid.

Strategy:
1. Precompute isosceles relationships using bitsets for O(1) lookups.
2. Multiple randomized greedy constructions with different orderings.
3. Aggressive minimality reduction using blocking-set analysis.
4. Local search: swap points to reduce size while maintaining maximality.
5. Time-bounded refinement with multiple passes.
"""
import time
import random
import numpy as np

N = 16
DEADLINE = 320.0

# Precompute all points
ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
N_POINTS = len(ALL_POINTS)

def _point_to_idx(p):
    return p[1] * N + p[0]

def _idx_to_point(idx):
    return (idx % N, idx // N)

# Precompute squared distances
def _precompute_dist2():
    d2 = np.zeros((N_POINTS, N_POINTS), dtype=np.int32)
    for i in range(N_POINTS):
        x1, y1 = _idx_to_point(i)
        for j in range(i + 1, N_POINTS):
            x2, y2 = _idx_to_point(j)
            d = (x1 - x2)**2 + (y1 - y2)**2
            d2[i, j] = d
            d2[j, i] = d
    return d2

DIST2 = _precompute_dist2()

# Precompute for each pair (i,j), the set of points k that form isosceles triangle with i,j
# iso_block[i][j] = bitset of all k such that (i,j,k) is isosceles
# Also: for each point k, iso_pairs[k] = list of (i,j) pairs that block k

def _precompute_iso():
    """Precompute isosceles relationships."""
    # For each point k, which pairs (i,j) form an isosceles triangle with k?
    # k is blocked by pair (i,j) if any two of the three distances are equal
    
    # iso_block[k] = bitset of pairs... actually let's think differently.
    # For a set S, point k (not in S) can be added iff for ALL pairs (i,j) in S, 
    # (i,j,k) is NOT isosceles.
    # So k is blocked by S iff there EXISTS a pair (i,j) in S such that (i,j,k) is isosceles.
    
    # Precompute: for each k, the set of all pairs (i,j) with i<j that make isosceles with k.
    # Store as list of (i,j) tuples.
    
    k_blocks = [[] for _ in range(N_POINTS)]  # k_blocks[k] = list of (i,j) pairs
    pair_iso = {}  # pair_iso[(i,j)] = bitset of k values that form isosceles with i,j
    
    for i in range(N_POINTS):
        for j in range(i+1, N_POINTS):
            d_ij = DIST2[i, j]
            iso_k = 0
            for k in range(N_POINTS):
                if k == i or k == j:
                    continue
                d_ik = DIST2[i, k]
                d_jk = DIST2[j, k]
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    iso_k |= (1 << k)
                    k_blocks[k].append((i, j))
            pair_iso[(i, j)] = iso_k
            pair_iso[(j, i)] = iso_k
    
    return k_blocks, pair_iso

K_BLOCKS, PAIR_ISO = _precompute_iso()

def _can_add_idx(p_idx, chosen_set):
    """Check if point p_idx can be added to chosen_set without creating isosceles triangle.
    chosen_set is a set of indices.
    """
    if p_idx in chosen_set:
        return False
    # p_idx is blocked if there exists a pair (i,j) in chosen_set such that (i,j,p_idx) is isosceles
    # k_blocks[p_idx] contains all pairs (i,j) that block p_idx
    for (i, j) in K_BLOCKS[p_idx]:
        if i in chosen_set and j in chosen_set:
            return False
    return True

def _is_maximal_idx(chosen_set):
    """Check if chosen_set is maximal (no point can be added)."""
    for p in range(N_POINTS):
        if p in chosen_set:
            continue
        if _can_add_idx(p, chosen_set):
            return False
    return True

def _make_maximal(chosen_list):
    """Add points to make the set maximal."""
    chosen_set = set(chosen_list)
    changed = True
    while changed:
        changed = False
        for p in range(N_POINTS):
            if p in chosen_set:
                continue
            if _can_add_idx(p, chosen_set):
                chosen_set.add(p)
                changed = True
    return list(chosen_set)

def _reduce_to_minimal(chosen_list, all_indices, time_limit=None, start_time=None):
    """
    Iteratively remove points from chosen_set if the remaining set is still maximal.
    Uses randomized order for better results.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 5.0
    
    chosen_list = list(chosen_list)
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            break
        changed = False
        # Try removing each point in random order
        indices = list(range(len(chosen_list)))
        random.shuffle(indices)
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            p = chosen_list[i]
            S_test = chosen_list[:i] + chosen_list[i+1:]
            
            if len(S_test) < 2:
                continue
            
            S_test_set = set(S_test)
            # Check maximality
            maximal = True
            for q in all_indices:
                if q in S_test_set:
                    continue
                if _can_add_idx(q, S_test_set):
                    maximal = False
                    break
            
            if maximal:
                chosen_list = S_test
                changed = True
                break
                
    return chosen_list

def _greedy_random(order, rng=None):
    """
    Greedy construction: iterate through points in given order, add if valid.
    """
    if rng is None:
        rng = random.Random()
    chosen = []
    chosen_set = set()
    for p in order:
        if p in chosen_set:
            continue
        if _can_add_idx(p, chosen_set):
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _local_search_swap(chosen, all_indices, time_limit=20.0):
    """
    Try to improve chosen by swapping one point for another.
    """
    start_time = time.time()
    chosen_list = list(chosen)
    n = len(chosen_list)
    rng = random.Random(123)
    
    for i in range(n):
        if time.time() - start_time > time_limit:
            break
        
        p_remove = chosen_list[i]
        S_without = chosen_list[:i] + chosen_list[i+1:]
        S_without_set = set(S_without)
        
        # Find all points that can replace p_remove
        candidates = []
        for p_add in all_indices:
            if p_add in S_without_set:
                continue
            if p_add == p_remove:
                continue
            
            if not _can_add_idx(p_add, S_without_set):
                continue
            
            S_new = S_without + [p_add]
            S_new_set = set(S_new)
            
            # Check if maximal
            maximal = True
            for q in all_indices:
                if q in S_new_set:
                    continue
                if _can_add_idx(q, S_new_set):
                    maximal = False
                    break
            
            if maximal:
                reduced = _reduce_to_minimal(S_new, all_indices, time_limit=3.0)
                if len(reduced) < n:
                    return reduced
                    
    return None

def _multi_remove_search(chosen, all_indices, time_limit=30.0):
    """
    Try removing 2 points at a time and see if maximality is preserved.
    """
    start_time = time.time()
    chosen_list = list(chosen)
    n = len(chosen_list)
    
    if n < 4:
        return None
    
    for i in range(n):
        if time.time() - start_time > time_limit:
            break
        for j in range(i+1, n):
            if time.time() - start_time > time_limit:
                break
            
            S_test = chosen_list[:i] + chosen_list[i+1:j] + chosen_list[j+1:]
            S_test_set = set(S_test)
            
            if len(S_test) < 2:
                continue
            
            maximal = True
            for q in all_indices:
                if q in S_test_set:
                    continue
                if _can_add_idx(q, S_test_set):
                    maximal = False
                    break
            
            if maximal:
                reduced = _reduce_to_minimal(S_test, all_indices, time_limit=5.0)
                if len(reduced) < n:
                    return reduced
                    
    return None

def _construct_with_seed(seed_points, all_indices):
    """
    Start from a seed set and grow to maximal, then reduce.
    """
    chosen = list(seed_points)
    chosen_set = set(chosen)
    
    # Make maximal
    changed = True
    while changed:
        changed = False
        for p in all_indices:
            if p in chosen_set:
                continue
            if _can_add_idx(p, chosen_set):
                chosen_set.add(p)
                changed = True
    
    chosen = list(chosen_set)
    reduced = _reduce_to_minimal(chosen, all_indices, time_limit=10.0)
    return reduced

def construct():
    start_time = time.time()
    
    all_indices = list(range(N_POINTS))
    
    best_set = None
    best_size = N_POINTS + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Random greedy + reduction
    for attempt in range(500):
        if time.time() - start_time > DEADLINE - 40:
            break
        
        order = all_indices[:]
        rng.shuffle(order)
        
        chosen = _greedy_random(order, rng)
        
        # Ensure maximality
        chosen_set = set(chosen)
        changed = True
        while changed:
            changed = False
            for p in all_indices:
                if p in chosen_set:
                    continue
                if _can_add_idx(p, chosen_set):
                    chosen_set.add(p)
                    changed = True
        
        chosen = list(chosen_set)
        reduced = _reduce_to_minimal(chosen, all_indices, time_limit=5.0)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 7:
            break
    
    # Strategy 2: Local search on best found
    if best_set is not None:
        for ls_round in range(5):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            improved = _local_search_swap(best_set, all_indices, time_limit=15.0)
            if improved is not None and len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
            
            improved2 = _multi_remove_search(best_set, all_indices, time_limit=10.0)
            if improved2 is not None and len(improved2) < best_size:
                best_size = len(improved2)
                best_set = improved2[:]
                
            if best_size <= 7:
                break
    
    # Strategy 3: More random restarts with different seeds
    for attempt in range(200):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Start with a small random seed
        seed_size = rng.randint(2, 4)
        seed = rng.sample(all_indices, seed_size)
        
        # Check if seed is valid (no isosceles)
        valid = True
        seed_set = set(seed)
        for i in range(len(seed)):
            for j in range(i+1, len(seed)):
                for k in range(len(seed)):
                    if k == i or k == j:
                        continue
                    a, b, c = seed[i], seed[j], seed[k]
                    d_ab = DIST2[a, b]
                    d_ac = DIST2[a, c]
                    d_bc = DIST2[b, c]
                    if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        reduced = _construct_with_seed(seed, all_indices)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 7:
            break
    
    # Strategy 4: Final local search passes
    if best_set is not None:
        for _ in range(3):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            improved = _local_search_swap(best_set, all_indices, time_limit=10.0)
            if improved is not None and len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
            
            improved2 = _multi_remove_search(best_set, all_indices, time_limit=8.0)
            if improved2 is not None and len(improved2) < best_size:
                best_size = len(improved2)
                best_set = improved2[:]
                
            if best_size <= 7:
                break
    
    # Strategy 5: One more round of random greedy with best so far as guide
    if best_set is not None and time.time() - start_time < DEADLINE - 15:
        # Try to find even smaller sets by being more selective
        for attempt in range(100):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            order = all_indices[:]
            rng.shuffle(order)
            
            chosen = _greedy_random(order, rng)
            chosen_set = set(chosen)
            
            # Make maximal
            changed = True
            while changed:
                changed = False
                for p in all_indices:
                    if p in chosen_set:
                        continue
                    if _can_add_idx(p, chosen_set):
                        chosen_set.add(p)
                        changed = True
            
            chosen = list(chosen_set)
            reduced = _reduce_to_minimal(chosen, all_indices, time_limit=3.0)
            
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced[:]
                
            if best_size <= 7:
                break
    
    if best_set is None:
        # Fallback: simple greedy
        chosen = _greedy_random(all_indices[:], rng)
        return [ALL_POINTS[i] for i in chosen]
    
    return [ALL_POINTS[i] for i in best_set]
