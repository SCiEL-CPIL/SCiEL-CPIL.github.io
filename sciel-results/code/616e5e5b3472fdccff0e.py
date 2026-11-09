import time
import random
import math
from itertools import combinations

# Precompute all possible triples on 16 points
ALL_TRIPLES = []
ALL_TRIPLE_MASKS = []
TRIPLE_PAIRS = [] # List of (pair1, pair2, pair3) for each triple

for i in range(16):
    for j in range(i + 1, 16):
        for k in range(j + 1, 16):
            ALL_TRIPLES.append([i, j, k])
            mask = (1 << i) | (1 << j) | (1 << k)
            ALL_TRIPLE_MASKS.append(mask)
            TRIPLE_PAIRS.append(((i, j), (i, k), (j, k)))

NUM_TRIPLES = len(ALL_TRIPLES)

# Precompute pair-to-index mapping and triple pair masks
pair_to_idx = {}
_idx = 0
for _i in range(16):
    for _j in range(_i + 1, 16):
        pair_to_idx[(_i, _j)] = _idx
        _idx += 1

TRIPLE_PAIR_MASKS = []
for _pairs in TRIPLE_PAIRS:
    _mask = 0
    for _p in _pairs:
        _mask |= (1 << pair_to_idx[_p])
    TRIPLE_PAIR_MASKS.append(_mask)

# Precompute conflict sets: for each triple, which other triples share a pair
# Using bitmasks over triple indices (560 bits, fits in Python int)
TRIPLE_CONFLICT_MASKS = [0] * NUM_TRIPLES
for _a in range(NUM_TRIPLES):
    for _b in range(_a + 1, NUM_TRIPLES):
        if TRIPLE_PAIR_MASKS[_a] & TRIPLE_PAIR_MASKS[_b]:
            TRIPLE_CONFLICT_MASKS[_a] |= (1 << _b)
            TRIPLE_CONFLICT_MASKS[_b] |= (1 << _a)

def compute_alpha_fast(H_indices):
    """
    Compute the independence number of the hypergraph defined by H_indices.
    Uses branch-and-bound on the complement graph (clique in original conflict graph).
    H_indices is a set of indices into ALL_TRIPLES.
    """
    if not H_indices:
        return 16
    
    H_list = sorted(H_indices)
    n = len(H_list)
    if n == 0:
        return 16
    if n == 1:
        # One triple covers 3 points, so 13 points are free
        return 13
    
    # Build the "independent" relation: two points are "independent" if no triple contains both.
    # Actually, for the hypergraph independence: a set S is independent if no triple is fully contained in S.
    # We want max |S| such that no triple T in H has T ⊆ S.
    # Equivalently: for each triple T, S must miss at least one point of T.
    
    # This is the complement of a hitting set problem. Let's use a different approach:
    # The independence number = 16 - minimum vertex cover of the hypergraph.
    # But that's also hard. Let's just use a direct branch-and-bound.
    
    # Alternative: enumerate subsets from largest to smallest using a smarter method.
    # Since n <= ~100 triples max in practice, and we have 16 points,
    # let's use a DP/branch approach on points.
    
    # For each subset of points, check if it's independent.
    # But 2^16 = 65536 subsets. For each, check against H_list.
    # That's 65536 * n checks. With n up to ~100, that's ~6.5M ops. Fast enough.
    
    # Precompute for each point, which triples contain it
    point_triples = [[] for _ in range(16)]
    for idx in H_list:
        for pt in ALL_TRIPLES[idx]:
            point_triples[pt].append(idx)
    
    # For a given mask (set of points), it's independent iff for every triple in H,
    # the triple is NOT a subset of the mask.
    # i.e., for every triple T, (T_mask & mask) != T_mask
    
    # We can precompute a "covered" bitmask: for each point subset, which triples are fully covered.
    # But simpler: iterate masks from large to small.
    
    # Optimization: use the fact that if a mask is independent, all its subsets are also independent.
    # So we can use a recursive approach.
    
    # Let's just do the brute force with early termination, checking from size 16 down.
    # But we can be smarter: start from size 16 and use pruning.
    
    # Actually, let's use a different strategy: 
    # The minimum number of points to remove so that no triple is fully contained = minimum vertex cover.
    # alpha = 16 - min_vertex_cover.
    
    # For branch and bound on vertex cover:
    # We need to find the minimum set of points C such that every triple in H has at least one point in C.
    
    best_cover_size = 16  # worst case: all points
    best_cover = None
    
    def branch_and_bound(remaining_triples, points_available, current_cover_size, depth):
        nonlocal best_cover_size
        
        if not remaining_triples:
            if current_cover_size < best_cover_size:
                best_cover_size = current_cover_size
            return
        
        if current_cover_size >= best_cover_size:
            return
        
        # Pick a triple with the fewest available points (MRV heuristic)
        min_count = 100
        min_triple = None
        for t in remaining_triples:
            count = 0
            for pt in ALL_TRIPLES[t]:
                if points_available & (1 << pt):
                    count += 1
            if count < min_count:
                min_count = count
                min_triple = t
                if count <= 1:
                    break
        
        if min_count == 0:
            # This triple can't be covered - shouldn't happen if points_available is correct
            # But if all points of a triple are already removed, it's covered
            # Actually, if min_count == 0, the triple has no available points, so it's already "covered"
            # by the fact that its points aren't available. Wait, no - we're choosing points to REMOVE.
            # If a triple has no available points, it means all its points are already in the cover,
            # so it's covered.
            # Let me re-think: points_available are points NOT yet in the cover.
            # A triple is covered if at least one of its points is in the cover,
            # i.e., at least one of its points is NOT in points_available.
            # So a triple is uncovered iff all its points are in points_available.
            
            # If min_count == 0, no triple is fully available, so all are covered.
            if current_cover_size < best_cover_size:
                best_cover_size = current_cover_size
            return
        
        # Branch on each available point of min_triple
        triple_points = [pt for pt in ALL_TRIPLES[min_triple] if points_available & (1 << pt)]
        
        for pt in triple_points:
            new_points_available = points_available & ~(1 << pt)
            # Update remaining triples: remove any triple that is now covered
            new_remaining = []
            for t in remaining_triples:
                # Check if t is still uncovered (all points in new_points_available)
                uncovered = True
                for p in ALL_TRIPLES[t]:
                    if not (new_points_available & (1 << p)):
                        uncovered = False
                        break
                if uncovered:
                    new_remaining.append(t)
            branch_and_bound(new_remaining, new_points_available, current_cover_size + 1, depth + 1)
    
    # Initial: all 16 points available, all triples in H need to be covered
    branch_and_bound(H_list, (1 << 16) - 1, 0, 0)
    
    return 16 - best_cover_size

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_alpha = 16
    best_system = []
    
    # Strategy:
    # 1. Generate a random greedy packing.
    # 2. Local search with simulated annealing + targeted bottleneck-breaking moves to minimize alpha.
    
    # Parameters for Simulated Annealing
    initial_temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    for restart in range(500): # More restarts with faster alpha computation
        if time.time() > deadline:
            break
            
        # 1. Random Greedy Packing
        H_indices = set()
        used_pairs_mask = 0
        
        available = list(range(NUM_TRIPLES))
        random.shuffle(available)
        
        for idx in available:
            pmask = TRIPLE_PAIR_MASKS[idx]
            if (used_pairs_mask & pmask) == 0:
                H_indices.add(idx)
                used_pairs_mask |= pmask
        
        current_alpha = compute_alpha_fast(H_indices)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_system = [ALL_TRIPLES[i] for i in H_indices]
        
        # 2. Simulated Annealing with targeted moves
        temp = initial_temp
        last_improvement = time.time()
        
        while time.time() < deadline:
            if time.time() - last_improvement > 3.0:
                break
                
            move_type = random.random()
            
            if move_type < 0.3 and H_indices:
                # Targeted move: remove a triple that intersects the current max independent set
                # Find a large independent set to target
                # For efficiency, just pick a random triple to remove and try to add
                # a triple that covers more of the "gaps"
                
                rem_idx = random.choice(list(H_indices))
                rem_pmask = TRIPLE_PAIR_MASKS[rem_idx]
                used_pairs_mask_temp = used_pairs_mask & ~rem_pmask
                
                found_add = False
                for _ in range(20):
                    add_idx = random.randint(0, NUM_TRIPLES - 1)
                    if add_idx in H_indices:
                        continue
                    add_pmask = TRIPLE_PAIR_MASKS[add_idx]
                    if (used_pairs_mask_temp & add_pmask) == 0:
                        H_indices.remove(rem_idx)
                        H_indices.add(add_idx)
                        used_pairs_mask = used_pairs_mask_temp | add_pmask
                        
                        new_alpha = compute_alpha_fast(H_indices)
                        delta = new_alpha - current_alpha
                        
                        if delta <= 0 or random.random() < math.exp(-delta / temp):
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_system = [ALL_TRIPLES[i] for i in H_indices]
                                last_improvement = time.time()
                        else:
                            H_indices.remove(add_idx)
                            H_indices.add(rem_idx)
                            used_pairs_mask = (used_pairs_mask & ~add_pmask) | rem_pmask
                        found_add = True
                        break
                        
            elif move_type < 0.6:
                # Swap move
                if H_indices:
                    rem_idx = random.choice(list(H_indices))
                    rem_pmask = TRIPLE_PAIR_MASKS[rem_idx]
                    used_pairs_mask_temp = used_pairs_mask & ~rem_pmask
                    
                    found_add = False
                    for _ in range(15):
                        add_idx = random.randint(0, NUM_TRIPLES - 1)
                        if add_idx in H_indices:
                            continue
                        add_pmask = TRIPLE_PAIR_MASKS[add_idx]
                        if (used_pairs_mask_temp & add_pmask) == 0:
                            H_indices.remove(rem_idx)
                            H_indices.add(add_idx)
                            used_pairs_mask = used_pairs_mask_temp | add_pmask
                            
                            new_alpha = compute_alpha_fast(H_indices)
                            delta = new_alpha - current_alpha
                            
                            if delta <= 0 or random.random() < math.exp(-delta / temp):
                                current_alpha = new_alpha
                                if current_alpha < best_alpha:
                                    best_alpha = current_alpha
                                    best_system = [ALL_TRIPLES[i] for i in H_indices]
                                    last_improvement = time.time()
                            else:
                                H_indices.remove(add_idx)
                                H_indices.add(rem_idx)
                                used_pairs_mask = (used_pairs_mask & ~add_pmask) | rem_pmask
                            found_add = True
                            break
                        
            else:
                # Add move
                for _ in range(8):
                    add_idx = random.randint(0, NUM_TRIPLES - 1)
                    if add_idx in H_indices:
                        continue
                    add_pmask = TRIPLE_PAIR_MASKS[add_idx]
                    if (used_pairs_mask & add_pmask) == 0:
                        H_indices.add(add_idx)
                        used_pairs_mask |= add_pmask
                        
                        new_alpha = compute_alpha_fast(H_indices)
                        delta = new_alpha - current_alpha
                        
                        if delta <= 0 or random.random() < math.exp(-delta / temp):
                            current_alpha = new_alpha
                            if current_alpha < best_alpha:
                                best_alpha = current_alpha
                                best_system = [ALL_TRIPLES[i] for i in H_indices]
                                last_improvement = time.time()
                        else:
                            H_indices.remove(add_idx)
                            used_pairs_mask &= ~add_pmask
                        break
            
            temp *= cooling_rate
            if temp < min_temp:
                break

    return best_system

def _fallback_system():
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

_orig_construct = construct

def construct_safe():
    try:
        res = _orig_construct()
        if not res:
            return _fallback_system()
        # Validate linearity just in case
        used = set()
        for t in res:
            i,j,k = t
            pairs = [tuple(sorted((i,j))), tuple(sorted((i,k))), tuple(sorted((j,k)))]
            for p in pairs:
                if p in used:
                    return _fallback_system()
                used.add(p)
        return res
    except:
        return _fallback_system()

# Replace construct
construct = construct_safe
