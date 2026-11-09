import time
import random
from itertools import combinations
from math import exp

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

# Precompute subsets by size for independence check
SUBSETS_BY_SIZE = {k: [] for k in range(1, 17)}
for mask in range(1, 1 << 16):
    bc = bin(mask).count('1')
    if bc <= 16:
        SUBSETS_BY_SIZE[bc].append(mask)

def is_independent(mask, H_list):
    """
    Check if the set of vertices represented by mask is independent
    given the list of triple indices H_list.
    """
    for idx in H_list:
        if (ALL_TRIPLE_MASKS[idx] & mask) == ALL_TRIPLE_MASKS[idx]:
            return False
    return True

def compute_alpha(H_indices):
    """
    Compute the independence number of the hypergraph defined by H_indices.
    H_indices is a set of indices into ALL_TRIPLES.
    """
    if not H_indices:
        return 16
    
    H_list = list(H_indices)
    
    # Check from largest possible size down
    for k in range(16, 0, -1):
        for mask in SUBSETS_BY_SIZE[k]:
            if is_independent(mask, H_list):
                return k
    return 0

def generate_algebraic_triples():
    """
    Generate triples using algebraic structure of GF(4)^2 or Z_4 x Z_4.
    We'll use the 4-bit representation: each point is a pair (a, b) where a,b in {0,1,2,3}.
    Point index = a * 4 + b.
    
    We'll define triples based on affine lines and quadratic curves to ensure linearity.
    """
    triples = []
    used_pairs = set()
    
    # Method 1: Affine lines in Z_4 x Z_4
    # For each direction (dx, dy) and intercept, generate lines
    # A line: {(a, b), (a+dx, b+dy), (a+2dx, b+2dy)} mod 4
    # But we need to ensure these are valid triples (3 distinct points)
    
    directions = [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1), (1, 3)]
    
    for dx, dy in directions:
        for a0 in range(4):
            for b0 in range(4):
                # Generate the line: (a0 + t*dx, b0 + t*dy) for t=0,1,2
                pts = []
                for t in range(3):
                    a = (a0 + t * dx) % 4
                    b = (b0 + t * dy) % 4
                    idx = a * 4 + b
                    pts.append(idx)
                
                # Check if all distinct
                if len(set(pts)) == 3:
                    pts.sort()
                    key = tuple(pts)
                    # Check linearity
                    pairs = [tuple(sorted((pts[0], pts[1]))), 
                             tuple(sorted((pts[0], pts[2]))), 
                             tuple(sorted((pts[1], pts[2])))]
                    if not any(p in used_pairs for p in pairs):
                        triples.append(pts)
                        for p in pairs:
                            used_pairs.add(p)
    
    # Method 2: Add more triples from quadratic curves or other structures
    # Try to fill in with random triples that maintain linearity
    random.seed(42)
    for _ in range(500):
        # Try to add a random triple
        a = random.randint(0, 15)
        b = random.randint(0, 15)
        c = random.randint(0, 15)
        if a == b or b == c or a == c:
            continue
        
        pts = sorted([a, b, c])
        pairs = [tuple(sorted((pts[0], pts[1]))), 
                 tuple(sorted((pts[0], pts[2]))), 
                 tuple(sorted((pts[1], pts[2])))]
        
        if not any(p in used_pairs for p in pairs):
            triples.append(pts)
            for p in pairs:
                used_pairs.add(p)
    
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_alpha = 16
    best_system = []
    
    # Strategy:
    # 1. Start with algebraic construction
    # 2. Use local search to improve
    
    # Generate initial algebraic system
    algebraic_triples = generate_algebraic_triples()
    
    # Convert to indices
    triple_to_idx = {}
    for idx, t in enumerate(ALL_TRIPLES):
        triple_to_idx[tuple(t)] = idx
    
    H_indices = set()
    for t in algebraic_triples:
        if tuple(t) in triple_to_idx:
            H_indices.add(triple_to_idx[tuple(t)])
    
    current_alpha = compute_alpha(H_indices)
    best_alpha = current_alpha
    best_system = [ALL_TRIPLES[i] for i in H_indices]
    
    # Now do local search to improve
    # Use simulated annealing
    
    pair_to_idx = {}
    idx = 0
    for i in range(16):
        for j in range(i + 1, 16):
            pair_to_idx[(i, j)] = idx
            idx += 1
    
    TRIPLE_PAIR_MASKS = []
    for pairs in TRIPLE_PAIRS:
        mask = 0
        for p in pairs:
            mask |= (1 << pair_to_idx[p])
        TRIPLE_PAIR_MASKS.append(mask)
    
    used_pairs_mask = 0
    for idx in H_indices:
        used_pairs_mask |= TRIPLE_PAIR_MASKS[idx]
    
    initial_temp = 1.0
    cooling_rate = 0.999
    min_temp = 0.01
    
    temp = initial_temp
    last_improvement = time.time()
    
    while time.time() < deadline:
        if time.time() - last_improvement > 3.0:
            break
        
        # Choose a move
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
                    # Apply move
                    H_indices.remove(rem_idx)
                    H_indices.add(add_idx)
                    used_pairs_mask = used_pairs_mask_temp | add_pmask
                    
                    new_alpha = compute_alpha(H_indices)
                    delta = new_alpha - current_alpha
                    
                    if delta <= 0 or random.random() < exp(-delta / temp):
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = [ALL_TRIPLES[i] for i in H_indices]
                            last_improvement = time.time()
                    else:
                        # Revert
                        H_indices.remove(add_idx)
                        H_indices.add(rem_idx)
                        used_pairs_mask = (used_pairs_mask & ~add_pmask) | rem_pmask
                    found_add = True
                    break
            
            if not found_add:
                # Try to add a new triple
                for _ in range(8):
                    add_idx = random.randint(0, NUM_TRIPLES - 1)
                    if add_idx in H_indices:
                        continue
                    add_pmask = TRIPLE_PAIR_MASKS[add_idx]
                    if (used_pairs_mask & add_pmask) == 0:
                        H_indices.add(add_idx)
                        used_pairs_mask |= add_pmask
                        
                        new_alpha = compute_alpha(H_indices)
                        delta = new_alpha - current_alpha
                        
                        if delta <= 0 or random.random() < exp(-delta / temp):
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
            temp = min_temp
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
