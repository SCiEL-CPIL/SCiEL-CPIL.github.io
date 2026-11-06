import time
import random
from itertools import combinations

# Pre-compute all 4-subsets of [1, 16] as bitmasks
ALL_SETS = []
for combo in combinations(range(16), 4):
    mask = 0
    for i in combo:
        mask |= (1 << i)
    ALL_SETS.append(mask)

NUM_SETS = len(ALL_SETS)  # 1820

# For each pair of sets, precompute their intersection
# We'll use this to speed up sunflower checking
# But for efficiency, let's use a different approach

# For a given set S, and family F, S cannot be added if there exist A, B in F
# such that A ∩ B = A ∩ S = B ∩ S
# This means A ∩ B is a subset of S, and A ∩ S = B ∩ S = A ∩ B
# Equivalently: (A & B) == (A & S) == (B & S)

# Pre-compute pairwise intersections for all pairs
# This is 1820 * 1819 / 2 ≈ 1.65M pairs, each a bitmask AND
# Store in a dict or 2D array for quick lookup

# Actually, let's think about a smarter approach.
# For each set S, define for each other set T: inter[S][T] = S & T
# Then S can't be added to F if there exist A, B in F with inter[A][B] == inter[A][S] == inter[B][S]

# Let's pre-compute the intersection table
# inter[i][j] = ALL_SETS[i] & ALL_SETS[j]
inter = [[0] * NUM_SETS for _ in range(NUM_SETS)]
for i in range(NUM_SETS):
    mi = ALL_SETS[i]
    for j in range(i + 1, NUM_SETS):
        v = mi & ALL_SETS[j]
        inter[i][j] = v
        inter[j][i] = v

def can_add_fast(S_idx, F_indices):
    """Check if set S_idx can be added to family F (list of indices)."""
    S = ALL_SETS[S_idx]
    n = len(F_indices)
    for i in range(n):
        a = F_indices[i]
        AS = inter[a][S_idx]
        for j in range(i + 1, n):
            b = F_indices[j]
            AB = inter[a][b]
            if AB == AS:
                if AB == inter[b][S_idx]:
                    return False
    return True

def greedy_min_degree(seed):
    """Greedy: repeatedly pick the set with minimum conflict degree."""
    random.seed(seed)
    candidates = list(range(NUM_SETS))
    random.shuffle(candidates)
    
    F = []
    # For tracking conflicts, we need to know which pairs in F conflict with which candidate
    
    # A better approach: for each candidate, count how many pairs (a,b) in F 
    # would form a sunflower with it. But that's expensive.
    
    # Alternative: just use random order greedy but with multiple restarts
    # and a smarter local search.
    
    for idx in candidates:
        if can_add_fast(idx, F):
            F.append(idx)
    return F

def local_search_improved(F, seed, time_limit):
    """Local search: try to swap out one set and add multiple others."""
    random.seed(seed)
    F = F[:]
    best_F = F[:]
    best_size = len(F)
    
    F_set = set(F)
    
    start = time.time()
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        # Try removing each set and see if we can add more
        for i in range(len(F)):
            if time.time() >= time_limit:
                break
            removed = F[i]
            F.pop(i)
            F_set.discard(removed)
            
            # Try to add as many as possible
            # Sort candidates by how many new conflicts they create
            # For now, just try in random order
            candidates = [x for x in range(NUM_SETS) if x not in F_set]
            random.shuffle(candidates)
            
            added = 0
            for c in candidates:
                if can_add_fast(c, F):
                    F.append(c)
                    F_set.add(c)
                    added += 1
            
            if len(F) > best_size:
                best_size = len(F)
                best_F = F[:]
                improved = True
            
            if len(F) < best_size:
                # Revert: we need to restore F to before we removed
                # This is tricky. Let's just keep the best.
                pass
        
        # Reset F to best
        F = best_F[:]
        F_set = set(F)
    
    return best_F

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, 10s buffer
    
    best_F = []
    best_size = 0
    
    # First, run a few quick greedy restarts
    seed = 42
    while time.time() < deadline - 10:
        F = greedy_min_degree(seed)
        if len(F) > best_size:
            best_size = len(F)
            best_F = F[:]
        seed += 1
        
        # If we've found something decent, do local search
        if best_size >= 100 and time.time() < deadline - 5:
            improved_F = local_search_improved(best_F, seed + 1000, deadline - 5)
            if len(improved_F) > best_size:
                best_size = len(improved_F)
                best_F = improved_F[:]
    
    # Convert bitmasks back to sorted lists of 1..16
    result = []
    for idx in best_F:
        mask = ALL_SETS[idx]
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(s)
    
    return result
