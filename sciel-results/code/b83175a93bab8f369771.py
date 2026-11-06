# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1,16]
_N = 16
_K = 4
_ALL = [tuple(sorted(c)) for c in combinations(range(1, _N + 1), _K)]
_IDX = {s: i for i, s in enumerate(_ALL)}
_M = len(_ALL)  # 1820

# Precompute intersection sizes between all pairs
# inter[i][j] = |A_i ∩ A_j|
# Since |A_i|=4, possible intersection sizes are 0,1,2,3,4
# We only care about pairs (i,j) with i != j
# For sunflower check: three distinct sets A,B,C form a sunflower iff
#   |A∩B| == |A∩C| == |B∩C|
# (This includes the empty-core case where all intersections are 0.)

# Actually, let me re-read: "No three distinct sets may have all three pairwise intersections equal"
# So the condition is: for any three distinct sets A, B, C, NOT (|A∩B| == |A∩C| == |B∩C|)

# Precompute pairwise intersection sizes as a 2D array
# Use a flat list for speed: inter[i*M + j]
_inter = [[0] * _M for _ in range(_M)]
for i in range(_M):
    si = set(_ALL[i])
    for j in range(i + 1, _M):
        sj = set(_ALL[j])
        inter_size = len(si & sj)
        _inter[i][j] = inter_size
        _inter[j][i] = inter_size

# For faster checking, precompute for each pair (i,j) the intersection size
# When adding a new set k, we need to check all pairs (i,j) in current family:
#   _inter[i][j] == _inter[i][k] == _inter[j][k]
# This is O(|F|^2) per addition which is too slow for large families.

# Alternative: use a different approach.
# For each pair of intersection sizes (a, b), track which pairs in the family have that intersection.
# Actually, let's think differently.

# When we add set k to family F, we need to verify that no existing pair (i,j) in F
# has _inter[i][j] == _inter[i][k] == _inter[j][k].

# For a fixed k, group existing sets i in F by _inter[i][k].
# For each group G_t = {i in F : _inter[i][k] = t}, we need to check that no pair (i,j) in G_t
# has _inter[i][j] == t.

# So: for each t in {0,1,2,3}, look at the subfamily F_t = {i in F : _inter[i][k] = t}.
# If any pair (i,j) in F_t has _inter[i][j] == t, then adding k creates a sunflower.

# This is still O(|F|^2) in the worst case, but we can optimize by tracking for each t
# whether F_t contains a pair with intersection t.

# Let's use a different strategy: local search / greedy with randomization.

def _is_compatible(family_set, new_idx):
    """Check if adding new_idx to family_set (a set of indices) maintains the no-sunflower property."""
    # For each t in 0..3, collect indices in family_set with intersection t with new_idx
    groups = {t: [] for t in range(4)}
    for i in family_set:
        t = _inter[i][new_idx]
        if t < 4:  # t=4 means identical set, shouldn't happen
            groups[t].append(i)
    
    for t in range(4):
        g = groups[t]
        if len(g) < 2:
            continue
        # Check if any pair in g has intersection exactly t
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                if _inter[g[a]][g[b]] == t:
                    return False
    return True

def _greedy_randomized(deadline):
    """Randomized greedy: repeatedly pick a random valid set to add."""
    family = set()
    remaining = list(range(_M))
    random.shuffle(remaining)
    
    while remaining and time.time() < deadline:
        # Try to add a random element from remaining
        added = False
        for _ in range(len(remaining)):
            pos = random.randint(0, len(remaining) - 1)
            idx = remaining[pos]
            if _is_compatible(family, idx):
                family.add(idx)
                remaining.pop(pos)
                added = True
                break
        if not added:
            # No more can be added
            break
    
    return family

def _local_search(family, deadline):
    """Try to improve family by swapping one set out and another in."""
    family = set(family)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_list = list(family)
        random.shuffle(current_list)
        
        for remove_idx in current_list:
            if time.time() >= deadline:
                break
            trial_family = family - {remove_idx}
            # Try to add some set not in family
            candidates = [i for i in range(_M) if i not in family]
            random.shuffle(candidates)
            for add_idx in candidates:
                if _is_compatible(trial_family, add_idx):
                    family = trial_family | {add_idx}
                    improved = True
                    break
    
    return family

def _multi_start(deadline):
    """Run multiple greedy starts and local searches."""
    best = set()
    best_size = 0
    start_time = time.time()
    
    # Reserve 10% of time for safety
    safe_deadline = start_time + (deadline - start_time) * 0.85
    
    n_starts = 0
    while time.time() < safe_deadline:
        n_starts += 1
        # Randomized greedy
        fam = _greedy_randomized(safe_deadline)
        size = len(fam)
        
        if size > best_size:
            best = set(fam)
            best_size = size
        
        # Local search if time permits
        if time.time() < safe_deadline - 1.0 and size >= best_size:
            fam2 = _local_search(fam, safe_deadline)
            if len(fam2) > best_size:
                best = fam2
                best_size = len(fam2)
    
    return best

def construct():
    start = time.time()
    # Hard deadline: return before 85s to be safe (evaluator kills at 90s)
    deadline = start + 80.0
    
    # First, try a deterministic greedy to get a baseline
    # Sort by some heuristic: prefer sets that are "spread out"
    # For now, just do multi-start randomized
    
    best = _multi_start(deadline)
    
    # Convert to sorted tuples
    result = [sorted(_ALL[i]) for i in best]
    result.sort()
    
    return result
# EVOLVE-BLOCK-END
