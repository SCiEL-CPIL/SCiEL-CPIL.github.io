"""Improved program: greedy set packing with bitmask optimization and randomized restarts."""
import time
import random
from itertools import combinations

# Precompute all 4-element subsets of [1,16] as bitmasks
N = 16
K = 4
ALL_SETS = []
SET_TO_INDEX = {}

def _precompute():
    for combo in combinations(range(1, N + 1), K):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        ALL_SETS.append(mask)
        SET_TO_INDEX[mask] = len(ALL_SETS) - 1

_precompute()
NUM_SETS = len(ALL_SETS)  # C(16,4) = 1820

# Precompute pairwise intersection sizes and whether two sets form a "pair"
# For a sunflower triple {A, B, C}: all three pairwise intersections must be equal.
# i.e., A∩B == A∩C == B∩C
# This includes the empty core case.

# Precompute for each pair of sets, the intersection mask
# and also whether a third set could complete a sunflower with this pair.

# For efficiency: given two selected sets A and B, a third set C conflicts if:
#   A∩B == A∩C == B∩C
# Let I = A∩B. Then we need A∩C == I and B∩C == I.
# This means C must contain I, and C must not contain (A\I) ∪ (B\I) except possibly...
# Actually: A∩C = I means C contains all of I and contains none of A\I.
# B∩C = I means C contains all of I and contains none of B\I.
# So C must contain I, and C must be disjoint from (A\I) ∪ (B\I).
# Also C is a 4-element set.

# Let's precompute for each pair (i, j) the "forbidden mask" for a third set:
# must_contain = I (the intersection)
# must_avoid = (A_i \ I) | (A_j \ I)  = A_i | A_j (since I = A_i ∩ A_j)
# Wait: A_i \ I are elements in A_i not in A_j. Similarly for B.
# So must_avoid = (A_i & ~I) | (A_j & ~I) = A_i | A_j | ... no.
# A_i \ I = A_i & ~A_j (since I = A_i ∩ A_j)
# So must_avoid = (A_i & ~A_j) | (A_j & ~A_i) = A_i XOR A_j (symmetric difference)

# So for pair (i,j): 
#   must_contain_mask = A_i & A_j
#   must_avoid_mask = (A_i ^ A_j)  # symmetric difference
# A set C is forbidden if:
#   (C & must_contain_mask) == must_contain_mask  (C contains all of I)
#   (C & must_avoid_mask) == 0  (C is disjoint from sym diff)
#   AND C != A_i and C != A_j (distinct sets)

# Precompute these for all pairs
pair_must_contain = [[0] * NUM_SETS for _ in range(NUM_SETS)]
pair_must_avoid = [[0] * NUM_SETS for _ in range(NUM_SETS)]

for i in range(NUM_SETS):
    for j in range(i + 1, NUM_SETS):
        mi = ALL_SETS[i]
        mj = ALL_SETS[j]
        ic = mi & mj
        av = mi ^ mj
        pair_must_contain[i][j] = ic
        pair_must_avoid[i][j] = av
        pair_must_contain[j][i] = ic
        pair_must_avoid[j][i] = av

def is_valid_addition(candidate_mask, selected_indices):
    """Check if adding candidate_mask to selected set creates a sunflower triple."""
    for idx in selected_indices:
        s_mask = ALL_SETS[idx]
        # Check if candidate and s_mask with any other selected form a sunflower
        # Actually we need to check all pairs among (selected ∪ {candidate})
        # But more efficiently: for each pair (candidate, s1) and (candidate, s2) and (s1, s2)
        # The new triples involving candidate are: (candidate, s1, s2) for all pairs s1, s2 in selected
        # So we need to check: for each pair (s1, s2) in selected, does {candidate, s1, s2} form a sunflower?
        # Also we need to check pairs (candidate, s1) with... no, a sunflower requires 3 sets.
        # So only triples where candidate is one of the three.
        # For each pair (s1, s2) in selected: check if candidate, s1, s2 form a sunflower.
        pass
    # This is O(n^2) per addition which is too slow. Let's use a different approach.
    return True

# Better approach: maintain for each pair of selected sets, and when adding a new set,
# check all pairs involving the new set against the constraint.
# Actually, the sunflower condition is: no three distinct sets A, B, C have A∩B = A∩C = B∩C.
# When we add a new set C, we need to check for all pairs (A, B) in the current selected set:
# does C form a sunflower with A and B?
# i.e., A∩B == A∩C == B∩C

def can_add(candidate_mask, selected_masks):
    """Check if candidate_mask can be added to selected_masks without creating a sunflower."""
    for i in range(len(selected_masks)):
        a = selected_masks[i]
        for j in range(i + 1, len(selected_masks)):
            b = selected_masks[j]
            # Check if {candidate, a, b} forms a sunflower
            ab = a & b
            ac = a & candidate_mask
            bc = b & candidate_mask
            if ab == ac == bc:
                return False
    return True

def greedy_randomized(seed_val, time_limit):
    """Run one greedy pass with randomization."""
    rng = random.Random(seed_val)
    selected = []
    remaining = list(range(NUM_SETS))
    rng.shuffle(remaining)
    
    # Use a priority-based approach: at each step, pick the set with minimum conflicts
    # But for speed, do a simple greedy with random order and local improvements
    
    start_time = time.time()
    
    for idx in remaining:
        if time.time() - start_time > time_limit:
            break
        mask = ALL_SETS[idx]
        if can_add(mask, [ALL_SETS[s] for s in selected]):
            selected.append(idx)
    
    return selected

def greedy_min_degree(time_limit):
    """Greedy: always pick the set with fewest conflicts with already selected."""
    selected = []
    remaining = set(range(NUM_SETS))
    start_time = time.time()
    
    while remaining:
        if time.time() - start_time > time_limit:
            break
        best_idx = None
        best_conflicts = float('inf')
        for idx in remaining:
            mask = ALL_SETS[idx]
            conflicts = 0
            for s in selected:
                a = ALL_SETS[s]
                # Count how many pairs in selected would form a sunflower with mask and a
                for t in selected:
                    if t == s:
                        continue
                    b = ALL_SETS[t]
                    ab = a & b
                    ac = a & mask
                    bc = b & mask
                    if ab == ac == bc:
                        conflicts += 1
                        break
            if conflicts == 0:
                best_idx = idx
                best_conflicts = 0
                break
            if conflicts < best_conflicts:
                best_conflicts = conflicts
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)
    
    return selected

def local_improvement(selected, time_limit):
    """Try to improve the solution by swapping or adding sets."""
    selected_set = set(selected)
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try adding any unselected set
        for idx in range(NUM_SETS):
            if idx in selected_set:
                continue
            mask = ALL_SETS[idx]
            if can_add(mask, [ALL_SETS[s] for s in selected]):
                selected.append(idx)
                selected_set.add(idx)
                improved = True
                break
        # Try swapping: remove one, add another
        if not improved:
            for i in range(len(selected)):
                for j in range(NUM_SETS):
                    if j in selected_set:
                        continue
                    new_selected = [s for k, s in enumerate(selected) if k != i]
                    if can_add(ALL_SETS[j], [ALL_SETS[s] for s in new_selected]):
                        selected[i] = j
                        selected_set.discard(selected[i])
                        selected_set.add(j)
                        improved = True
                        break
                if improved:
                    break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best_solution = []
    
    # Run multiple strategies
    # Strategy 1: Greedy min-degree
    if time.time() < deadline:
        sol = greedy_min_degree(deadline - time.time() - 1)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Strategy 2: Multiple randomized greedy passes
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        remaining_time = deadline - time.time() - 1
        if remaining_time < 0.5:
            break
        sol = greedy_randomized(r, remaining_time)
        if len(sol) > len(best_solution):
            best_solution = sol
    
    # Local improvement on best solution
    if time.time() < deadline and best_solution:
        best_solution = local_improvement(best_solution, deadline - time.time() - 1)
    
    # Convert to sorted lists
    result = [sorted([x for x in range(1, 17) if (ALL_SETS[idx] >> x) & 1]) for idx in best_solution]
    return result
