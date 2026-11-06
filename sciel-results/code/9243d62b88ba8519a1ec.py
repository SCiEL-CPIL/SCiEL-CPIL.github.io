"""
Improved program: maximize reciprocal weight of 3-AP-free subset of [1,256].
Uses branch-and-bound on conflict graph with fast numpy-based conflict checking.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

N = 256
TIME_LIMIT = 58.0  # Safety margin before 60s kill

# Precompute conflict structure using numpy for speed
def precompute_conflict_bitsets():
    """
    For each x, precompute a numpy bitset of elements that conflict with x.
    Two elements a, b conflict with x if {x, a, b} forms a 3-AP.
    We use a different approach: for each pair (a,b), determine which x values they conflict with.
    Then build adjacency.
    """
    # conflict_mask[x] is a numpy uint64 array representing which elements conflict with x
    # Actually, let's use Python integers as bitmasks for simplicity and speed
    
    # For each x, find all pairs (a,b) such that {x,a,b} is a 3-AP
    # Then x conflicts with a and b (in the sense that if both a,b are in S, x cannot be added)
    # But for independent set formulation, we need: two elements are adjacent if they cannot both be in S.
    # Two elements a, b cannot both be in S if there exists x such that {a,b,x} is a 3-AP.
    # That is, if a+b = 2x for some x in [1,N], or a+x=2b, or b+x=2a.
    # This means: a and b conflict if (a+b) is even and (a+b)/2 is in [1,N], 
    #            or (a+b)/2 is an integer in [1,N]... wait.
    # Actually: {a,b,x} is a 3-AP iff one of them is the average of the other two.
    # So a and b are in a 3-AP together with some x iff:
    #   x = (a+b)/2 (if a+b is even) and x != a, x != b
    #   a = (b+x)/2 => x = 2a - b (if 1 <= 2a-b <= N and 2a-b != a, 2a-b != b)
    #   b = (a+x)/2 => x = 2b - a (if 1 <= 2b-a <= N and 2b-a != a, 2b-a != b)
    
    # So a and b are "conflicting" (cannot both be in a 3-AP-free set) if:
    #   There exists x in [1,N], x != a, x != b, such that {a,b,x} is a 3-AP.
    # This happens iff:
    #   (a+b) % 2 == 0 and 1 <= (a+b)//2 <= N and (a+b)//2 != a and (a+b)//2 != b
    #   OR 1 <= 2*a-b <= N and 2*a-b != a and 2*a-b != b
    #   OR 1 <= 2*b-a <= N and 2*b-a != a and 2*b-a != b
    
    # Let's build the conflict graph
    conflict = [0] * (N + 1)  # conflict[x] is a bitmask of elements that conflict with x
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # Check if {a, b, x} is a 3-AP for some x
            # Case 1: x = (a+b)/2
            if (a + b) % 2 == 0:
                x = (a + b) // 2
                if 1 <= x <= N and x != a and x != b:
                    # a and b conflict
                    conflict[a] |= (1 << (b - 1))
                    conflict[b] |= (1 << (a - 1))
            # Case 2: x = 2*a - b (a is middle)
            x = 2 * a - b
            if 1 <= x <= N and x != a and x != b:
                conflict[a] |= (1 << (b - 1))
                conflict[b] |= (1 << (a - 1))
            # Case 3: x = 2*b - a (b is middle)
            x = 2 * b - a
            if 1 <= x <= N and x != a and x != b:
                conflict[a] |= (1 << (b - 1))
                conflict[b] |= (1 << (a - 1))
    
    return conflict

CONFLICT = precompute_conflict_bitsets()

def is_conflict(x, y, mask):
    """Check if x and y are in conflict (cannot both be in a 3-AP-free set)."""
    # x and y conflict if there's a third element z such that {x,y,z} is a 3-AP
    # This is precomputed in CONFLICT
    return (CONFLICT[x] >> (y - 1)) & 1

def can_add_fast(x, selected_mask):
    """
    Check if x can be added to the set represented by selected_mask.
    x can be added iff for all y in selected_mask, x and y do not conflict.
    That is, conflict[x] & selected_mask == 0.
    """
    if (selected_mask >> (x - 1)) & 1:
        return False
    # Check if any element in selected_mask conflicts with x
    return (CONFLICT[x] & selected_mask) == 0

def reciprocal_sum_float(mask):
    """Compute reciprocal sum as float for fast comparison."""
    total = 0.0
    for i in range(1, N + 1):
        if (mask >> (i - 1)) & 1:
            total += 1.0 / i
    return total

def reciprocal_sum_exact(mask):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for i in range(1, N + 1):
        if (mask >> (i - 1)) & 1:
            total += Fraction(1, i)
    return total

def greedy_construct():
    """Greedy construction: add elements in order of decreasing weight (smallest x first)."""
    mask = 0
    for x in range(1, N + 1):
        if can_add_fast(x, mask):
            mask |= (1 << (x - 1))
    return mask

def local_search_1swap(initial_mask, time_limit):
    """
    1-swap local search: try removing one element and adding another.
    Uses float scoring for speed.
    """
    start_time = time.time()
    current_mask = initial_mask
    current_score = reciprocal_sum_float(current_mask)
    best_mask = current_mask
    best_score = current_score
    
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
    
    # Precompute list of elements in mask
    def get_elements(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]
    
    def get_candidates(mask):
        return [i + 1 for i in range(N) if not ((mask >> i) & 1)]
    
    # Phase 1: Add-only
    while True:
        if time.time() - start_time > time_limit:
            break
        improved = False
        candidates = get_candidates(current_mask)
        for x in sorted(candidates):  # Smallest x first (highest weight)
            if time.time() - start_time > time_limit:
                break
            if can_add_fast(x, current_mask):
                current_mask |= (1 << (x - 1))
                current_score += weights[x]
                improved = True
                if current_score > best_score:
                    best_mask = current_mask
                    best_score = current_score
        if not improved:
            break
    
    # Phase 2: 1-swap
    while True:
        if time.time() - start_time > time_limit:
            break
        improved = False
        elements = get_elements(current_mask)
        # Try removing larger elements first (lower weight, more likely to be swappable)
        elements.sort(reverse=True)
        
        for x_out in elements:
            if time.time() - start_time > time_limit:
                break
            
            mask_without = current_mask & ~(1 << (x_out - 1))
            score_without = current_score - weights[x_out]
            
            candidates = get_candidates(mask_without)
            # Try adding smaller elements first (higher weight)
            for y in sorted(candidates):
                if time.time() - start_time > time_limit:
                    break
                if y == x_out:
                    continue
                if can_add_fast(y, mask_without):
                    new_score = score_without + weights[y]
                    if new_score > current_score + 1e-12:
                        current_mask = mask_without | (1 << (y - 1))
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_mask = current_mask
                            best_score = current_score
                        break
        
        if not improved:
            break
    
    return best_mask

def branch_and_bound(maximize_weight, time_limit):
    """
    Branch and bound for maximum weight independent set.
    This is the key improvement: exact optimization on the conflict graph.
    """
    start_time = time.time()
    best_weight = 0.0
    best_mask = 0
    
    # Weights: weight[i] = 1/i for i in 1..N
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
    
    # Order vertices by weight (descending) for better pruning
    order = list(range(1, N + 1))
    order.sort(key=lambda x: -weights[x])
    
    # Precompute for each vertex, the list of neighbors (conflicting vertices)
    neighbors = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            if x != y and (CONFLICT[x] >> (y - 1)) & 1:
                neighbors[x].append(y)
    
    # For faster neighbor lookup, also store as bitmask
    neighbor_mask = [0] * (N + 1)
    for x in range(1, N + 1):
        for y in neighbors[x]:
            neighbor_mask[x] |= (1 << (y - 1))
    
    # Upper bound: sum of weights of all remaining candidates
    def compute_upper_bound(candidates_mask):
        total = 0.0
        for i in range(1, N + 1):
            if (candidates_mask >> (i - 1)) & 1:
                total += weights[i]
        return total
    
    # Branch and bound
    def bb(candidates_mask, selected_mask, current_weight):
        nonlocal best_weight, best_mask, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        # Prune: if current_weight + upper_bound <= best_weight, no need to explore
        ub = current_weight + compute_upper_bound(candidates_mask)
        if ub <= best_weight + 1e-12:
            return
        
        if candidates_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_mask = selected_mask
            return
        
        # Select next vertex to branch on
        # Pick the vertex with highest weight from candidates
        best_v = -1
        best_w = -1.0
        for v in order:
            if (candidates_mask >> (v - 1)) & 1:
                if weights[v] > best_w:
                    best_w = weights[v]
                    best_v = v
                    break
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        new_selected = selected_mask | (1 << (best_v - 1))
        # Remove best_v and all its neighbors from candidates
        new_candidates = candidates_mask & ~(1 << (best_v - 1)) & ~neighbor_mask[best_v]
        bb(new_candidates, new_selected, current_weight + weights[best_v])
        
        # Branch 2: Exclude best_v
        new_candidates = candidates_mask & ~(1 << (best_v - 1))
        bb(new_candidates, selected_mask, current_weight)
    
    # Start with all vertices as candidates
    all_candidates = 0
    for i in range(1, N + 1):
        all_candidates |= (1 << (i - 1))
    
    bb(all_candidates, 0, 0.0)
    
    return best_mask

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    greedy_mask = greedy_construct()
    greedy_score = reciprocal_sum_float(greedy_mask)
    
    # Phase 2: Local search (1-swap)
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 5.0:
        ls_mask = local_search_1swap(greedy_mask, min(remaining_time - 2.0, 10.0))
        ls_score = reciprocal_sum_float(ls_mask)
        if ls_score > greedy_score:
            greedy_mask = ls_mask
            greedy_score = ls_score
    
    # Phase 3: Branch and bound (exact optimization)
    remaining_time = TIME_LIMIT - (time.time() - start_time)
    if remaining_time > 10.0:
        bb_mask = branch_and_bound(None, remaining_time - 2.0)
        bb_score = reciprocal_sum_float(bb_mask)
        if bb_score > greedy_score:
            greedy_mask = bb_mask
            greedy_score = bb_score
    
    # Convert to sorted list
    result = [i + 1 for i in range(N) if (greedy_mask >> i) & 1]
    return sorted(result)

# EVOLVE-BLOCK-END
