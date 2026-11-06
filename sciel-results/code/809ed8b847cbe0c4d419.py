"""Improved program for minimum maximal no-isosceles set using Reduction and Precomputation."""
import itertools
import random
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)
POINT_TO_INDEX = {p: i for i, p in enumerate(POINTS)}

# Precompute isosceles triples
# iso_mask[a][b] will be a set (or bitmask) of indices p such that (POINTS[a], POINTS[b], POINTS[p]) is isosceles
# We can use a list of sets for simplicity, or bitmasks for speed.
# Let's use bitmasks (integers) for fast union operations.

def _compute_isosceles_masks():
    masks = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        a = POINTS[i]
        for j in range(i + 1, NUM_POINTS):
            b = POINTS[j]
            mask = 0
            # d_ab is fixed
            dx_ab = a[0] - b[0]
            dy_ab = a[1] - b[1]
            d_ab_sq = dx_ab**2 + dy_ab**2
            
            for k in range(NUM_POINTS):
                p = POINTS[k]
                if p == a or p == b:
                    continue
                # Check if (a,b,p) is isosceles
                # Distances: d_ap, d_bp, d_ab
                dx_ap = a[0] - p[0]
                dy_ap = a[1] - p[1]
                d_ap_sq = dx_ap**2 + dy_ap**2
                
                dx_bp = b[0] - p[0]
                dy_bp = b[1] - p[1]
                d_bp_sq = dx_bp**2 + dy_bp**2
                
                if d_ap_sq == d_ab_sq or d_bp_sq == d_ab_sq or d_ap_sq == d_bp_sq:
                    mask |= (1 << k)
            masks[i][j] = mask
            masks[j][i] = mask
    return masks

ISO_MASKS = _compute_isosceles_masks()

def _is_maximal_fast(chosen_indices):
    """Check if the set of indices is maximal using precomputed masks."""
    if not chosen_indices:
        return NUM_POINTS == 0 # Empty set is maximal only if no points exist? 
                               # Actually, if S is empty, is it maximal? 
                               # "No further grid point can be added". 
                               # If S is empty, we can add any point (since no isosceles formed yet). 
                               # So empty set is NOT maximal unless N=0.
                               # Wait, the definition: "maximal no-isosceles set". 
                               # If S is empty, it is no-isosceles. Can we add a point? Yes. So not maximal.
    
    # Union of all masks for pairs in chosen
    blocked_mask = 0
    chosen_list = list(chosen_indices)
    for i in range(len(chosen_list)):
        a = chosen_list[i]
        for j in range(i + 1, len(chosen_list)):
            b = chosen_list[j]
            blocked_mask |= ISO_MASKS[a][b]
    
    # Check if all points not in chosen are blocked
    # Create mask of chosen points
    chosen_mask = 0
    for idx in chosen_indices:
        chosen_mask |= (1 << idx)
        
    # All points mask
    all_mask = (1 << NUM_POINTS) - 1
    
    # Unchosen points
    unchosen_mask = all_mask & ~chosen_mask
    
    # If any unchosen point is not blocked, it's not maximal
    # Blocked points are those in blocked_mask.
    # We need unchosen_mask to be a subset of blocked_mask.
    # i.e., unchosen_mask & ~blocked_mask == 0
    
    return (unchosen_mask & ~blocked_mask) == 0

def _greedy_construction_fast(order):
    """Construct a maximal no-isosceles set using the given order of indices."""
    chosen = []
    chosen_set = set()
    
    for p_idx in order:
        # Check if p_idx can be added (no isosceles with any pair in chosen)
        can_add = True
        for i in range(len(chosen)):
            a = chosen[i]
            # If p, a, and any other b in chosen form isosceles?
            # No, we just need to ensure {p, a, b} is not isosceles for any b in chosen.
            # Wait, the condition is "no three points in the set form an isosceles".
            # So when adding p, we must check if p forms an isosceles with ANY pair (a,b) in chosen.
            # This is equivalent to: for all b in chosen, {p, a, b} is not isosceles?
            # No, we check if there exists b in chosen such that {p, a, b} is isosceles.
            # If such b exists, we cannot add p.
            
            # Using precomputed masks:
            # ISO_MASKS[a][p] contains all points q such that {a, p, q} is isosceles.
            # If any b in chosen is in ISO_MASKS[a][p], then adding p would create an isosceles with a and b.
            # So we check if (ISO_MASKS[a][p] & chosen_mask) != 0.
            
            # Optimization: We can just check against the current chosen set.
            # Let's stick to the simple check for now, or use the mask.
            # Let's use the mask for speed.
            pass
        
        # Actually, simpler check:
        # For each pair (a, b) in chosen, check if p is in ISO_MASKS[a][b].
        # If yes, conflict.
        # This is O(|chosen|^2) per point.
        
        conflict = False
        for i in range(len(chosen)):
            a = chosen[i]
            for j in range(i + 1, len(chosen)):
                b = chosen[j]
                if (ISO_MASKS[a][b] >> p_idx) & 1:
                    conflict = True
                    break
            if conflict:
                break
        
        if not conflict:
            chosen.append(p_idx)
            
    return chosen

def _reduce_set(chosen_indices):
    """Try to remove points from the maximal set while keeping it maximal."""
    chosen_list = list(chosen_indices)
    changed = True
    while changed:
        changed = False
        # Sort by some heuristic? Maybe points that block the fewest others?
        # Or just try all.
        for i in range(len(chosen_list)):
            # Try removing chosen_list[i]
            candidate = chosen_list[:i] + chosen_list[i+1:]
            if _is_maximal_fast(candidate):
                chosen_list = candidate
                changed = True
                break # Restart loop as list size changed
    return chosen_list

def construct():
    best_set = None
    best_size = float('inf')
    
    num_restarts = 100
    all_indices = list(range(NUM_POINTS))
    
    for _ in range(num_restarts):
        random.shuffle(all_indices)
        candidate = _greedy_construction_fast(all_indices)
        
        # Ensure it's maximal (Greedy should produce maximal, but let's be safe? 
        # Actually, greedy adds if possible. At the end, any point not added was blocked by the set AT THAT TIME.
        # Since the set only grows, it is still blocked by the final set. So yes, it is maximal.)
        
        # Apply reduction
        reduced_candidate = _reduce_set(candidate)
        
        if len(reduced_candidate) < best_size:
            best_size = len(reduced_candidate)
            best_set = [POINTS[idx] for idx in reduced_candidate]

    return best_set
