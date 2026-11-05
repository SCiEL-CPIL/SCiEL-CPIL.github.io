"""Rewritten program for minimum maximal no-isosceles set on [0,16)^2."""
N = 16
TOTAL = N * N
import random
import time
import itertools
import math
import numpy as np

# Precompute isosceles conflicts
def _precompute_conflicts():
    """
    Precompute conflict masks using distance matrix.
    """
    points = list(itertools.product(range(N), repeat=2))
    pts = np.array(points)
    
    # Compute squared distance matrix D
    # D[i, j] = dist^2(points[i], points[j])
    # (pts[:, None, :] - pts[None, :, :])**2 sums over last axis
    D = np.sum((pts[:, None, :] - pts[None, :, :])**2, axis=2)
    
    conflict_mask = [0] * TOTAL
    
    # For each pair (i, j), find k such that {i, j, k} is isosceles
    # Condition: D[i,j] == D[i,k] or D[i,j] == D[j,k] or D[j,k] == D[i,k]
    # Note: D[i,i] = 0. We must exclude k=i, k=j.
    
    for i in range(TOTAL):
        for j in range(i + 1, TOTAL):
            dij = D[i, j]
            # Find all k such that isosceles condition holds
            # We iterate k. For speed, we can use numpy masks, but loop is O(N^3) which is ~16M.
            # Let's try to vectorize the k-loop for a fixed pair (i,j).
            
            # Condition 1: D[i, k] == dij  (k is on circle centered at i passing through j)
            # Condition 2: D[j, k] == dij  (k is on circle centered at j passing through i)
            # Condition 3: D[i, k] == D[j, k] (k is on perpendicular bisector of ij)
            
            # Using numpy for each pair might be slow due to overhead.
            # Pure python loop with precomputed D might be faster for N=256?
            # 16M iterations in Python is ~1-2s.
            
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                dik = D[i, k]
                djk = D[j, k]
                if dij == dik or dij == djk or dik == djk:
                    conflict_mask[i] |= (1 << j)
                    conflict_mask[j] |= (1 << i)
                    conflict_mask[i] |= (1 << k)
                    conflict_mask[k] |= (1 << i)
                    conflict_mask[j] |= (1 << k)
                    conflict_mask[k] |= (1 << j)
                    
    return points, conflict_mask

POINTS, CONFLICT_MASK = _precompute_conflicts()

def construct():
    """
    Construct a minimal maximal no-isosceles set.
    """
    deadline = time.time() + 280
    
    def is_maximal_mask(mask):
        """Check if mask is a maximal independent set."""
        # For each point not in mask, it must conflict with some point in mask
        # ~mask & TOTAL_MASK
        not_in_mask = ((1 << TOTAL) - 1) ^ mask
        while not_in_mask:
            lsb = not_in_mask & (-not_in_mask)
            p = lsb.bit_length() - 1
            if CONFLICT_MASK[p] & mask == 0:
                return False
            not_in_mask ^= lsb
        return True

    def greedy_construct(order):
        """Greedy construction. order is a list of indices."""
        mask = 0
        for p in order:
            if CONFLICT_MASK[p] & mask == 0:
                mask |= (1 << p)
        return mask

    def make_maximal(mask):
        """Add points until the set is maximal."""
        changed = True
        while changed:
            changed = False
            # Iterate through all points
            # Optimization: Only check points not in mask
            not_in_mask = ((1 << TOTAL) - 1) ^ mask
            while not_in_mask:
                lsb = not_in_mask & (-not_in_mask)
                p = lsb.bit_length() - 1
                if CONFLICT_MASK[p] & mask == 0:
                    mask |= (1 << p)
                    changed = True
                    not_in_mask ^= lsb # p is now in mask
                else:
                    not_in_mask ^= lsb
        return mask

    def local_search(mask):
        """
        Local search using Safe Removal.
        """
        if time.time() > deadline - 10:
            return mask
            
        improved = True
        max_iters = 1000
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            
            # Try to remove each point in S
            # Iterate over set bits of mask
            m = mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                
                # Check if v can be removed
                # v can be removed if for all u in N(v) \ S, N(u) \cap S != {v}
                # i.e., u is dominated by someone else in S.
                
                neighbors_not_in_S = CONFLICT_MASK[v] & ~mask
                can_remove = True
                u_mask = neighbors_not_in_S
                while u_mask:
                    ulsb = u_mask & (-u_mask)
                    u = ulsb.bit_length() - 1
                    # Check if u is dominated only by v in S
                    # N(u) & S should be {v}
                    if (CONFLICT_MASK[u] & mask) == (1 << v):
                        can_remove = False
                        break
                    u_mask ^= ulsb
                
                if can_remove:
                    mask ^= (1 << v)
                    improved = True
                    # Restart iteration to be safe, or continue?
                    # If we remove v, other points might become removable.
                    # Break and restart the while loop for correctness/simplicity
                    # Or just continue to next bit in original m?
                    # If we continue, we might miss interactions.
                    # But removing v only makes domination *weaker*, so it doesn't make
                    # other points *more* removable (it might make them *less* removable).
                    # So we can continue scanning the remaining points in the original mask?
                    # Wait, if we remove v, the set S shrinks.
                    # Points that were safe to remove might become unsafe.
                    # Points that were unsafe might become safe?
                    # If u was dominated by {v, w}, and we remove v, u is dominated by {w}.
                    # If w is removed later, u becomes undominated.
                    # So removing points doesn't create new removal opportunities for *other* points
                    # in terms of "is it safe to remove this specific point".
                    # Actually, if I remove v, the condition for removing w depends on S_new.
                    # S_new is smaller. So N(u) & S_new is smaller.
                    # If N(u) & S_old was {v, w}, and we remove v, N(u) & S_new is {w}.
                    # If we then check w, is it safe?
                    # u is dominated by w. If we remove w, u is undominated. So w is NOT safe.
                    # So removing points makes it *harder* to remove others.
                    # So we can just continue scanning the rest of the bits in the original `m`?
                    # No, `m` is a local copy. We should probably restart to be sure we find all removable points.
                    # But for speed, we can just clear the current bit and continue.
                    # Let's restart to be safe and simple.
                    break 
                
                m ^= lsb
            
        return mask

    def mask_to_points(mask):
        """Convert mask to list of (x,y) points."""
        result = []
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            result.append(POINTS[idx])
            m ^= lsb
        return result

    best = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    # Precompute total mask
    TOTAL_MASK = (1 << TOTAL) - 1
    
    while time.time() < deadline - 30:
        # Generate a random order for greedy construction
        order = list(range(TOTAL))
        rng.shuffle(order)
        
        # Greedy construction
        mask = greedy_construct(order)
        
        # Ensure maximality
        mask = make_maximal(mask)
        
        # Local search to minimize
        mask = local_search(mask)
        
        if mask.bit_count() < best_len:
            best = mask
            best_len = mask.bit_count()
        
        # Early termination if we find a very good solution
        if best_len <= 8:
            break

    if best is None:
        # Fallback: simple greedy
        order = list(range(TOTAL))
        best = greedy_construct(order)
        best = make_maximal(best)
    
    return mask_to_points(best)
