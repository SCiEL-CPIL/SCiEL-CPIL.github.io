# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy algorithm with bitmask tracking.
    
    The goal is to maximize |S| = 2401 - |N|, where S is the set of "excluded" points.
    Condition: For every x in F_49^2, there is a line L through x such that L \ {x} is disjoint from S.
    This is equivalent to: For every x, the set of directions v such that L(x, v) \ {x} intersects S is not all 50 directions.
    
    We greedily add points to S. Adding p to S blocks direction v for x if p lies on L(x, v) and p != x.
    We track blocked directions for each x using a 64-bit integer bitmask.
    """
    p_prime = 7
    q = 49
    
    # Field arithmetic for F_49 = F_7[alpha] with alpha^2 = 3
    # Element i is (i//7, i%7) = a + b*alpha
    
    def idx_to_ab(i):
        return (i // 7, i % 7)
    
    def ab_to_idx(a, b):
        return a * 7 + b
    
    # Precompute addition and multiplication tables
    add_table = np.zeros((49, 49), dtype=np.int32)
    mul_table = np.zeros((49, 49), dtype=np.int32)
    inv_table = np.zeros(49, dtype=np.int32)
    
    for i in range(49):
        a1, b1 = idx_to_ab(i)
        for j in range(49):
            a2, b2 = idx_to_ab(j)
            # Addition
            sa = (a1 + a2) % 7
            sb = (b1 + b2) % 7
            add_table[i, j] = ab_to_idx(sa, sb)
            
            # Multiplication
            pa = (a1 * a2 + 3 * b1 * b2) % 7
            pb = (a1 * b2 + a2 * b1) % 7
            mul_table[i, j] = ab_to_idx(pa, pb)
            
    # Inverses
    inv_table[0] = 0
    for i in range(1, 49):
        a1, b1 = idx_to_ab(i)
        norm = (a1 * a1 - 3 * b1 * b1) % 7
        # norm should be non-zero for i != 0
        norm_inv = 1
        while (norm * norm_inv) % 7 != 1:
            norm_inv += 1
        # inv(a + b*alpha) = (a - b*alpha) * norm_inv
        ia = (a1 * norm_inv) % 7
        ib = (-b1 * norm_inv) % 7
        inv_table[i] = ab_to_idx(ia, ib)
        
    # Precompute direction from x to p for all x, p
    # dir[x, p] is the index of the direction (0..48 for slopes, 49 for vertical)
    # If x == p, direction is undefined, we don't use it.
    # Directions:
    # v = 0..48: slope m in F_49
    # v = 49: vertical
    # Line through x with slope m: y = m(x - x0) + y0.
    # Direction vector (1, m).
    # Vertical: x = x0. Direction vector (0, 1).
    
    # To find direction from x to p:
    # If x1 == p1, direction is vertical (49).
    # Else, m = (p2 - x2) * (p1 - x1)^-1.
    # m is an element of F_49, so index 0..48.
    
    # We store points as indices 0..2400.
    # Point i -> (xa, xb, ya, yb)
    # xa = (i // 343) % 7
    # xb = (i // 49) % 7
    # ya = (i // 7) % 7
    # yb = i % 7
    
    # Precompute coordinates for all points
    coords = np.zeros((2401, 4), dtype=np.int32)
    for i in range(2401):
        coords[i, 0] = (i // 343) % 7
        coords[i, 1] = (i // 49) % 7
        coords[i, 2] = (i // 7) % 7
        coords[i, 3] = i % 7
        
    # Precompute direction indices
    # dir_idx[x, p] = direction index
    dir_idx = np.zeros((2401, 2401), dtype=np.int32)
    for x in range(2401):
        xa, xb, ya, yb = coords[x]
        xa_f = ab_to_idx(xa, xb)
        ya_f = ab_to_idx(ya, yb)
        for p in range(2401):
            if x == p:
                dir_idx[x, p] = -1
                continue
            pa, pb, pc, pd = coords[p]
            pa_f = ab_to_idx(pa, pb)
            pc_f = ab_to_idx(pc, pd)
            
            # dx = pa - xa, dy = pc - ya
            dx = add_table[pa_f, inv_table[xa_f]] if xa_f != 0 else pa_f # Wait, subtraction is add with inverse?
            # Subtraction: a - b = a + (-b). In F_7, -b = 6*b? No, additive inverse.
            # Let's just use the tables properly.
            # We need negation table too.
            
    # Let's re-do precomputation with negation
    neg_table = np.zeros(49, dtype=np.int32)
    for i in range(49):
        a1, b1 = idx_to_ab(i)
        na = (-a1) % 7
        nb = (-b1) % 7
        neg_table[i] = ab_to_idx(na, nb)
        
    # Recalculate dir_idx
    for x in range(2401):
        xa, xb, ya, yb = coords[x]
        xa_f = ab_to_idx(xa, xb)
        ya_f = ab_to_idx(ya, yb)
        neg_xa_f = neg_table[xa_f]
        neg_ya_f = neg_table[ya_f]
        
        for p in range(2401):
            if x == p:
                dir_idx[x, p] = -1
                continue
            pa, pb, pc, pd = coords[p]
            pa_f = ab_to_idx(pa, pb)
            pc_f = ab_to_idx(pc, pd)
            
            # dx = pa_f - xa_f = pa_f + neg_xa_f
            dx = add_table[pa_f, neg_xa_f]
            # dy = pc_f - ya_f = pc_f + neg_ya_f
            dy = add_table[pc_f, neg_ya_f]
            
            if dx == 0:
                dir_idx[x, p] = 49
            else:
                dx_inv = inv_table[dx]
                m = mul_table[dy, dx_inv]
                dir_idx[x, p] = m
                
    # Greedy algorithm
    in_S = np.zeros(2401, dtype=bool)
    blocked_mask = np.zeros(2401, dtype=np.int64)
    
    rng = np.random.RandomState(42)
    candidates = np.arange(2401)
    rng.shuffle(candidates)
    
    start_time = time.time()
    deadline = start_time + 550
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Tentatively add p
        # For each x != p, check if direction dir_idx[x, p] is newly blocked
        # If so, set bit. If count becomes 50, revert.
        
        changes = []
        valid = True
        
        for x in range(2401):
            if x == p:
                continue
                
            v = dir_idx[x, p]
            # v is 0..49
            bit = 1 << v
            
            if not (blocked_mask[x] & bit):
                blocked_mask[x] |= bit
                changes.append(x)
                if blocked_mask[x].bit_count() == 50:
                    valid = False
                    break
        
        if valid:
            in_S[p] = True
        else:
            # Revert
            for x in changes:
                v = dir_idx[x, p]
                bit = 1 << v
                blocked_mask[x] &= ~bit
                
    # Construct N
    # N is complement of S
    n_points = np.sum(~in_S)
    
    result_list = []
    for i in range(2401):
        if not in_S[i]:
            xa, xb, ya, yb = coords[i]
            result_list.append([
                [xa, xb],
                [ya, yb]
            ])
            
    if len(result_list) == 0:
        return np.empty((0, 2, 2), dtype=np.int64)
        
    result = np.array(result_list, dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
