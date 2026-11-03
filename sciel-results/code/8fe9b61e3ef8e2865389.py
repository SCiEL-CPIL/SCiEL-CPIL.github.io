import numpy as np
import itertools
import time

D = 11

def generate_vectors(weight):
    vectors = []
    indices = list(range(D))
    for comb in itertools.combinations(indices, weight):
        for signs in itertools.product([1, -1], repeat=weight):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            vectors.append(tuple(v))
    return vectors

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Generate weight 4 and 5 vectors
    # Weight 4: Norm sqrt(4)=2. Constraint: u.v <= 2
    # Weight 5: Norm sqrt(5). Constraint: u.v <= 2 (if mixed with w4, max norm sqrt(5)? No, max norm is sqrt(5) if w5 is present.
    # If we have w4 and w5, max norm is sqrt(5).
    # Condition: dist >= sqrt(5).
    # w4-w4: 4+4-2u.v >= 5 => 3 >= 2u.v => u.v <= 1.5 => 1.
    # Wait. If max norm is sqrt(5), then w4-w4 must satisfy dist >= sqrt(5).
    # ||u-v||^2 = 8 - 2u.v >= 5 => 2u.v <= 3 => u.v <= 1.5 => 1.
    # So if we mix w4 and w5, the constraint on w4-w4 becomes u.v <= 1.
    # This is TIGHTER than w4-w4 alone (which was <= 2).
    # So mixing w4 and w5 hurts the w4-w4 packing.
    
    # Let's re-evaluate.
    # If we use ONLY weight 4 vectors:
    # Max norm = 2.
    # Condition: ||u-v|| >= 2.
    # 4+4-2u.v >= 4 => 2u.v <= 4 => u.v <= 2.
    # This is the loosest constraint for weight 4.
    
    # If we use ONLY weight 5 vectors:
    # Max norm = sqrt(5) ~ 2.236.
    # Condition: ||u-v|| >= sqrt(5).
    # 5+5-2u.v >= 5 => 5 >= 2u.v => u.v <= 2.5 => 2.
    # So w5-w5 constraint is u.v <= 2.
    # Same as w4-w4 (with max norm 2).
    # But w5 has a larger pool.
    
    # If we mix w4 and w5:
    # Max norm = sqrt(5).
    # w4-w4: u.v <= 1.
    # w4-w5: 4+5-2u.v >= 5 => 4 >= 2u.v => u.v <= 2.
    # w5-w5: u.v <= 2.
    # The w4-w4 constraint becomes tighter (<=1 instead of <=2).
    # This might reduce the number of w4 vectors we can pack.
    # However, we can pack more w5 vectors.
    # Which is better?
    # Pool w4: 5280. Constraint <= 2.
    # Pool w5: 14784. Constraint <= 2.
    # If we just use w5, we have a larger pool with the same constraint.
    # So w5-only is likely better than w4-only.
    # And w5-only is likely better than mixed (because mixed tightens w4-w4).
    
    # What about weight 6?
    # Max norm sqrt(6) ~ 2.45.
    # w6-w6: 6+6-2u.v >= 6 => 6 >= 2u.v => u.v <= 3.
    # Pool w6: 29568. Constraint <= 3.
    # This is LOOSER than w5 (<=2).
    # And the pool is larger.
    # So w6-only might be even better!
    
    # Let's check w7.
    # Max norm sqrt(7) ~ 2.64.
    # w7-w7: 7+7-2u.v >= 7 => 7 >= 2u.v => u.v <= 3.5 => 3.
    # Pool w7: C(11,7)*128 = 330*128 = 42240. Constraint <= 3.
    # Same constraint as w6, larger pool.
    
    # w8.
    # Max norm sqrt(8) ~ 2.82.
    # w8-w8: 8+8-2u.v >= 8 => 8 >= 2u.v => u.v <= 4.
    # Pool w8: C(11,8)*256 = 165*256 = 42240. Constraint <= 4.
    
    # w9.
    # Max norm sqrt(9)=3.
    # w9-w9: 9+9-2u.v >= 9 => 9 >= 2u.v => u.v <= 4.5 => 4.
    # Pool w9: C(11,9)*512 = 55*512 = 28160. Constraint <= 4.
    
    # w10.
    # Max norm sqrt(10) ~ 3.16.
    # w10-w10: 10+10-2u.v >= 10 => 10 >= 2u.v => u.v <= 5.
    # Pool w10: C(11,10)*1024 = 11*1024 = 11264. Constraint <= 5.
    
    # w11.
    # Max norm sqrt(11) ~ 3.31.
    # w11-w11: 11+11-2u.v >= 11 => 11 >= 2u.v => u.v <= 5.5 => 5.
    # Pool w11: 1*2048 = 2048. Constraint <= 5.

    # It seems higher weights allow looser constraints (higher dot product allowed) and have large pools.
    # The constraint is roughly u.v <= weight/2.
    # For w4, limit 2.
    # For w8, limit 4.
    # The "density" of allowed vectors increases.
    # So we should use high weight vectors.
    # Let's try weight 8 or 9 or 10.
    # Weight 10 pool is 11264. Constraint <= 5.
    # Weight 8 pool is 42240. Constraint <= 4.
    
    # Let's try to pack weight 8 vectors.
    # Or mix high weights?
    # If we mix w8 and w10, max norm is sqrt(10).
    # w8-w8: 8+8-2u.v >= 10 => 6 >= 2u.v => u.v <= 3. (Tighter than 4)
    # w8-w10: 8+10-2u.v >= 10 => 8 >= 2u.v => u.v <= 4.
    # w10-w10: u.v <= 5.
    # Mixing tightens the lower weight vectors.
    # So it's best to use a SINGLE weight with the largest pool and loosest constraint.
    # w8 has pool 42240, constraint 4.
    # w9 has pool 28160, constraint 4.
    # w10 has pool 11260, constraint 5.
    # w7 has pool 42240, constraint 3.
    
    # w8 seems best: large pool, loose constraint.
    # Let's generate weight 8 vectors and run greedy.
    
    candidates = generate_vectors(8)
    np.random.shuffle(candidates)
    
    selected = []
    selected_arr = np.array([]).reshape(0, D)
    
    for v in candidates:
        v_arr = np.array(v)
        if len(selected_arr) == 0:
            selected.append(v)
            selected_arr = v_arr.reshape(1, D)
        else:
            # Check dot products
            dots = selected_arr @ v_arr
            if np.all(dots <= 4):
                selected.append(v)
                selected_arr = np.vstack([selected_arr, v_arr.reshape(1, D)])
                
        if time.time() > deadline:
            break
            
    return np.array(selected, dtype=np.int64)
