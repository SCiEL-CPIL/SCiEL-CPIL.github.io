import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # All weight-4 vectors in {-1,0,1}^11 are mutually compatible (max dot product 2 <= 4/2).
    # All weight-1 vectors in {-2,0,2}^11 are mutually compatible (max dot product 4 <= 4/2? No.
    # Wait. v1=(2,0..), v1'=(2,0..). Distinct?
    # If v1=(2,0..), v1'=(-2,0..). Dot -4. Dist sq 16. Norm sq 4. 16>=4. OK.
    # If v1=(2,0..), v1'=(0,2..). Dot 0. Dist sq 8. 8>=4. OK.
    # Max dot for distinct v1 is 0?
    # v1=(2,0..), v1'=(2,0..) is same.
    # So yes, all v1 are compatible.
    
    # Check compatibility between v4 and v1.
    # u (w4, norm sq 4), v (w1, norm sq 4).
    # Condition: 4+4 - 2 u.v >= 4 => u.v <= 2.
    # Max dot: u=(1,1,1,1,0..), v=(2,0,0,0,0..). Dot 2.
    # 2 <= 2. OK.
    # So ALL v4 and v1 are mutually compatible.
    
    final_vectors = np.vstack([vectors_v4, vectors_v1])
    
    return final_vectors
