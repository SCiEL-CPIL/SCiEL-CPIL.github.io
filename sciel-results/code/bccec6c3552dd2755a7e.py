import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate V1 vectors (22 total)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Generate V4 vectors using Angular Sectoring (1 per group)
    # Group defined by first 3 non-zero positions and their signs.
    # For each group, pick the vector with the smallest 4th position index and sign +1.
    
    vectors_v4 = []
    
    # Iterate through all possible "First 3" configurations
    for pos3 in itertools.combinations(range(D), 3):
        p1, p2, p3 = pos3
        for signs3 in itertools.product([1, -1], repeat=3):
            s1, s2, s3 = signs3
            
            # Find the smallest 4th position index > p3
            # It must be in range(D) and not in pos3
            # Actually, any position > p3 is a candidate for the 4th position
            # To be deterministic and simple, pick the smallest available index > p3
            # If none, skip? No, there are 11 positions.
            # pos3 has 3. Remaining 8.
            # Smallest index > p3 might not exist if p3=10.
            # If p3=10, no index > 10.
            # But the 4th position just needs to be distinct from the first 3.
            # It doesn't have to be > p3.
            # However, "first 3 non-zero positions" are the smallest indices in the support.
            # So the 4th position MUST be > p3.
            # If p3=10, there is no 4th position > 10.
            # So vectors with first 3 positions ending in 10 cannot exist?
            # Wait. If support is {1, 2, 10}, the first 3 are 1, 2, 10.
            # The 4th position must be > 10? No, the support has 4 elements.
            # If the first 3 are 1, 2, 10, the 4th element must be > 10?
            # No, the first 3 are the smallest 3.
            # So if 10 is the 3rd smallest, the 4th must be > 10.
            # Since max index is 10, there is no 4th position.
            # So any vector with 10 as the 3rd smallest index is impossible?
            # Let's check.
            # Support size 4.
            # Sorted support $i_1 < i_2 < i_3 < i_4$.
            # First 3 are $i_1, i_2, i_3$.
            # We need $i_4 \le 10$.
            # So $i_3$ must be at most 9.
            # If $i_3 = 10$, then $i_4 > 10$, impossible.
            # So we only iterate pos3 where max(pos3) < 10.
            
            if p3 == 10:
                continue
                
            # Find smallest index > p3
            p4 = p3 + 1
            # Check if p4 is in pos3? No, p4 > p3 >= max(pos3) is false?
            # p4 = p3 + 1. Since p3 is the max of pos3, p4 is not in pos3.
            # And p4 <= 10.
            
            # Construct vector
            v = [0] * D
            v[p1] = s1
            v[p2] = s2
            v[p3] = s3
            v[p4] = 1  # Fix sign to +1 for determinism
            
            vectors_v4.append(v)
            
    vectors_v4 = np.array(vectors_v4, dtype=np.int64)
    
    # Combine
    final_vectors = np.vstack([vectors_v4, vectors_v1])
    
    return final_vectors
