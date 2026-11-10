import numpy as np

def kissing_number11() -> np.ndarray:
    points = []
    
    # Generate scaled E8 minimal vectors (first 8 dimensions)
    # Type 1: +/- 2 e_i +/- 2 e_j
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    vec = np.zeros(11, dtype=np.int64)
                    vec[i] = 2 * s1
                    vec[j] = 2 * s2
                    points.append(vec)
                    
    # Type 2: +/- 1, ..., +/- 1 with even number of -1s
    # There are 2^7 = 128 such vectors
    # Iterate over all 2^8 sign combinations, check if number of -1s is even
    for mask in range(1 << 8):
        vec = np.zeros(11, dtype=np.int64)
        num_neg = 0
        for i in range(8):
            if (mask >> i) & 1:
                vec[i] = -1
                num_neg += 1
            else:
                vec[i] = 1
        if num_neg % 2 == 0:
            points.append(vec)
            
    # Generate scaled D3 minimal vectors (last 3 dimensions)
    # D3 minimal vectors are +/- e_i +/- e_j for i<j in {0,1,2}
    # Scaled by 2: +/- 2 e_i +/- 2 e_j
    for i in range(3):
        for j in range(i + 1, 3):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    vec = np.zeros(11, dtype=np.int64)
                    # Place in dimensions 8, 9, 10
                    vec[8 + i] = 2 * s1
                    vec[8 + j] = 2 * s2
                    points.append(vec)
                    
    return np.array(points, dtype=np.int64)
