import numpy as np

def construct():
    D = 11
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            # 4 combinations of signs for positions i and j
            # 1, 1
            v1 = np.zeros(D, dtype=np.int64)
            v1[i] = 1
            v1[j] = 1
            vectors.append(v1)
            
            # 1, -1
            v2 = np.zeros(D, dtype=np.int64)
            v2[i] = 1
            v2[j] = -1
            vectors.append(v2)
            
            # -1, 1
            v3 = np.zeros(D, dtype=np.int64)
            v3[i] = -1
            v3[j] = 1
            vectors.append(v3)
            
            # -1, -1
            v4 = np.zeros(D, dtype=np.int64)
            v4[i] = -1
            v4[j] = -1
            vectors.append(v4)
            
    return np.array(vectors)
