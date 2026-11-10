import numpy as np

def kissing_number11() -> np.ndarray:
       d = 11
       vectors = []
       
       # Type 1: Integer E8 minimal vectors scaled by 2
       # Original: two 1s (or -1s), rest 0.
       # Scaled: two 2s (or -2s), rest 0.
       for i in range(8):
           for j in range(i + 1, 8):
               for s1 in [1, -1]:
                   for s2 in [1, -1]:
                       v = np.zeros(8, dtype=np.int64)
                       v[i] = 2 * s1
                       v[j] = 2 * s2
                       vectors.append(v)
       
       # Type 2: Half-integer E8 minimal vectors scaled by 2
       # Original: eight +/- 0.5, even number of +1s.
       # Scaled: eight +/- 1, even number of +1s.
       # Iterate over all 2^8 sign combinations
       for mask in range(256):
           # Check if number of set bits (or +1s) is even
           # Let's define +1 as bit 0? No, just count.
           # If we use 1 for +1 and -1 for -1.
           # Let's generate signs.
           signs = []
           count_plus = 0
           for k in range(8):
               if (mask >> k) & 1:
                   signs.append(1)
                   count_plus += 1
               else:
                   signs.append(-1)
           
           if count_plus % 2 == 0:
               v = np.array(signs, dtype=np.int64)
               vectors.append(v)
       
       # Pad to 11 dimensions
       points = np.zeros((len(vectors), 11), dtype=np.int64)
       for idx, v in enumerate(vectors):
           points[idx, :8] = v
       
       return points
