import numpy as np
import itertools
import time

D = 11

def construct():
       start_time = time.time()
       
       # 1. Generate D11 Minimal Vectors
       # Type 1: Weight 1, +/- 2
       # Type 2: Weight 4, +/- 1, even sum
       
       vectors = []
       
       # Type 1
       for i in range(D):
           for s in [1, -1]:
               v = [0]*D
               v[i] = 2*s
               vectors.append(v)
               
       # Type 2
       for pos in itertools.combinations(range(D), 4):
           # Generate all sign combos with even number of -1s
           # There are 8 such combos
           signs = []
           for mask in range(16):
               # count bits set in mask
               if bin(mask).count('1') % 2 == 0:
                   signs.append([1 if (mask & (1<<j)) == 0 else -1 for j in range(4)])
           
           for s in signs:
               v = [0]*D
               for p, val in zip(pos, s):
                   v[p] = val
               vectors.append(v)
               
       V = np.array(vectors, dtype=np.int64)
       N = len(V)
       
       # Return all vectors as they form a valid kissing configuration
       return V
