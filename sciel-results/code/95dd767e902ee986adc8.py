import numpy as np

def construct():
       N = 17
       MIN_DIST = 6
       TOTAL = 2**N
       
       # Precompute popcount
       pc = np.array([bin(i).count('1') for i in range(TOTAL)], dtype=np.uint8)
       
       best_code = []
       best_size = 0
       
       # Run multiple trials
       for seed in range(20):
           np.random.seed(seed)
           cands = np.random.permutation(TOTAL)
           
           # Buffer for selected codewords
           buf = np.zeros(TOTAL, dtype=np.int32)
           count = 0
           
           for cand in cands:
               cand = int(cand)
               if count == 0:
                   buf[0] = cand
                   count = 1
                   continue
               
               # Vectorized distance check
               # Note: buf[:count] is a view
               diffs = cand ^ buf[:count]
               dists = pc[diffs]
               if np.min(dists) >= MIN_DIST:
                   buf[count] = cand
                   count += 1
           
           if count > best_size:
               best_size = count
               best_code = buf[:count].tolist()
       
       return best_code
