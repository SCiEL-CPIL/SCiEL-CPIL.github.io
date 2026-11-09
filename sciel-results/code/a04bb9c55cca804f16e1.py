import numpy as np
import itertools
import time

def kissing_number11() -> np.ndarray:
       d = 11
       start_time = time.time()
       deadline = start_time + 590 # 10s margin
       
       best_points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
       
       # Try R=1
       # Candidates: +/- e_i
       cands_1 = []
       for i in range(d):
           v1 = np.zeros(d, dtype=int); v1[i] = 1
           v2 = np.zeros(d, dtype=int); v2[i] = -1
           cands_1.extend([v1, v2])
       cands_1 = np.array(cands_1)
       # All pairs have dist^2 >= 2. R=1. 2 > 1. Valid.
       # Size 22.
       if len(cands_1) > len(best_points):
           best_points = cands_1
           
       # Try R=2
       if time.time() < deadline:
           cands_2 = []
           for pos in itertools.combinations(range(d), 2):
               for s1 in [-1, 1]:
                   for s2 in [-1, 1]:
                       v = np.zeros(d, dtype=int)
                       v[pos[0]] = s1
                       v[pos[1]] = s2
                       cands_2.append(v)
           cands_2 = np.array(cands_2)
           
           # Greedy MIS
           n = len(cands_2)
           # Precompute conflicts
           # Conflict if dot >= 1
           conflict = np.zeros((n, n), dtype=bool)
           for i in range(n):
               for j in range(i+1, n):
                   if np.dot(cands_2[i], cands_2[j]) >= 1:
                       conflict[i, j] = True
                       conflict[j, i] = True
                       
           best_set_2 = []
           while time.time() < deadline:
               avail = np.ones(n, dtype=bool)
               curr = []
               order = np.random.permutation(n)
               for i in order:
                   if not avail[i]: continue
                   curr.append(i)
                   avail[conflict[i]] = False
                   avail[i] = False
               if len(curr) > len(best_set_2):
                   best_set_2 = curr
                   
           if len(best_set_2) > len(best_points):
               best_points = cands_2[best_set_2]
               
       # Try R=3
       if time.time() < deadline:
           # Vectors with three +/- 1
           cands_3 = []
           for pos in itertools.combinations(range(d), 3):
               for s1 in [-1, 1]:
                   for s2 in [-1, 1]:
                       for s3 in [-1, 1]:
                           v = np.zeros(d, dtype=int)
                           v[pos[0]] = s1
                           v[pos[1]] = s2
                           v[pos[2]] = s3
                           cands_3.append(v)
           cands_3 = np.array(cands_3)
           n = len(cands_3)
           # Conflict if dot >= 2 (since R=3, dot < 1.5 => dot <= 1 ok, dot >= 2 conflict)
           # Wait: R=3. Condition dot < 1.5. Integers: dot <= 1.
           # Conflict if dot >= 2.
           conflict = np.zeros((n, n), dtype=bool)
           for i in range(n):
               for j in range(i+1, n):
                   if np.dot(cands_3[i], cands_3[j]) >= 2:
                       conflict[i, j] = True
                       conflict[j, i] = True
                       
           best_set_3 = []
           while time.time() < deadline:
               avail = np.ones(n, dtype=bool)
               curr = []
               order = np.random.permutation(n)
               for i in order:
                   if not avail[i]: continue
                   curr.append(i)
                   avail[conflict[i]] = False
                   avail[i] = False
               if len(curr) > len(best_set_3):
                   best_set_3 = curr
                   
           if len(best_set_3) > len(best_points):
               best_points = cands_3[best_set_3]
               
       # Validation
       # Ensure shape (k, 11)
       best_points = np.array(best_points, dtype=np.int64)
       
       return best_points
