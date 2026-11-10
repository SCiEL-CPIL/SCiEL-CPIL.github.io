import numpy as np
import time
from itertools import combinations

def kissing_number11() -> np.ndarray:
       start_time = time.time()
       deadline = start_time + 580
       
       d = 11
       
       # Type 1 vectors
       type1 = []
       for i in range(d):
           for sign in [1, -1]:
               v = [0] * d
               v[i] = 2 * sign
               type1.append(v)
       type1 = np.array(type1, dtype=np.int64)
       
       # Type 2 vectors
       type2_list = []
       for idx in combinations(range(d), 4):
           for s1 in [1, -1]:
               for s2 in [1, -1]:
                   for s3 in [1, -1]:
                       for s4 in [1, -1]:
                           v = [0] * d
                           v[idx[0]] = s1
                           v[idx[1]] = s2
                           v[idx[2]] = s3
                           v[idx[3]] = s4
                           type2_list.append(v)
       
       type2 = np.array(type2_list, dtype=np.int64)
       n_type2 = len(type2)
       
       # Precompute cliques for each type2 vector
       # vector i corresponds to support A_i and signs s_i
       # cliques[i] is a list of 4 tuple keys
       vector_cliques = []
       for v in type2:
           nz_indices = [i for i in range(d) if v[i] != 0]
           signs = [v[i] for i in nz_indices]
           cliques = []
           # Generate 4 triplets from the 4 indices
           # Indices: 0,1,2,3
           triplets = [
               (0,1,2), (0,1,3), (0,2,3), (1,2,3)
           ]
           for t in triplets:
               # Get actual coordinate indices
               coords = tuple(sorted(nz_indices[x] for x in t))
               # Get signs for these coords
               s_t = tuple(signs[x] for x in t)
               cliques.append((coords, s_t))
           vector_cliques.append(cliques)
           
       best_selected_type2 = []
       rng = np.random.RandomState(42)
       
       for restart in range(1000):
           if time.time() > deadline - 10:
               break
           
           order = rng.permutation(n_type2)
           blocked = set()
           selected = []
           
           for idx in order:
               cliques = vector_cliques[idx]
               # Check if any clique is blocked
               is_valid = True
               for c in cliques:
                   if c in blocked:
                       is_valid = False
                       break
               
               if is_valid:
                   selected.append(idx)
                   for c in cliques:
                       blocked.add(c)
           
           if len(selected) > len(best_selected_type2):
               best_selected_type2 = selected[:]
       
       # Combine
       result = np.vstack([type1, type2[best_selected_type2]])
       
       # Verify
       # ... (standard verification)
       
       return np.around(result).astype(np.int64)
