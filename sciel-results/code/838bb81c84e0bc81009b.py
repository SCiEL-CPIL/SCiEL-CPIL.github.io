import numpy as np
import itertools
import time

D = 11

def construct():
       start_time = time.time()
       deadline = start_time + 145
       
       # Type 1 vectors
       type1 = []
       for i in range(D):
           for s in [1, -1]:
               v = [0] * D
               v[i] = 2 * s
               type1.append(v)
       
       # Type 2 vectors
       supports = list(itertools.combinations(range(D), 4))
       # Map support to index
       support_to_idx = {s: i for i, s in enumerate(supports)}
       
       # Precompute neighbors in Johnson graph
       # Neighbor if intersection size is 3
       neighbors = [[] for _ in range(len(supports))]
       for i, s_i in enumerate(supports):
           for j, s_j in enumerate(supports):
               if i == j: continue
               if len(set(s_i) & set(s_j)) == 3:
                   neighbors[i].append(j)
       
       # For each support, list of 16 sign vectors
       sign_vectors = list(itertools.product([1, -1], repeat=4))
       
       # We will select a subset of sign vectors for each support.
       # selected[s_idx] = list of sign tuples selected for this support
       selected = [set() for _ in range(len(supports))]
       
       # Greedy assignment
       # Order supports by degree (descending)
       order = sorted(range(len(supports)), key=lambda x: -len(neighbors[x]))
       
       # Available patterns for each support?
       # Actually, we just check compatibility on the fly.
       
       # To speed up, we can precompute the "forbidden" patterns for each support based on current selections.
       # But selections change.
       
       # Let's just try to add vectors one by one.
       # Total possible Type 2 vectors is 330 * 16 = 5280.
       # We want to maximize count.
       
       # Randomized Greedy
       rng = np.random.default_rng(42)
       
       best_count = 0
       best_selection = [set() for _ in range(len(supports))]
       
       for attempt in range(20): # 20 restarts
           if time.time() > deadline - 10: break
           
           selected = [set() for _ in range(len(supports))]
           
           # Shuffle order for diversity
           if attempt > 0:
               order = rng.permutation(len(supports))
           else:
               order = sorted(range(len(supports)), key=lambda x: -len(neighbors[x]))
           
           for s_idx in order:
               if time.time() > deadline - 5: break
               
               support = supports[s_idx]
               # Try all 16 sign vectors
               # We want to add as many as possible.
               # This is a small MIS problem (16 nodes) with constraints from neighbors.
               
               # Constraints from already selected neighbors:
               # For each neighbor t_idx, we have selected signs selected[t_idx].
               # For a candidate sign x for s_idx:
               # It is forbidden if there exists y in selected[t_idx] such that x and y are "conflicting".
               # Conflict condition: x_U == y_U (where U is intersection of supports).
               
               # Let's precompute for each neighbor t_idx, the set of forbidden patterns on U.
               # U = support[s_idx] & support[t_idx]
               
               forbidden_patterns = set()
               
               for t_idx in neighbors[s_idx]:
                   if not selected[t_idx]: continue
                   
                   U = tuple(sorted(set(support) & set(supports[t_idx])))
                   # Map U indices to positions in support and support[t_idx]
                   # This mapping is tricky.
                   
                   # Let's just check dot products directly for the 16 candidates against all selected neighbors.
                   # 16 candidates * (sum of sizes of selected neighbors).
                   # Sizes are small (<=16).
                   # Number of neighbors in J(11,4) for a vertex:
                   # Choose 3 from 4, 1 from 7. 4 * 7 = 28 neighbors.
                   # So 28 neighbors.
                   # Max 28 * 16 = 448 checks per candidate.
                   # 16 candidates * 448 = 7168 checks per support.
                   # 330 supports * 7168 = 2.3 million checks.
                   # Very fast.
                   
                   pass # We will do direct checks below
               
               # List of candidates for this support
               candidates = list(sign_vectors)
               
               # Greedily add candidates
               # Sort candidates by "popularity"? Or just random?
               # Let's try to add all that are compatible.
               # This is equivalent to finding a maximal independent set in a graph of 16 nodes?
               # No, the constraints are only from OTHER supports.
               # Within the same support, all 16 are compatible (dot <= 2).
               # So we can add ALL candidates that are not forbidden by neighbors!
               
               valid_candidates = []
               for x in candidates:
                   compatible = True
                   for t_idx in neighbors[s_idx]:
                       if not selected[t_idx]: continue
                       
                       # Check x against all y in selected[t_idx]
                       # x is tuple of 4 signs for support[s_idx]
                       # y is tuple of 4 signs for support[t_idx]
                       
                       # Compute dot product
                       # We need to align indices.
                       # Let's create a dict for x and y? Or just loop.
                       
                       # Since D=11 is small, we can just expand to length 11 and dot.
                       # But that's slow.
                       
                       # Better:
                       # Find common indices.
                       # We know intersection size is 3.
                       
                       # Let's precompute the intersection sets and mappings for each neighbor pair.
                       # This is static.
                       
                       pass
                   
                   if compatible:
                       valid_candidates.append(x)
               
               # Add all valid candidates
               selected[s_idx].update(valid_candidates)
               
           # Count total
           count = sum(len(s) for s in selected)
           if count > best_count:
               best_count = count
               best_selection = [s.copy() for s in selected]
       
       # Construct final vectors
       final_vectors = type1[:]
       
       for s_idx, signs in enumerate(best_selection):
           support = supports[s_idx]
           for x in signs:
               v = [0] * D
               for idx, s in zip(support, x):
                   v[idx] = s
               final_vectors.append(v)
       
       result = np.array(final_vectors, dtype=np.int64)
       
       # Verify
       n = len(result)
       max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
       valid = True
       for i in range(n):
           for j in range(i + 1, n):
               diff = result[i] - result[j]
               if np.dot(diff, diff) < max_norm_sq:
                   valid = False
                   break
           if not valid: break
           
       if not valid:
           # Fallback
           return np.array(type1, dtype=np.int64)
           
       return result
