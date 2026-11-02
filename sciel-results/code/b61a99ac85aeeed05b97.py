import random
import numpy as np
from fractions import Fraction
import time

def construct():
       n, m, p = 2, 3, 7
       rank_school = n * m * p
       
       # Precompute valid indices
       valid_indices = []
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   a = i * m + j
                   b = j * p + k
                   c = k * n + i
                   valid_indices.append((a, b, c))
       
       def verify_float(U, V, W, rank):
           U = np.array(U, dtype=float)
           V = np.array(V, dtype=float)
           W = np.array(W, dtype=float)
           # Efficient check:
           # For each r, we have a rank-1 tensor.
           # Sum them up.
           # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
           # We can compute this as a series of outer products.
           # But n*m=6, m*p=21, p*n=14.
           # Total size 6*21*14 = 1764.
           # Rank ~40.
           # 40 * 1764 operations. Very fast.
           
           T = np.zeros((n*m, m*p, p*n), dtype=float)
           for r in range(rank):
               u = U[:, r]
               v = V[:, r]
               w = W[:, r]
               # Outer product u[:, None, None] * v[None, :, None] * w[None, None, :]
               T += np.einsum('a,b,c->abc', u, v, w)
               
           # Check constraints
           # 1. Valid indices must be 1
           for (a, b, c) in valid_indices:
               if abs(T[a, b, c] - 1.0) > 1e-6:
                   return False
           
           # 2. Invalid indices must be 0
           # Instead of checking all 1764, we can check the sum of squares or similar?
           # No, we need exact 0.
           # But float check: if any |T| > 1e-6 at invalid spots, fail.
           # We can iterate all a,b,c
           for a in range(n*m):
               for b in range(m*p):
                   for c in range(p*n):
                       if (a, b, c) not in valid_indices: # This is slow lookup
                           pass
           
           # Better: Create a mask
           mask = np.zeros((n*m, m*p, p*n), dtype=bool)
           for (a, b, c) in valid_indices:
               mask[a, b, c] = True
               
           # Check invalid spots
           invalid_vals = T[~mask]
           if np.any(np.abs(invalid_vals) > 1e-6):
               return False
               
           return True

       def verify_exact(U, V, W, rank):
           # Convert to Fraction
           Uf = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
           Vf = [[Fraction(V[i][r]) for r in range(rank)] for i in range(m*p)]
           Wf = [[Fraction(W[i][r]) for r in range(rank)] for i in range(p*n)]
           
           for (a, b, c) in valid_indices:
               s = Fraction(0)
               for r in range(rank):
                   s += Uf[a][r] * Vf[b][r] * Wf[c][r]
               if s != 1:
                   return False
           
           # Check invalids
           # To save time, only check a random subset or all if rank is low?
           # With rank 40, checking all 1764 spots with Fractions is ~1764*40 = 70k ops.
           # Fraction ops are slow, but 70k is manageable in 150s?
           # Maybe.
           for a in range(n*m):
               for b in range(m*p):
                   for c in range(p*n):
                       if (a, b, c) in valid_indices:
                           continue
                       s = Fraction(0)
                       for r in range(rank):
                           s += Uf[a][r] * Vf[b][r] * Wf[c][r]
                       if s != 0:
                           return False
           return True

       # Search Strategy
       # Try Rank 41, 40, 39...
       # Use Random Search with Float Filter
        
       entries = [-2, -1, -0.5, 0, 0.5, 1, 2]
       
       best_rank = rank_school
       best_U = None
       best_V = None
       best_W = None
       
       # Initialize Schoolbook
       U_sb = [[0]*rank_school for _ in range(n*m)]
       V_sb = [[0]*rank_school for _ in range(m*p)]
       W_sb = [[0]*rank_school for _ in range(p*n)]
       col = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   a = i*m+j
                   b = j*p+k
                   c = k*n+i
                   U_sb[a][col] = 1
                   V_sb[b][col] = 1
                   W_sb[c][col] = 1
                   col += 1
                   
       best_U, best_V, best_W = U_sb, V_sb, W_sb
       
       start_time = time.time()
       deadline = start_time + 140.0
       
       for target_rank in range(rank_school - 1, 29, -1):
           if time.time() > deadline - 10.0:
               break
           
           found = False
           # Number of iterations depends on time
           # Let's do 1000 float-check iterations per rank
           for _ in range(1000):
               if time.time() > deadline - 5.0:
                   break
               
               # Generate random U, V, W
               # Bias towards small integers
               U = [[random.choice(entries) for _ in range(target_rank)] for _ in range(n*m)]
               V = [[random.choice(entries) for _ in range(target_rank)] for _ in range(m*p)]
               W = [[random.choice(entries) for _ in range(target_rank)] for _ in range(p*n)]
               
               # Fast Float Check
               if verify_float(U, V, W, target_rank):
                   # Exact Check
                   if verify_exact(U, V, W, target_rank):
                       best_rank = target_rank
                       best_U, best_V, best_W = U, V, W
                       found = True
                       break
           
           if found:
               # If we found a lower rank, we can continue searching for even lower
               # But usually, finding one is the big jump.
               # We can try to go lower.
               continue
           else:
               # If we didn't find target_rank, maybe target_rank-1 is easier?
               # No, lower rank is harder.
               # So if we fail 41, we likely fail 40.
               # We should break.
               break

       # Return best
       return {
           "U": best_U,
           "V": best_V,
           "W": best_W
       }
