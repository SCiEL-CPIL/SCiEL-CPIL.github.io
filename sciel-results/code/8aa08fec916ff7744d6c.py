import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

def construct():
       n, m, p = DIMENSIONS
       dim_u = n * m
       dim_v = m * p
       dim_w = p * n
       
       T = np.zeros((dim_u, dim_v, dim_w))
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   a = i * m + j
                   b = j * p + k
                   c = k * n + i
                   T[a, b, c] = 1.0

       start_time = time.time()
       deadline = 145.0 # 5s margin

       # Grid of allowed values
       grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
       
       def error_calc(U, V, W):
           T_calc = np.einsum('ar,br,cr->abc', U, V, W)
           return np.sum((T_calc - T)**2)

       def check_exact(U, V, W):
           # Check if U, V, W are in the grid (or close to it)
           # And if the product is exactly T
           # Since we search on the grid, they are in the grid.
           # We just need to check the product.
           # To avoid floating point issues, we can multiply by 2 and check integers.
           U_int = np.round(U * 2)
           V_int = np.round(V * 2)
           W_int = np.round(W * 2)
           T_calc = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
           return np.all(T_calc == 8 * T)

       def random_init(r):
           U = np.random.choice(grid, size=(dim_u, r))
           V = np.random.choice(grid, size=(dim_v, r))
           W = np.random.choice(grid, size=(dim_w, r))
           # Gauge fixing: Fix U[:, 0] to be [1, 0, ..., 0]
           U[:, 0] = 0
           U[0, 0] = 1
           return U, V, W

       def local_search(r, max_iters=10000):
           U, V, W = random_init(r)
           current_err = error_calc(U, V, W)
           best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
           best_err = current_err
           
           # Simulated Annealing parameters
           T_start = 1.0
           T_end = 0.01
           alpha = 0.999
           T = T_start
           
           for it in range(max_iters):
               if time.time() > deadline - 5:
                   break
               
               # Propose a move: pick a random variable and change it
               # We can pick from U, V, or W
               mat_idx = np.random.randint(3)
               if mat_idx == 0:
                   mat = U
                   name = 'U'
               elif mat_idx == 1:
                   mat = V
                   name = 'V'
               else:
                   mat = W
                   name = 'W'
                   
               # Don't change the fixed gauge variable
               if name == 'U' and np.random.randint(r) == 0:
                   # Pick another column
                   col = np.random.randint(1, r)
               else:
                   col = np.random.randint(r)
                   
               row = np.random.randint(mat.shape[0])
               
               # Choose a new value from the grid, different from current
               old_val = mat[row, col]
               new_val_idx = np.random.randint(len(grid))
               new_val = grid[new_val_idx]
               if new_val == old_val:
                   continue
                   
               # Apply move
               mat[row, col] = new_val
               new_err = error_calc(U, V, W)
               
               delta = new_err - current_err
               if delta < 0 or np.random.random() < np.exp(-delta / T):
                   current_err = new_err
                   if current_err < best_err:
                       best_err = current_err
                       best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                       if best_err < 1e-9:
                           break
               else:
                   # Revert
                   mat[row, col] = old_val
                   
               T *= alpha
               
           if check_exact(best_U, best_V, best_W):
               return best_U, best_V, best_W
           return None

       # Try ranks 15, 16, 17, 18
       for r in [15, 16, 17, 18]:
           if time.time() > deadline - 10:
               break
           
           # Try a few restarts
           for restart in range(5):
               if time.time() > deadline - 5:
                   break
               res = local_search(r, max_iters=5000)
               if res:
                   U, V, W = res
                   return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

       # Fallback to schoolbook rank 18
       rank_fb = n * m * p
       U_fb = np.zeros((dim_u, rank_fb))
       V_fb = np.zeros((dim_v, rank_fb))
       W_fb = np.zeros((dim_w, rank_fb))
       col = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   U_fb[i * m + j, col] = 1
                   V_fb[j * p + k, col] = 1
                   W_fb[k * n + i, col] = 1
                   col += 1
                   
       return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
