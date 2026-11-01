import scipy
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
      n, m, p = DIMENSIONS
      target_rank = 23
      start_time = time.time()
      deadline = start_time + 100 # Safety margin
      
      # Target tensor T[i, j, k] = 1
      T = np.ones((n, m, p))
      
      best_rank = n * m * p
      best_solution = None
      
      # Fallback: Schoolbook
      U_school = np.zeros((n * m, best_rank))
      V_school = np.zeros((m * p, best_rank))
      W_school = np.zeros((p * n, best_rank))
      col = 0
      for i in range(n):
          for j in range(m):
              for k in range(p):
                  U_school[i * m + j, col] = 1
                  V_school[j * p + k, col] = 1
                  W_school[k * n + i, col] = 1
                  col += 1
      best_solution = {'U': U_school, 'V': V_school, 'W': W_school}
      
      # Try to find rank 23
      for attempt in range(50):
          if time.time() > deadline:
              break
          
          U = np.random.randn(n * m, target_rank)
          V = np.random.randn(m * p, target_rank)
          W = np.random.randn(p * n, target_rank)
          
          # Define objective
          def objective(params):
              U_flat = params[:n * m * target_rank].reshape(n * m, target_rank)
              V_flat = params[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
              W_flat = params[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
              
              # Compute tensor
              # T_calc[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
              # Note: W index is k, i.
              # U is (i, j, r)
              # V is (j, k, r)
              # W is (k, i, r)
              
              # Reshape for einsum
              # U: (i, j, r)
              # V: (j, k, r)
              # W: (k, i, r)
              
              # Sum over r
              # T_calc[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
              
              # Use einsum
              # 'ijr,jkr,ki r->ijk'
              # Note: W is stored as (k, i, r) in the flattened (p*n, rank) matrix?
              # Flattened index for W is k*n + i.
              # So W_matrix[k*n + i, r] corresponds to W[k, i, r].
              
              T_calc = np.einsum('ijr,jkr,kr->ijk', U_flat, V_flat, W_flat) # Wait, W_flat is (k*n+i, r).
              # Need to reshape W_flat to (k, i, r).
              W_reshaped = W_flat.reshape(p, n, target_rank)
              # W_reshaped[k, i, r]
              
              # V_flat is (j*p+k, r). Reshape to (j, k, r).
              V_reshaped = V_flat.reshape(m, p, target_rank)
              
              # U_flat is (i*m+j, r). Reshape to (i, j, r).
              U_reshaped = U_flat.reshape(n, m, target_rank)
              
              T_calc = np.einsum('ijr,jkr,ki r->ijk', U_reshaped, V_reshaped, W_reshaped)
              
              error = np.sum((T - T_calc) ** 2)
              return error
          
          params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
          
          # Optimize
          try:
              res = scipy.optimize.minimize(objective, params, method='L-BFGS-B', options={'maxiter': 50})
              if res.fun < 1e-6:
                  # Verify
                  U_opt = res.x[:n * m * target_rank].reshape(n * m, target_rank)
                  V_opt = res.x[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
                  W_opt = res.x[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
                  
                  # Round to half-integers
                  U_round = np.round(U_opt * 2) / 2
                  V_round = np.round(V_opt * 2) / 2
                  W_round = np.round(W_opt * 2) / 2
                  
                  # Verify exactly
                  U_r = U_round.reshape(n, m, target_rank)
                  V_r = V_round.reshape(m, p, target_rank)
                  W_r = W_round.reshape(p, n, target_rank)
                  
                  T_check = np.einsum('ijr,jkr,ki r->ijk', U_r, V_r, W_r)
                  
                  if np.allclose(T_check, T, atol=1e-9):
                      best_rank = target_rank
                      best_solution = {'U': U_round, 'V': V_round, 'W': W_round}
                      break
          except Exception:
              pass
              
      return {
          'U': best_solution['U'].tolist(),
          'V': best_solution['V'].tolist(),
          'W': best_solution['W'].tolist()
      }
