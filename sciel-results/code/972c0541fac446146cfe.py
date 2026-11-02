import numpy as np
import time

def construct():
          n, m, p = 2, 3, 5
          # Try ranks 29, 28
          for R in [29, 28]:
              deadline = time.time() + 120
              attempts = 0
              while time.time() < deadline - 5 and attempts < 1000:
                  attempts += 1
                  U = np.random.randint(-1, 2, (n*m, R))
                  V = np.random.randint(-1, 2, (m*p, R))
                  
                  valid = True
                  W_sol = np.zeros((p*n, R))
                  
                  for i in range(n):
                      for k in range(p):
                          # Construct M (m x R)
                          # U rows: i*m ... i*m+m-1
                          # V cols: k ... k+p-1 ? No.
                          # V index is j*p + k.
                          # For fixed k, j varies.
                          # So V[j*p+k] for j=0..m-1.
                          
                          M = np.zeros((m, R))
                          for j in range(m):
                              u_row = U[i*m + j]
                              v_row = V[j*p + k] # Wait, V is (m*p, R). Index j*p+k.
                              M[j] = u_row * v_row
                          
                          # Solve M w = 1
                          try:
                              # Use least squares
                              w_res, residuals, rank, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                              
                              # Check residual
                              if np.max(np.abs(M @ w_res - np.ones(m))) > 1e-9:
                                  valid = False
                                  break
                              
                              # Try to snap to half-integers
                              w_half = np.round(w_res * 2) / 2.0
                              
                              # Verify exactness
                              if np.max(np.abs(M @ w_half - np.ones(m))) < 1e-9:
                                  W_sol[k*n + i] = w_half
                              else:
                                  valid = False
                                  break
                          
                          except:
                              valid = False
                              break
                          
                  if valid:
                      # Return solution
                      return {"U": U.tolist(), "V": V.tolist(), "W": W_sol.tolist()}
          
          # Fallback to schoolbook
          rank = n*m*p
          U = [[0]*rank for _ in range(n*m)]
          V = [[0]*rank for _ in range(m*p)]
          W = [[0]*rank for _ in range(p*n)]
          col = 0
          for i in range(n):
              for j in range(m):
                  for k in range(p):
                      U[i*m+j][col] = 1
                      V[j*p+k][col] = 1
                      W[k*n+i][col] = 1
                      col += 1
          return {"U": U, "V": V, "W": W}
