import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
      n, m, p = DIMENSIONS
      start_time = time.time()
      deadline = start_time + 140
      
      # Schoolbook solution
      rank_sb = n * m * p
      U_sb = [[0] * rank_sb for _ in range(n * m)]
      V_sb = [[0] * rank_sb for _ in range(m * p)]
      W_sb = [[0] * rank_sb for _ in range(p * n)]
      col = 0
      for i in range(n):
          for j in range(m):
              for k in range(p):
                  U_sb[i * m + j][col] = 1
                  V_sb[j * p + k][col] = 1
                  W_sb[k * n + i][col] = 1
                  col += 1
          
      best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
      best_rank = rank_sb
          
      # Search for lower rank
      # Try ranks from 80 to 95
      for R in range(80, best_rank):
          if time.time() > deadline - 10:
              break
          
          found = False
          # Increase trials for lower ranks
          trials = 500 if R < 90 else 100
          for trial in range(trials):
              if time.time() > deadline - 10:
                  break
              
              # Random U, V with entries in {-1, 0, 1}
              U_flat = np.random.randint(-1, 2, (n * m, R))
              V_flat = np.random.randint(-1, 2, (m * p, R))
              
              # Check if solution exists
              valid = True
              W_flat = np.zeros((p * n, R))
              
              for i in range(n):
                  for k in range(p):
                      M_ik = np.zeros((m * m, R))
                      for j in range(m):
                          for l in range(m):
                              for r in range(R):
                                  M_ik[j * m + l, r] = U_flat[i * m + j, r] * V_flat[l * p + k, r]
                      
                      b_ik = np.zeros(m * m)
                      for j in range(m):
                          b_ik[j * m + j] = 1
                      
                      # Solve M_ik w = b_ik
                      w, residuals, rank, s = np.linalg.lstsq(M_ik, b_ik, rcond=None)
                      
                      if len(residuals) == 0 or residuals[0] > 1e-9:
                          valid = False
                          break
                      
                      W_flat[k * n + i, :] = w
                  
                  if not valid:
                      break
              
              if valid:
                  # Snap to half integers
                  W_snapped = np.round(W_flat * 2) / 2
                  
                  # Verify
                  valid_snapped = True
                  for i in range(n):
                      for k in range(p):
                          for j in range(m):
                              for l in range(m):
                                  val = 0
                                  for r in range(R):
                                      val += W_snapped[k * n + i, r] * U_flat[i * m + j, r] * V_flat[l * p + k, r]
                                  if abs(val - (1 if j == l else 0)) > 1e-9:
                                      valid_snapped = False
                                      break
                              if not valid_snapped: break
                          if not valid_snapped: break
                      if not valid_snapped: break
                  
                  if valid_snapped:
                      U_list = U_flat.tolist()
                      V_list = V_flat.tolist()
                      W_list = W_snapped.tolist()
                      
                      def convert(mat):
                          return [[float(v) for v in row] for row in mat]
                      
                      best_solution = {"U": convert(U_list), "V": convert(V_list), "W": convert(W_list)}
                      best_rank = R
                      found = True
                      break
          
          if found:
              break
              
      return best_solution
