      def construct():
          n, m, p = 2, 4, 6
          rank = 42
          U = [[0] * rank for _ in range(n * m)]
          V = [[0] * rank for _ in range(m * p)]
          W = [[0] * rank for _ in range(p * n)]
          
          # Strassen coefficients
          # u_s: [a, b, c, d]
          u_coeffs = [
              [1, 0, 0, 1], # M0
              [0, 0, 1, 1], # M1
              [1, 0, 0, 0], # M2
              [0, 0, 0, 1], # M3
              [1, 1, 0, 0], # M4
              [-1, 0, 1, 0],# M5
              [0, 1, 0, -1] # M6
          ]
          
          # v_s: [e, f, g, h]
          v_coeffs = [
              [1, 0, 0, 1], # M0
              [1, 0, 0, 0], # M1
              [0, 1, 0, -1],# M2
              [0, -1, 1, 0],# M3
              [0, 0, 0, 1], # M4
              [1, 1, 0, 0], # M5
              [0, 0, 1, 1]  # M6
          ]
          
          # w_s: [C00, C01, C10, C11]
          w_coeffs = [
              [1, 0, 0, 1], # M0
              [0, 0, 1, -1],# M1
              [0, 1, 0, 1], # M2
              [1, 0, 1, 0], # M3
              [-1, 1, 0, 0],# M4
              [0, 0, 0, 1], # M5
              [1, 0, 0, 0]  # M6
          ]
          
          for g in range(3):
              for part in range(2):
                  for s in range(7):
                      r = (g * 2 + part) * 7 + s
                      
                      # Fill U
                      for i in range(2):
                          for jp in range(2):
                              val = u_coeffs[s][i * 2 + jp]
                              if val != 0:
                                  U[i * 4 + part * 2 + jp][r] = val
                              
                      # Fill V
                      for jp in range(2):
                          for kp in range(2):
                              val = v_coeffs[s][jp * 2 + kp]
                              if val != 0:
                                  V[(part * 2 + jp) * 6 + g * 2 + kp][r] = val
                                  
                      # Fill W
                      for kp in range(2):
                          for i in range(2):
                              val = w_coeffs[s][kp * 2 + i]
                              if val != 0:
                                  W[(g * 2 + kp) * 2 + i][r] = val
                                  
          return {"U": U, "V": V, "W": W}
      