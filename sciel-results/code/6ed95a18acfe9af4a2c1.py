   def construct():
       # Hardcoded rank-23 decomposition for 3x3x3
       U3 = [...]
       V3 = [...]
       W3 = [...]
       
       # Construct 3x3x6
       n, m, p = 3, 3, 6
       rank = 46
       U = [[0]*rank for _ in range(n*m)]
       V = [[0]*rank for _ in range(m*p)]
       W = [[0]*rank for _ in range(p*n)]
       
       # Fill first half
       for i in range(9):
           for r in range(23):
               U[i][r] = U3[i][r]
       
       # Fill second half (U is same)
       for i in range(9):
           for r in range(23):
               U[i][r+23] = U3[i][r]
               
       # Fill V
       # V3 indices: j*3+k (0..8)
       # V indices: j*6+k (0..17)
       for j in range(3):
           for k in range(3):
               idx_v3 = j*3 + k
               idx_v1 = j*6 + k
               idx_v2 = j*6 + (k+3)
               for r in range(23):
                   V[idx_v1][r] = V3[idx_v3][r]
                   V[idx_v2][r+23] = V3[idx_v3][r]
                   
       # Fill W
       # W3 indices: k*3+i (0..8)
       # W indices: k*3+i (0..17)
       for k in range(3):
           for i in range(3):
               idx_w3 = k*3 + i
               idx_w1 = k*3 + i
               idx_w2 = (k+3)*3 + i
               for r in range(23):
                   W[idx_w1][r] = W3[idx_w3][r]
                   W[idx_w2][r+23] = W3[idx_w3][r]
                   
       return {"U": U, "V": V, "W": W}
   