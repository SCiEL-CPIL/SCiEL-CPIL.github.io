   def construct():
       n, m, p = 2, 5, 5
       R = 34
       U = [[0]*R for _ in range(n*m)]
       V = [[0]*R for _ in range(m*p)]
       W = [[0]*R for _ in range(p*n)]
       
       col = 0
       
       def add_term(u, v, w, val=1):
           nonlocal col
           U[u][col] += val
           V[v][col] += val
           W[w][col] += val
           col += 1
           
       # Helper for 2x2x2 block
       def add_222(i_off, j_off, k_off):
           # i in {0,1}, j in {0,1}, k in {0,1}
           for i in range(2):
               for j in range(2):
                   for k in range(2):
                       u_idx = i * 5 + (j + j_off)
                       v_idx = (j + j_off) * 5 + (k + k_off)
                       w_idx = (k + k_off) * 2 + i
                       add_term(u_idx, v_idx, w_idx)
                       
       # Helper for 2x2x1 block
       def add_221(i_off, j_off, k_off):
           for i in range(2):
               for j in range(2):
                   u_idx = i * 5 + (j + j_off)
                   v_idx = (j + j_off) * 5 + k_off
                   w_idx = k_off * 2 + i
                   add_term(u_idx, v_idx, w_idx)
                   
       # Helper for 2x1x5 block
       def add_215(i_off, j_off, k_off):
           for i in range(2):
               for k in range(5):
                   u_idx = i * 5 + (j + j_off)
                   v_idx = (j + j_off) * 5 + (k + k_off)
                   w_idx = (k + k_off) * 2 + i
                   add_term(u_idx, v_idx, w_idx)
                   
       # Block 1: j=0,1
       add_222(0, 0, 0)
       add_222(0, 0, 2)
       add_221(0, 0, 4)
       
       # Block 2: j=2,3
       add_222(0, 2, 0)
       add_222(0, 2, 2)
       add_221(0, 2, 4)
       
       # Block 3: j=4
       add_215(0, 4, 0)
       
       return {"U": U, "V": V, "W": W}
   