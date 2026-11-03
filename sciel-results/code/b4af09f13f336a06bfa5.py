   try:
       sol, _, _, _ = linalg.lstsq(A_mat, b_vec)
       U[i * m + j, :] = sol
   except:
       pass
   