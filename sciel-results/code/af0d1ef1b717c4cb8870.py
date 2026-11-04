  for j in range(2):
      for k in range(2):
          ii = i_off + 2
          jj = j_off + j
          kk = k_off + k
          add_rank1_term([ii*m+jj], [1.0], [jj*p+kk], [1.0], [kk*n+ii], [1.0])
  