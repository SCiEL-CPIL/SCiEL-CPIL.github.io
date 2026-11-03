   def get_tensor(U, V, W):
       # U: nm x R, V: mp x R, W: pn x R
       # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
       U_r = U.reshape(n, m, R)
       V_r = V.reshape(m, p, R)
       W_r = W.reshape(p, n, R)
       
       # T[i, j, k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
       # This is a contraction over r.
       # We can use einsum: 'ijr,jkr,kir->ijk'
       T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
       return T
   