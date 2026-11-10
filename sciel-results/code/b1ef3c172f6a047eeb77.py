   def has_c6(u, v, adj):
       nu = adj[u] & ~(1 << v)
       nv = adj[v] & ~(1 << u)
       a_mask = nu
       while a_mask:
           la = a_mask & -a_mask
           a_mask ^= la
           a = la.bit_length() - 1
           b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a)
           while b_mask:
               lb = b_mask & -b_mask
               b_mask ^= lb
               b = lb.bit_length() - 1
               c_mask = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << b)
               while c_mask:
                   lc = c_mask & -c_mask
                   c_mask ^= lc
                   c = lc.bit_length() - 1
                   d_mask = adj[c] & nv & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << c)
                   if d_mask:
                       return True
       return False
   