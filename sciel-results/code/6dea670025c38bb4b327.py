   def strassen_decomposition(n, m, p):
       # n=m=p=4
       terms = []
       
       def recurse(a_blocks, b_blocks, i_off, j_off, k_off, size):
           if size == 1:
               # 1x1x1
               terms.append((i_off, j_off, k_off, 1.0))
               return
               
           half = size // 2
           
           # Split A into 4 blocks
           # A00, A01, A10, A11
           
           # Strassen steps
           # M1 = (A00+A11)(B00+B11)
           # ...
           
           # This is hard to implement recursively with block lists.
           
       