   def construct():
       n, m, p = 3, 3, 7
       rank = 55
       U = [[0] * rank for _ in range(n * m)]
       V = [[0] * rank for _ in range(m * p)]
       W = [[0] * rank for _ in range(p * n)]
       
       col = 0
       
       # Laderman for block 0 (cols 0,1,2)
       # ... add 23 terms
       
       # Laderman for block 1 (cols 3,4,5)
       # ... add 23 terms
       
       # Schoolbook for block 2 (col 6)
       # ... add 9 terms
       
       return {"U": U, "V": V, "W": W}
   