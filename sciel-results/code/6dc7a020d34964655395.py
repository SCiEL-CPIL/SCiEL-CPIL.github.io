    # ... (Part 1 and Part 2 construction) ...
    
    rank = len(terms)
    
    # If the rank is worse than schoolbook, just return schoolbook
    if rank > n * m * p:
         # Schoolbook construction
         U = [[0] * (n*m*p) for _ in range(n*m)]
         V = [[0] * (n*m*p) for _ in range(m*p)]
         W = [[0] * (n*m*p) for _ in range(p*n)]
         c = 0
         for i in range(n):
             for j in range(m):
                 for k in range(p):
                     U[i*m+j][c] = 1
                     V[j*p+k][c] = 1
                     W[k*n+i][c] = 1
                     c += 1
         return {"U": U, "V": V, "W": W}

    # Otherwise, build from terms
    # ...
