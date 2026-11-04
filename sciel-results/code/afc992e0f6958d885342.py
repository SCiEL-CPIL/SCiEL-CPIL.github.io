<<<<<<< SEARCH
    # Add schoolbook terms for the rest
    # Covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # Remaining:
    # 1. i=2, any j, any k
    # 2. i in {0,1}, any j, k=4
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if covered by Strassen
                if i in [0, 1] and j in [0, 1, 2, 3] and k in [0, 1, 2, 3]:
                    continue
                
                # Schoolbook term
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
=======
    # Add schoolbook terms for the rest
    # Covered by Strassen: i in {0,1}, j in {0,1}, k in {0,1} = 8 terms
    # Remaining: all other (i,j,k) terms = 60 - 8 = 52 terms
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if covered by Strassen (only the 2x2x2 region)
                if i in [0, 1] and j in [0, 1] and k in [0, 1]:
                    continue
                
                # Schoolbook term
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
>>>>>>> REPLACE

