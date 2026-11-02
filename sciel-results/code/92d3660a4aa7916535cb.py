   def get_strassen_2x2():
       # Returns U, V, W for 2x2x2 Strassen
       # U shape (4, 7), V shape (4, 7), W shape (4, 7)
       # Indices: A is 2x2, B is 2x2, C is 2x2.
       # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
       U = [[0]*7 for _ in range(4)]
       V = [[0]*7 for _ in range(4)]
       W = [[0]*7 for _ in range(4)]
       
       # P1 = (A11+A22)(B11+B22)
       # A11: U[0,0]=1, A22: U[3,0]=1
       # B11: V[0,0]=1, B22: V[3,0]=1
       U[0][0] = 1; U[3][0] = 1
       V[0][0] = 1; V[3][0] = 1
       # C11 += P1, C22 += P1
       # C11 is W[0*2+0]=W[0], C22 is W[3*2+3]? No, W[k*n+i]. n=2.
       # C[i,j]. Index in W is j*2+i?
       # Prompt: W[k*n+i, r]. Here k is col index?
       # In seed: W[k*n+i][col]. k is col of C, i is row of C.
       # So W[j*2+i] corresponds to C[i,j].
       # P1 contributes to C11 (i=0,j=0 -> W[0]) and C22 (i=1,j=1 -> W[3]).
       W[0][0] = 1; W[3][0] = 1
       
       # P2 = (A21+A22)B11
       # A21: U[2,1]=1, A22: U[3,1]=1
       # B11: V[0,1]=1
       U[2][1] = 1; U[3][1] = 1
       V[0][1] = 1
       # C21 += P2 (i=1,j=0 -> W[1])
       W[1][1] = 1
       
       # P3 = A11(B12-B22)
       # A11: U[0,2]=1
       # B12: V[1,2]=1, B22: V[3,2]=-1
       U[0][2] = 1
       V[1][2] = 1; V[3][2] = -1
       # C12 += P3 (i=0,j=1 -> W[2])
       W[2][2] = 1
       
       # P4 = A22(B21-B11)
       # A22: U[3,3]=1
       # B21: V[2,3]=1, B11: V[0,3]=-1
       U[3][3] = 1
       V[2][3] = 1; V[0][3] = -1
       # C21 += P4 (i=1,j=0 -> W[1]), C11 += P4 (i=0,j=0 -> W[0])
       W[1][3] = 1; W[0][3] = 1
       
       # P5 = (A11+A12)B22
       # A11: U[0,4]=1, A12: U[1,4]=1
       # B22: V[3,4]=1
       U[0][4] = 1; U[1][4] = 1
       V[3][4] = 1
       # C11 -= P5 (i=0,j=0 -> W[0]), C12 += P5 (i=0,j=1 -> W[2])
       W[0][4] = -1; W[2][4] = 1
       
       # P6 = (A21-A11)(B11+B12)
       # A21: U[2,5]=1, A11: U[0,5]=-1
       # B11: V[0,5]=1, B12: V[1,5]=1
       U[2][5] = 1; U[0][5] = -1
       V[0][5] = 1; V[1][5] = 1
       # C22 += P6 (i=1,j=1 -> W[3])
       W[3][5] = 1
       
       # P7 = (A12-A22)(B21+B22)
       # A12: U[1,6]=1, A22: U[3,6]=-1
       # B21: V[2,6]=1, B22: V[3,6]=1
       U[1][6] = 1; U[3][6] = -1
       V[2][6] = 1; V[3][6] = 1
       # C11 += P7 (i=0,j=0 -> W[0]), C22 -= P7?
       # Formula: C22 = P1 - P2 + P3 + P6.
       # Wait, standard Strassen:
       # C11 = P1 + P4 - P5 + P7
       # C12 = P3 + P5
       # C21 = P2 + P4
       # C22 = P1 - P2 + P3 + P6
       # So P7 only in C11.
       W[0][6] = 1
       
       return U, V, W
   