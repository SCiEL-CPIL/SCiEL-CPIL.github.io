import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 3, 4, 6
    
    def verify(U, V, W):
        """Verify the decomposition is correct."""
        # Check dimensions
        rank = len(U[0])
        if len(U) != n*m or len(V) != m*p or len(W) != p*n:
            return False
        for row in U:
            if len(row) != rank:
                return False
        for row in V:
            if len(row) != rank:
                return False
        for row in W:
            if len(row) != rank:
                return False
        
        # Check tensor decomposition
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix mult, T[i,j,k] = 1 (the multiplication tensor)
        # Actually, the tensor is T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
        # The rank-1 decomposition: C = sum_c U_c * V_c * W_c (outer products)
        # U_c[i,j] * V_c[j,k] * W_c[k,i]... no, that's not right.
        
        # The correct check: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal the tensor value.
        # For matrix multiplication, the tensor T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in this representation).
        # Wait, no. The multiplication tensor T has T[i,j,k] = 1, and the product is C[i,k] = sum_j A[i,j] T[i,j,k] B[j,k].
        # So the rank-1 decomposition is: T = sum_c u_c ⊗ v_c ⊗ w_c where u_c[i,j] = U[i*m+j,c], v_c[j,k] = V[j*p+k,c], w_c[k,i] = W[k*n+i,c].
        # And we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Strassen-based decomposition
    def strassen_based():
        def strassen_222():
            U_s = [[0]*7 for _ in range(4)]
            V_s = [[0]*7 for _ in range(4)]
            W_s = [[0]*7 for _ in range(4)]
            
            # P1 = (A00+A11)*(B00+B11)
            U_s[0][0] = 1; U_s[3][0] = 1
            V_s[0][0] = 1; V_s[3][0] = 1
            W_s[0][0] = 1; W_s[3][0] = 1
            
            # P2 = (A10+A11)*B00
            U_s[2][1] = 1; U_s[3][1] = 1
            V_s[0][1] = 1
            W_s[1][1] = 1
            
            # P3 = A00*(B01-B11)
            U_s[0][2] = 1
            V_s[1][2] = 1; V_s[3][2] = -1
            W_s[2][2] = 1; W_s[3][2] = 1
            
            # P4 = A11*(B10-B00)
            U_s[3][3] = 1
            V_s[2][3] = 1; V_s[0][3] = -1
            W_s[0][3] = 1; W_s[1][3] = 1
            
            # P5 = (A00+A01)*B11
            U_s[0][4] = 1; U_s[1][4] = 1
            V_s[3][4] = 1
            W_s[0][4] = -1
            
            # P6 = (A10-A00)*(B00+B10)
            U_s[2][5] = 1; U_s[0][5] = -1
            V_s[0][5] = 1; V_s[2][5] = 1
            W_s[3][5] = 1
            
            # P7 = (A01-A11)*(B01+B11)
            U_s[1][6] = 1; U_s[3][6] = -1
            V_s[1][6] = 1; V_s[3][6] = 1
            W_s[0][6] = 1
            
            return U_s, V_s, W_s
        
        def schoolbook_122():
            U_s = [[0]*2 for _ in range(2)]
            V_s = [[0]*2 for _ in range(4)]
            W_s = [[0]*2 for _ in range(2)]
            U_s[0][0] = 1
            V_s[0][0] = 1; V_s[2][0] = 1
            W_s[0][0] = 1; W_s[1][0] = 1
            U_s[1][1] = 1
            V_s[1][1] = 1; V_s[3][1] = 1
            W_s[0][1] = 1; W_s[1][1] = 1
            return U_s, V_s, W_s
        
        total_rank = 0
        blocks = []
        for i_off, i_size in [(0, 2), (2, 1)]:
            for j_off in [0, 2]:
                for k_off in [0, 2, 4]:
                    if i_size == 2:
                        total_rank += 7
                        blocks.append((i_off, 2, j_off, 2, k_off, 2, 'strassen'))
                    else:
                        total_rank += 2
                        blocks.append((i_off, 1, j_off, 2, k_off, 2, 'schoolbook'))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for i_off, i_size, j_off, j_size, k_off, k_size, method in blocks:
            if method == 'strassen':
                U_s, V_s, W_s = strassen_222()
            else:
                U_s, V_s, W_s = schoolbook_122()
            
            for c in range(len(U_s[0])):
                for r in range(len(U_s)):
                    i_local = r // 2
                    j_local = r % 2
                    U[(i_off + i_local) * m + (j_off + j_local)][col + c] += U_s[r][c]
                
                for r in range(len(V_s)):
                    j_local = r // 2
                    k_local = r % 2
                    V[(j_off + j_local) * p + (k_off + k_local)][col + c] += V_s[r][c]
                
                for r in range(len(W_s)):
                    k_local = r // 2
                    i_local = r % 2
                    W[(k_off + k_local) * n + (i_off + i_local)][col + c] += W_s[r][c]
            
            col += len(U_s[0])
        
        return {"U": U, "V": V, "W": W}
    
    # Try Strassen-based first
    result = strassen_based()
    if verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fall back to schoolbook
    return schoolbook()
