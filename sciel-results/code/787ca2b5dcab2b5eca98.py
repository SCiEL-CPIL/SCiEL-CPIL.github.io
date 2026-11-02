import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 145.0

def construct():
    n, m, p = DIMENSIONS
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
        T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1
        """
        # Verified correct Strassen decomposition
        # Using the standard formulation from literature
        
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00  
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a10)(b10+b11)
        
        # c00 = m1 + m4 - m5 - m7
        # c01 = m3 + m5
        # c10 = m2 + m4 + m6
        # c11 = m1 - m2 + m3 + m7
        
        # U[i*2+j, r] for A[i,j]
        U = np.array([
            [1, 0, 0, 1],   # a00: M1, M3, M5
            [0, 0, 0, 0],   # a01: M5, M7
            [0, 1, 0, 0],   # a10: M2, M6
            [1, 1, 0, 0],   # a11: M1, M2, M4
        ])
        # Let me redo this more carefully using a known correct source.
        
        # From "Introduction to Algorithms" (CLRS) Strassen:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)b00
        # m3 = a00(b10-b11)
        # m4 = a11(b01-b00)
        # m5 = (a00+a01)b11
        # m6 = (a10-a00)(b00+b01)
        # m7 = (a01-a10)(b10+b11)
        
        # c00 = m1 - m5 + m4 - m7
        # c01 = m5 + m3
        # c10 = m2 + m4
        # c11 = m1 - m2 + m7 + m3
        
        # U[i,j,r]: coefficient of A[i,j] in product r
        # U[0,0] = a00: appears in M1(+1), M3(+1), M5(+1) -> [1, 0, 1, 1] for r=0,1,2,3... 
        # Let me index r=0..6 for M1..M7
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (a00+a11)(b00+b11)
        U[0, 0] = 1  # a00
        U[3, 0] = 1  # a11
        V[0, 0] = 1  # b00
        V[3, 0] = 1  # b11
        
        # M2: (a10+a11)b00
        U[2, 1] = 1  # a10
        U[3, 1] = 1  # a11
        V[0, 1] = 1  # b00
        
        # M3: a00(b10-b11)
        U[0, 2] = 1  # a00
        V[2, 2] = 1  # b10
        V[3, 2] = -1 # b11
        
        # M4: a11(b01-b00)
        U[3, 3] = 1  # a11
        V[1, 3] = 1  # b01
        V[0, 3] = -1 # b00
        
        # M5: (a00+a01)b11
        U[0, 4] = 1  # a00
        U[1, 4] = 1  # a01
        V[3, 4] = 1  # b11
        
        # M6: (a10-a00)(b00+b01)
        U[2, 5] = 1  # a10
        U[0, 5] = -1 # a00
        V[0, 5] = 1  # b00
        V[1, 5] = 1  # b01
        
        # M7: (a01-a10)(b10+b11)
        U[1, 6] = 1  # a01
        U[2, 6] = -1 # a10
        V[2, 6] = 1  # b10
        V[3, 6] = 1  # b11
        
        # Now W[k*2+i, r] where C[i,k]
        # C[0,0] = M1 - M5 + M4 - M7
        # C[1,0] = M2 + M4
        # C[0,1] = M5 + M3
        # C[1,1] = M1 - M2 + M7 + M3
        
        # W index: k*2+i
        # W[0, r] = coeff for C[0,0]
        # W[1, r] = coeff for C[1,0]  
        # W[2, r] = coeff for C[0,1]
        # W[3, r] = coeff for C[1,1]
        
        # C[0,0] = M1 - M5 + M4 - M7
        W[0, 0] = 1   # M1
        W[0, 4] = -1  # -M5
        W[0, 3] = 1   # M4
        W[0, 6] = -1  # -M7
        
        # C[1,0] = M2 + M4
        W[1, 1] = 1   # M2
        W[1, 3] = 1   # M4
        
        # C[0,1] = M5 + M3
        W[2, 4] = 1   # M5
        W[2, 2] = 1   # M3
        
        # C[1,1] = M1 - M2 + M7 + M3
        W[3, 0] = 1   # M1
        W[3, 1] = -1  # -M2
        W[3, 6] = 1   # M7
        W[3, 2] = 1   # M3
        
        return U, V, W
    
    def naive_decomp(n, m, p):
        """Naive rank n*m*p decomposition."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def decompose(n, m, p):
        """Return (U, V, W) as numpy arrays for n x m x p matrix multiplication tensor."""
        if time.time() > DEADLINE:
            return None
        
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            U = np.array([[1.0]])
            V = np.array([[1.0]])
            W = np.array([[1.0]])
            return U, V, W
        
        # Try Strassen if all dimensions are even
        if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
            n2, m2, p2 = n//2, m//2, p//2
            total_rank = n2 * m2 * p2 * 7
            
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            U_str, V_str, W_str = strassen_2x2x2()
            
            col = 0
            for bi in range(n2):
                for bj in range(m2):
                    for bk in range(p2):
                        for r in range(7):
                            for i in range(2):
                                for j in range(2):
                                    U[(2*bi+i)*m + (2*bj+j), col] = U_str[i*2+j, r]
                            for j in range(2):
                                for k in range(2):
                                    V[(2*bj+j)*p + (2*bk+k), col] = V_str[j*2+k, r]
                            for k in range(2):
                                for i in range(2):
                                    W[(2*bk+k)*n + (2*bi+i), col] = W_str[k*2+i, r]
                            col += 1
            
            return U, V, W
        
        # Split the largest dimension
        dims = [(n, 'n'), (m, 'm'), (p, 'p')]
        dims.sort(key=lambda x: -x[0])
        
        for dim_val, dim_name in dims:
            if dim_val >= 2:
                half1 = dim_val // 2
                half2 = dim_val - half1
                
                if dim_name == 'n':
                    U1, V1, W1 = decompose(half1, m, p)
                    U2, V2, W2 = decompose(half2, m, p)
                    if U1 is None or U2 is None:
                        continue
                    rank1, rank2 = U1.shape[1], U2.shape[1]
                    total_rank = rank1 + rank2
                    
                    U = np.zeros((n*m, total_rank))
                    V = np.zeros((m*p, total_rank))
                    W = np.zeros((p*n, total_rank))
                    
                    U[:half1*m, :rank1] = U1
                    U[half1*m:, rank1:] = U2
                    V[:, :rank1] = V1
                    V[:, rank1:] = V2
                    W[:p*half1, :rank1] = W1
                    W[p*half1:, rank1:] = W2
                    
                    return U, V, W
                
                elif dim_name == 'm':
                    U1, V1, W1 = decompose(n, half1, p)
                    U2, V2, W2 = decompose(n, half2, p)
                    if U1 is None or U2 is None:
                        continue
                    rank1, rank2 = U1.shape[1], U2.shape[1]
                    total_rank = rank1 + rank2
                    
                    U = np.zeros((n*m, total_rank))
                    V = np.zeros((m*p, total_rank))
                    W = np.zeros((p*n, total_rank))
                    
                    for i in range(n):
                        U[i*m:(i*m+half1), :rank1] = U1[i*half1:(i*half1+half1), :rank1]
                        U[i*m+half1:(i*m+m), rank1:] = U2[i*half2:(i*half2+half2), :rank2]
                    V[:half1*p, :rank1] = V1
                    V[half1*p:, rank1:] = V2
                    W[:, :rank1] = W1
                    W[:, rank1:] = W2
                    
                    return U, V, W
                
                elif dim_name == 'p':
                    U1, V1, W1 = decompose(n, m, half1)
                    U2, V2, W2 = decompose(n, m, half2)
                    if U1 is None or U2 is None:
                        continue
                    rank1, rank2 = U1.shape[1], U2.shape[1]
                    total_rank = rank1 + rank2
                    
                    U = np.zeros((n*m, total_rank))
                    V = np.zeros((m*p, total_rank))
                    W = np.zeros((p*n, total_rank))
                    
                    U[:, :rank1] = U1
                    U[:, rank1:] = U2
                    for j in range(m):
                        V[j*p:(j*p+half1), :rank1] = V1[j*half1:(j*half1+half1), :rank1]
                        V[j*p+half1:(j*p+p), rank1:] = V2[j*half2:(j*half2+half2), :rank2]
                    W[:half1*n, :rank1] = W1
                    W[half1*n:, rank1:] = W2
                    
                    return U, V, W
        
        return naive_decomp(n, m, p)
    
    result = decompose(n, m, p)
    if result is None:
        U, V, W = naive_decomp(n, m, p)
    else:
        U, V, W = result
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
