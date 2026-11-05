import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=int)  # 8 x 56
    V = np.zeros((m * p, rank), dtype=int)  # 32 x 56
    W = np.zeros((p * n, rank), dtype=int)  # 16 x 56
    
    col = 0
    
    # We have 8 blocks of 2x2x2
    # j0 in {0, 2}
    # k0 in {0, 2, 4, 6}
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Local indices:
            # i in {0, 1}
            # j in {j0, j0+1}
            # k in {k0, k0+1}
            
            # Strassen terms for 2x2x2
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # U index: i*m + j
            # V index: j*p + k
            # W index: k*n + i
            
            # Term 1: M1
            # U: A00 (i=0, j=j0) -> 1; A11 (i=1, j=j0+1) -> 1
            # V: B00 (j=j0, k=k0) -> 1; B11 (j=j0+1, k=k0+1) -> 1
            # W: C00 (k=k0, i=0) -> 1; C11 (k=k0+1, i=1) -> 1
            U[0*m + j0, col] = 1
            U[1*m + j0+1, col] = 1
            V[j0*p + k0, col] = 1
            V[(j0+1)*p + k0+1, col] = 1
            W[k0*n + 0, col] = 1
            W[(k0+1)*n + 1, col] = 1
            col += 1
            
            # Term 2: M2 = (A10 + A11)B00
            # U: A10 (i=1, j=j0) -> 1; A11 (i=1, j=j0+1) -> 1
            # V: B00 (j=j0, k=k0) -> 1
            # W: C10 (k=k0, i=1) -> 1; C11 (k=k0+1, i=1) -> -1
            U[1*m + j0, col] = 1
            U[1*m + j0+1, col] = 1
            V[j0*p + k0, col] = 1
            W[k0*n + 1, col] = 1
            W[(k0+1)*n + 1, col] = -1
            col += 1
            
            # Term 3: M3 = A00(B01 - B11)
            # U: A00 (i=0, j=j0) -> 1
            # V: B01 (j=j0, k=k0+1) -> 1; B11 (j=j0+1, k=k0+1) -> -1
            # W: C01 (k=k0+1, i=0) -> 1; C11 (k=k0+1, i=1) -> 1
            U[0*m + j0, col] = 1
            V[j0*p + k0+1, col] = 1
            V[(j0+1)*p + k0+1, col] = -1
            W[(k0+1)*n + 0, col] = 1
            W[(k0+1)*n + 1, col] = 1
            col += 1
            
            # Term 4: M4 = A11(B10 - B00)
            # U: A11 (i=1, j=j0+1) -> 1
            # V: B10 (j=j0+1, k=k0) -> 1; B00 (j=j0, k=k0) -> -1
            # W: C00 (k=k0, i=0) -> 1; C10 (k=k0, i=1) -> 1
            U[1*m + j0+1, col] = 1
            V[(j0+1)*p + k0, col] = 1
            V[j0*p + k0, col] = -1
            W[k0*n + 0, col] = 1
            W[k0*n + 1, col] = 1
            col += 1
            
            # Term 5: M5 = (A00 + A01)B11
            # U: A00 (i=0, j=j0) -> 1; A01 (i=0, j=j0+1) -> 1
            # V: B11 (j=j0+1, k=k0+1) -> 1
            # W: C00 (k=k0, i=0) -> -1; C01 (k=k0+1, i=0) -> 1
            U[0*m + j0, col] = 1
            U[0*m + j0+1, col] = 1
            V[(j0+1)*p + k0+1, col] = 1
            W[k0*n + 0, col] = -1
            W[(k0+1)*n + 0, col] = 1
            col += 1
            
            # Term 6: M6 = (A10 - A00)(B00 + B01)
            # U: A10 (i=1, j=j0) -> 1; A00 (i=0, j=j0) -> -1
            # V: B00 (j=j0, k=k0) -> 1; B01 (j=j0, k=k0+1) -> 1
            # W: C11 (k=k0+1, i=1) -> 1
            U[1*m + j0, col] = 1
            U[0*m + j0, col] = -1
            V[j0*p + k0, col] = 1
            V[j0*p + k0+1, col] = 1
            W[(k0+1)*n + 1, col] = 1
            col += 1
            
            # Term 7: M7 = (A01 - A11)(B10 + B11)
            # U: A01 (i=0, j=j0+1) -> 1; A11 (i=1, j=j0+1) -> -1
            # V: B10 (j=j0+1, k=k0) -> 1; B11 (j=j0+1, k=k0+1) -> 1
            # W: C00 (k=k0, i=0) -> 1
            U[0*m + j0+1, col] = 1
            U[1*m + j0+1, col] = -1
            V[(j0+1)*p + k0, col] = 1
            V[(j0+1)*p + k0+1, col] = 1
            W[k0*n + 0, col] = 1
            col += 1
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
