import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen for 2x2x2 (cols 0, 1 of B)
    # Indices:
    # U: i*m+j
    # V: j*p+k
    # W: k*n+i
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # k=0, i=0,1? No.
    # W index is k*n+i.
    # For C00 (i=0, k=0), W[0*2+0]=W[0].
    # For C11 (i=1, k=1), W[1*2+1]=W[3].
    # M1 contributes to C00, C11.
    # C00 = M1 - M4 + M6 - M7
    # C11 = M1 - M3 + M2 - M6
    
    # Let's define the 7 Strassen terms explicitly.
    # Term 0: M1
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[5, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1 # C00 (k=0,i=0) and C11 (k=1,i=1)
    
    # Term 1: M2 = (a10+a11)b00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1; W[3, 1] += 1 # C10 (k=0,i=1) and C11 (k=1,i=1)
    
    # Term 2: M3 = a00(b01-b11)
    U[0, 2] += 1
    V[1, 2] += 1; V[5, 2] -= 1
    W[1, 2] += 1; W[3, 2] -= 1 # C01 (k=1,i=0) and C11 (k=1,i=1)
    
    # Term 3: M4 = a11(b10-b00)
    U[3, 3] += 1
    V[4, 3] += 1; V[0, 3] -= 1
    W[0, 3] -= 1; W[1, 3] += 1 # C00 (k=0,i=0) and C10 (k=0,i=1)
    
    # Term 4: M5 = (a00+a01)b11
    U[0, 4] += 1; U[1, 4] += 1
    V[5, 4] += 1
    W[1, 4] += 1 # C01 (k=1,i=0)
    
    # Term 5: M6 = (a10-a00)(b00+b01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[0, 5] += 1; W[1, 5] -= 1 # C00 (k=0,i=0) and C11 (k=1,i=1)
    
    # Term 6: M7 = (a01-a11)(b10+b11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[4, 6] += 1; V[5, 6] += 1
    W[0, 6] -= 1; W[1, 6] -= 1 # C00 (k=0,i=0) and C10 (k=0,i=1)
    
    # Schoolbook for col 2
    # C02 = a00 b02 + a01 b12
    # C12 = a10 b02 + a11 b12
    
    # Term 7: a00 b02
    U[0, 7] += 1
    V[2, 7] += 1 # j=0, k=2 -> 0*3+2=2
    W[2, 7] += 1 # k=2, i=0 -> 2*2+0=4? No.
    # W index k*n+i. k=2, i=0 -> 2*2+0=4.
    W[4, 7] += 1
    
    # Term 8: a01 b12
    U[1, 8] += 1
    V[5, 8] += 1 # j=1, k=2 -> 1*3+2=5
    W[4, 8] += 1 # k=2, i=0 -> 4
    
    # Term 9: a10 b02
    U[2, 9] += 1
    V[2, 9] += 1 # j=0, k=2 -> 2
    W[5, 9] += 1 # k=2, i=1 -> 2*2+1=5
    
    # Term 10: a11 b12
    U[3, 10] += 1
    V[5, 10] += 1 # j=1, k=2 -> 5
    W[5, 10] += 1 # k=2, i=1 -> 5

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    # Verify
    n, m, p = 2, 2, 3
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6, 7], [8, 9, 10]])
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                for c in range(11):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            C_comp[i, k] = s
            
    print("True:", C_true)
    print("Comp:", C_comp)
    if np.allclose(C_true, C_comp):
        print("Valid Rank 11")
    else:
        print("Invalid")
