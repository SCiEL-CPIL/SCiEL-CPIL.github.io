import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two decompositions.
    If M_{a,b,c} = sum U1[i,b+k] V1[k,c+j] W1[j,a+i] over rank r1
    and M_{d,e,f} = sum U2[i,e+k] V2[k,f+j] W2[j,d+i] over rank r2
    Then M_{a*d, b*e, c*f} has rank r1*r2
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    pass

def _kron_decomp_dims(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Kronecker product of two decompositions with explicit dimensions."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    n, m, p = n1 * n2, m1 * m2, p1 * p2
    r = r1 * r2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for i1 in range(n1):
        for k1 in range(m1):
            for i2 in range(n2):
                for k2 in range(m2):
                    u_idx = (i1 * n2 + i2) * (m1 * m2) + (k1 * m2 + k2)
                    for c1 in range(r1):
                        for c2 in range(r2):
                            c = c1 * r2 + c2
                            U[u_idx][c] = U1[i1 * m1 + k1][c1] * U2[i2 * m2 + k2][c2]
    
    for k1 in range(m1):
        for j1 in range(p1):
            for k2 in range(m2):
                for j2 in range(p2):
                    v_idx = (k1 * m2 + k2) * (p1 * p2) + (j1 * p2 + j2)
                    for c1 in range(r1):
                        for c2 in range(r2):
                            c = c1 * r2 + c2
                            V[v_idx][c] = V1[k1 * p1 + j1][c1] * V2[k2 * p2 + j2][c2]
    
    for j1 in range(p1):
        for i1 in range(n1):
            for j2 in range(p2):
                for i2 in range(n2):
                    w_idx = (j1 * p2 + j2) * (n1 * n2) + (i1 * n2 + i2)
                    for c1 in range(r1):
                        for c2 in range(r2):
                            c = c1 * r2 + c2
                            W[w_idx][c] = W1[j1 * n1 + i1][c1] * W2[j2 * n2 + i2][c2]
    
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Strassen's rank-7 decomposition for 2x2x2 matrix multiplication."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a10)b11
    # m6 = (a01-a10)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    # C00 = m1+m4-m5-m7
    # C01 = m3+m7
    # C10 = m2+m5
    # C11 = m1-m2+m3+m6
    
    # Tensor T[i,k,j] where i=row of A, k=col of A/row of B, j=col of B
    # T[i,k,j] = 1 if this is a valid multiplication term
    
    # For Strassen, we need to express the 8 schoolbook terms as 7 rank-1 terms
    # The 8 terms are: T[0,0,0], T[0,0,1], T[0,1,0], T[0,1,1], T[1,0,0], T[1,0,1], T[1,1,0], T[1,1,1]
    
    # U[i*m+k, c], V[k*p+j, c], W[j*n+i, c]
    # U indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # m1: (a00+a11)(b00+b11)
    # This contributes to C00, C11
    # U: a00 (idx 0) + a11 (idx 3) -> U[0]=1, U[3]=1
    # V: b00 (idx 0) + b11 (idx 3) -> V[0]=1, V[3]=1
    # W: contributes to C00 (j=0,i=0, idx 0) and C11 (j=1,i=1, idx 3)
    # W[0]=1, W[3]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2: (a10+a11)b00
    # U: a10 (idx 2) + a11 (idx 3) -> U[2]=1, U[3]=1
    # V: b00 (idx 0) -> V[0]=1
    # W: contributes to C10 (j=0,i=1, idx 2)
    # C10 = m2+m5, so W[2]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # m3: a00(b01-b11)
    # U: a00 (idx 0) -> U[0]=1
    # V: b01 (idx 1) - b11 (idx 3) -> V[1]=1, V[3]=-1
    # W: contributes to C01 (j=1,i=0, idx 1)
    # C01 = m3+m7, so W[1]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # m4: a11(b10-b00)
    # U: a11 (idx 3) -> U[3]=1
    # V: b10 (idx 2) - b00 (idx 0) -> V[2]=1, V[0]=-1
    # W: contributes to C00 (j=0,i=0, idx 0)
    # C00 = m1+m4-m5-m7, so W[0]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1
    
    # m5: (a00+a10)b11
    # U: a00 (idx 0) + a10 (idx 2) -> U[0]=1, U[2]=1
    # V: b11 (idx 3) -> V[3]=1
    # W: contributes to C10 (j=0,i=1, idx 2) and C00 (j=0,i=0, idx 0)
    # C10 = m2+m5, C00 = m1+m4-m5-m7
    # So W[2]=1, W[0]=-1
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1
    
    # m6: (a01-a10)(b00+b10)
    # U: a01 (idx 1) - a10 (idx 2) -> U[1]=1, U[2]=-1
    # V: b00 (idx 0) + b10 (idx 2) -> V[0]=1, V[2]=1
    # W: contributes to C11 (j=1,i=1, idx 3)
    # C11 = m1-m2+m3+m6, so W[3]=1
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # m7: (a01-a11)(b01+b11)
    # U: a01 (idx 1) - a11 (idx 3) -> U[1]=1, U[3]=-1
    # V: b01 (idx 1) + b11 (idx 3) -> V[1]=1, V[3]=1
    # W: contributes to C01 (j=1,i=0, idx 1) and C00 (j=0,i=0, idx 0)
    # C01 = m3+m7, C00 = m1+m4-m5-m7
    # So W[1]=1, W[0]=-1
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[0][6] = -1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_2x2x3():
    """Schoolbook rank-12 decomposition for 2x2x3."""
    n, m, p = 2, 2, 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank10_2x2x3():
    """Try to find a rank-10 decomposition for 2x2x3.
    
    The schoolbook has 12 terms. We try to merge some using linearity.
    """
    n, m, p = 2, 2, 3
    # Start with schoolbook and try to find linear dependencies
    # Actually, let me just use schoolbook for now and see if Kronecker helps
    # For 2x2x3, the minimum rank is believed to be 10
    
    # Let me try a constructive approach:
    # T[i,k,j] for i in {0,1}, k in {0,1}, j in {0,1,2}
    # 12 terms total
    
    # One approach: use the fact that we can combine terms
    # For example, T[0,0,0] + T[0,0,1] = (a00)(b00+b01)
    # But this doesn't directly help with rank reduction
    
    # Let me just return schoolbook for now
    return _schoolbook_2x2x3()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # M_{4,4,9} = M_{2,2,3} ⊗ M_{2,2,3}
    # Using schoolbook for 2x2x3: rank = 12*12 = 144 (same as schoolbook for 4x4x9)
    # Using Strassen for 2x2x2 doesn't directly apply
    
    # Alternative: M_{4,4,9} can be decomposed differently
    # 4x4x9 = (2x2x3) x (2x2x3) via Kronecker
    
    # Let me try: use Strassen-like for the 2x2 parts
    # Actually, let me think of it as:
    # M_{4,4,9} where 4=2*2, 9=3*3
    # = M_{2,2,3} ⊗ M_{2,2,3}
    
    # If I use rank-10 for 2x2x3, total rank = 100
    # If I use rank-12 (schoolbook), total rank = 144
    
    # Let me try to find a better 2x2x3 decomposition
    
    # For now, let me use the Kronecker product of two Strassen 2x2x2 decompositions
    # but that gives 4x4x4, not 4x4x9
    
    # Better approach: 
    # M_{4,4,9} = M_{2,2,3} ⊗ M_{2,2,3}
    # Use schoolbook 2x2x3 (rank 12) for both: total rank 144
    
    # To improve, I need a lower-rank 2x2x3 decomposition
    
    # Let me try to construct a rank-10 2x2x3 decomposition
    # by finding linear combinations
    
    n1, m1, p1 = 2, 2, 3
    n2, m2, p2 = 2, 2, 3
    
    # Use schoolbook for 2x2x3
    d1 = _schoolbook_2x2x3()
    d2 = _schoolbook_2x2x3()
    
    result = _kron_decomp_dims(d1["U"], d1["V"], d1["W"], n1, m1, p1,
                                d2["U"], d2["V"], d2["W"], n2, m2, p2)
    
    # Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
