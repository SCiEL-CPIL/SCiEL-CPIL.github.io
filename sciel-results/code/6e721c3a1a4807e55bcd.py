"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115  # 5s safety margin before 120s kill

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix-multiplication tensor T[a, b, c] where
    T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0.
    Indexing: a = i*m+j, b = j'*p+k, c = i'*n+k'
    """
    T = np.zeros((n*m, m*p, n*p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * n + k
                T[a, b, c] = 1.0
    return T


def _check_decomposition(U, V, W, T, tol=1e-9):
    """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T"""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)


def _construct_strassen_2x2x2():
    """
    Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    A is 2x2 with indices (i,j): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    B is 2x2 with indices (j,k): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    C is 2x2 with indices (i,k): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    T[(i,j),(j',k),(i',k')] = 1 iff i=i', j=j', k=k'
    
    The 7 products:
    P1 = (a00+a11)*(b00+b11) -> C00 += P1, C11 += P1
    P2 = (a10+a11)*b01        -> C10 += P2, C11 += P2
    P3 = a00*(b10-b00)        -> C01 += P3, C11 -= P3  [wait, let me re-derive]
    
    Standard Strassen:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b10-b00)
    M4 = a11(b11-b01)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b10)
    M7 = (a01-a11)(b01+b11)
    
    C00 = M1 - M2 + M3 + M7
    C01 = M4 + M5
    C10 = M1 + M2 - M4 - M6
    C11 = M1 - M2 + M3 + M5 + M6  [wait, let me check]
    
    Actually:
    C00 = M1 - M4 + M5 + M7
    C01 = M2 + M4
    C10 = M1 + M3 - M5 - M7
    C11 = M1 - M3 + M6 + M7
    
    Let me use the standard formulation:
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C = [[M1+M4-M5+M7, M2+M3],
         [M6+M4,       M1-M3+M5+M6]]
    
    Mapping: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
             e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
             C[0,0], C[0,1], C[1,0], C[1,1]
    
    U indices (A): 0=a, 1=b, 2=c, 3=d
    V indices (B): 0=e, 1=f, 2=g, 3=h
    W indices (C): 0=C00, 1=C01, 2=C10, 3=C11
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a+d)(e+h): U: a=1,d=1; V: e=1,h=1; W: C00+=1, C11+=1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # M2 = (c+d)e: U: c=1,d=1; V: e=1; W: C01+=1, C10+=1
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1; W[2,1]=1
    
    # M3 = a(f-h): U: a=1; V: f=1,h=-1; W: C01+=1, C11-=1
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=-1
    
    # M4 = d(g-e): U: d=1; V: g=1,e=-1; W: C00+=1, C10+=1
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # M5 = (a+b)h: U: a=1,b=1; V: h=1; W: C00-=1, C11+=1
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[3,4]=1
    
    # M6 = (c-a)(e+f): U: c=1,a=-1; V: e=1,f=1; W: C10+=1, C11+=1
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[2,5]=1; W[3,5]=1
    
    # M7 = (b-d)(g+h): U: b=1,d=-1; V: g=1,h=1; W: C00+=1
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W


def _construct_2x4x6_strassen():
    """
    Rank-42 decomposition for 2x4 * 4x6 using 6 Strassen 2x2x2 blocks.
    
    A (2x4) split into A_L (cols 0-1) and A_R (cols 2-3), each 2x2.
    B (4x6) split into B_T (rows 0-1) and B_B (rows 2-3), each 2x6.
    B_T split into 3 2x2 blocks by columns: B_T0 (cols 0-1), B_T1 (cols 2-3), B_T2 (cols 4-5).
    B_B split similarly: B_B0, B_B1, B_B2.
    
    C = A_L @ B_T + A_R @ B_B
      = A_L @ [B_T0 B_T1 B_T2] + A_R @ [B_B0 B_B1 B_B2]
    
    Each 2x2 * 2x2 product uses Strassen (rank 7).
    6 blocks × 7 = 42 terms.
    
    Indexing:
    U: A[i,j], i in {0,1}, j in {0..3} -> index i*4+j
    V: B[j,k], j in {0..3}, k in {0..5} -> index j*6+k
    W: C[i,k], i in {0,1}, k in {0..5} -> index k*2+i (trace convention W[k*n+i])
    """
    n, m, p = 2, 4, 6
    U_str, V_str, W_str = _construct_strassen_2x2x2()
    
    total_rank = 42
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    # 6 blocks: (a_block, k_block) where a_block in {0,1}, k_block in {0,1,2}
    # a_block=0: A cols 0-1, B rows 0-1
    # a_block=1: A cols 2-3, B rows 2-3
    # k_block: 0->k=0,1; 1->k=2,3; 2->k=4,5
    
    for a_block in range(2):
        for k_block in range(3):
            base = (a_block * 3 + k_block) * 7
            for r in range(7):
                col = base + r
                
                # U: A[i, j] where j = a_block*2 + j_loc, i in {0,1}
                # U_str[i, r] gives coefficient for A[i, j_loc]
                for i in range(2):
                    for j_loc in range(2):
                        j = a_block * 2 + j_loc
                        U[i * m + j, col] += U_str[i, r]
                
                # V: B[j, k] where j = a_block*2 + j_loc, k = k_block*2 + k_loc
                # V_str[j_loc, k_loc] gives coefficient for B[j, k]
                for j_loc in range(2):
                    j = a_block * 2 + j_loc
                    for k_loc in range(2):
                        k = k_block * 2 + k_loc
                        V[j * p + k, col] += V_str[j_loc, k_loc]
                
                # W: C[i, k] where k = k_block*2 + k_loc
                # W_str[i, k_loc] gives coefficient for C[i, k]
                # W index: k*n + i
                for i in range(2):
                    for k_loc in range(2):
                        k = k_block * 2 + k_loc
                        W[k * n + i, col] += W_str[i, k_loc]
    
    return U, V, W


def _construct_schoolbook():
    """Trivial rank-48 decomposition."""
    n, m, p = 2, 4, 6
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    # Strassen-based decomposition (rank 42)
    U, V, W = _construct_2x4x6_strassen()
    T = _build_tensor(n, m, p)
    
    if _check_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (rank 48)
    U, V, W = _construct_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
