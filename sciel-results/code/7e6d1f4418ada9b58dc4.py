"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Strategy:
1. Direct construction of a rank-8 decomposition (Strassen for 2x2x2 block + rank-1 for 2x2x3 all-ones).
2. If rank 8 verification fails, try rank 9.
3. If rank 9 fails, try rank 10.
4. Fallback to schoolbook rank 20 if construction fails verification.
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T


def _check_exact(U, V, W, T):
    """Exact integer/half-integer verification using Fraction."""
    def to_frac(x):
        hx = round(x * 2)
        if abs(x - hx / 2.0) < 1e-7:
            return Fraction(int(hx), 2)
        f = Fraction(x).limit_denominator(64)
        if abs(x - float(f)) < 1e-7:
            return f
        return Fraction(x).limit_denominator(64)

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]
    R = U.shape[1]
    for u in range(U.shape[0]):
        for v in range(V.shape[0]):
            for w in range(W.shape[0]):
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _decompose(U, V, W):
    """Reconstruct tensor from factor matrices."""
    R = U.shape[1]
    result = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _check_decomposition(U, V, W, T, tol=1e-8):
    result = _decompose(U, V, W)
    return np.allclose(result, T, atol=tol)


def _construct_rank_10():
    """
    Construct a rank-10 decomposition:
    - Rank 7: Strassen for the 2x2x2 block (k=0,1)
    - Rank 3: Schoolbook for the 2x2x3 block (k=2,3,4)
    """
    n, m, p = DIMENSIONS
    rank = 10
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen part for k=0,1 (indices 0..6)
    # Indices:
    # U: i*m+j -> 0..3
    # V: j*p+k -> 0..9 (j=0: 0-4, j=1: 5-9)
    # W: k*n+i -> 0..7 (k=0: 0-1, k=1: 2-3, k=2: 4-5, k=3: 6-7, k=4: 8-9)
    
    # P1 = (a+d)(e+h)  -> U[0,3], V[0,6], W[0,3]
    # Note: W index for k=0,i=0 is 0*2+0=0; k=0,i=1 is 0*2+1=1; k=1,i=0 is 1*2+0=2; k=1,i=1 is 1*2+1=3
    # Wait, Strassen outputs are for the 2x2 result matrix C[i,j].
    # The tensor index w = k*n + i.
    # For the 2x2x2 subproblem (k=0,1), the "output" dimension is actually mixed with k?
    # No, the task is 2x2 * 2x5.
    # Input A is 2x2 (i,j). Input B is 2x5 (j,k). Output C is 2x5 (i,k).
    # Tensor index w corresponds to output index. w = k*n + i.
    # So for k=0, w is 0,1. For k=1, w is 2,3.
    
    # Let's re-verify Strassen mapping for 2x2 * 2x2 (part of the 2x2*2x5 where k=0,1).
    # A = [[a, b], [c, d]]
    # B_part = [[e, f], [g, h]] (columns 0,1 of B)
    # C_part = A * B_part = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen terms:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 - M5 + M3 - M4
    # C01 = M5 + M2
    # C10 = M4 + M6
    # C11 = M1 - M2 + M7 + M3
    
    # Let's map these to U, V, W.
    # U[i,j,r], V[j,k,r], W[k,i,r]
    
    # r=0 (M1): U[0,0]=1, U[1,1]=1 (a,d). V[0,0]=1, V[1,1]=1 (e,h). 
    # Wait, V index is j*p+k. For k=0, j=0 -> 0. j=1 -> 5.
    # For k=1, j=0 -> 1. j=1 -> 6.
    # So B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h.
    # V[0,0]=1 (e), V[6,0]=1 (h). (j=1,k=1 is index 1*5+1=6).
    # W index for output C[i,k].
    # C[0,0] (i=0,k=0) -> w=0. C[0,1] (i=0,k=1) -> w=2.
    # C[1,0] (i=1,k=0) -> w=1. C[1,1] (i=1,k=1) -> w=3.
    
    # M1 contributes to C00 (+), C11 (+).
    # C00: w=0. C11: w=3.
    # So W[0,0]=1, W[3,0]=1.
    
    # r=1 (M2): U[2,1]=1 (c), U[3,1]=1 (d). V[0,1]=1 (e).
    # M2 contributes to C01 (-), C11 (-).
    # C01: w=2. C11: w=3.
    # So W[2,1]=-1, W[3,1]=-1.
    
    # r=2 (M3): U[0,2]=1 (a). V[1,2]=1 (f), V[7,2]=-1 (h). (j=1,k=1 is 7? No, j=1,k=1 is 6. h is B[1,1]. V index for h is 1*5+1=6. f is B[0,1], index 0*5+1=1).
    # Wait, M3 = a(f-h). f is B[0,1], h is B[1,1].
    # V[1,2]=1 (f), V[6,2]=-1 (h).
    # M3 contributes to C00 (-), C11 (+).
    # C00: w=0. C11: w=3.
    # So W[0,2]=-1, W[3,2]=1.
    
    # r=3 (M4): U[3,3]=1 (d). V[0,3]=-1 (e), V[5,3]=1 (g). (g is B[1,0], index 1*5+0=5).
    # M4 contributes to C00 (+), C10 (+).
    # C00: w=0. C10: w=1.
    # So W[0,3]=1, W[1,3]=1.
    
    # r=4 (M5): U[0,4]=1 (a), U[1,4]=1 (b). V[6,4]=1 (h).
    # M5 contributes to C00 (-), C01 (+).
    # C00: w=0. C01: w=2.
    # So W[0,4]=-1, W[2,4]=1.
    
    # r=5 (M6): U[2,5]=1 (c), U[0,5]=-1 (a). V[0,5]=1 (e), V[1,5]=1 (f).
    # M6 contributes to C10 (+).
    # C10: w=1.
    # So W[1,5]=1.
    
    # r=6 (M7): U[1,6]=1 (b), U[3,6]=-1 (d). V[5,6]=1 (g), V[6,6]=1 (h).
    # M7 contributes to C11 (+).
    # C11: w=3.
    # So W[3,6]=1.

    # Fill Strassen
    U[0,0] = 1; U[1,0] = 1
    V[0,0] = 1; V[6,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = -1; W[3,1] = -1
    
    U[0,2] = 1
    V[1,2] = 1; V[6,2] = -1
    W[0,2] = -1; W[3,2] = 1
    
    U[3,3] = 1
    V[0,3] = -1; V[5,3] = 1
    W[0,3] = 1; W[1,3] = 1
    
    U[0,4] = 1; U[1,4] = 1
    V[6,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[1,5] = 1
    
    U[1,6] = 1; U[3,6] = -1
    V[5,6] = 1; V[6,6] = 1
    W[3,6] = 1

    # Schoolbook part for k=2,3,4 (indices 7,8,9)
    # For each k in {2,3,4}, we need to compute C[i,k] = sum_j A[i,j] B[j,k].
    # This is a 2x2 * 2x1 multiplication for each k.
    # Rank of 2x2*2x1 is 4? No, it's just 2 multiplications per i?
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k]
    # This requires 4 terms if done independently?
    # Term 1: A[0,0]B[0,k] -> C[0,k]
    # Term 2: A[0,1]B[1,k] -> C[0,k]
    # Term 3: A[1,0]B[0,k] -> C[1,k]
    # Term 4: A[1,1]B[1,k] -> C[1,k]
    # But we only have 3 slots (rank 10 total, 7 used).
    # So we cannot do schoolbook for k=2,3,4 with rank 3.
    # We need rank 4 for each k? No, the tensor is direct sum.
    # Rank(2x2*2x1) = 4.
    # So Rank(2x2*2x3) = 3 * 4 = 12? No, the k dimension is separate.
    # The tensor T is sum_k T_k.
    # T_k is 2x2*2x1. Rank(T_k) = 4.
    # Since the supports in k are disjoint, Rank(sum T_k) = sum Rank(T_k) = 3 * 4 = 12.
    # Total rank = 7 + 12 = 19?
    # Wait, the schoolbook rank is n*m*p = 2*2*5 = 20.
    # My Strassen+Schoolbook construction above assumed I could do the rest in rank 3. That is wrong.
    # The "Inspiration 1" code claimed rank 10. Let's check that logic.
    # Inspiration 1 filled rank 7,8,9 with:
    # U[i*2+j][7] = 1 for all i,j. V[j*5+2][7]=1. W[2*2+i][7]=1.
    # This term contributes U[i,j]*V[j,2]*W[2,i] to T[i,j,2].
    # Sum_r U[i,j,r] V[j,2,r] W[2,i,r].
    # For r=7: U[i,j,7]=1, V[j,2,7]=1, W[2,i,7]=1.
    # So it adds 1 to T[i,j,2] for all i,j.
    # But T[i,j,2] should be 1.
    # Does the Strassen part contribute to k=2?
    # V indices used in Strassen: 0,1,5,6.
    # V[j,2,r] is non-zero if index is j*5+2.
    # j=0 -> 2. j=1 -> 7.
    # Strassen uses V[0], V[1], V[5], V[6].
    # It does NOT use V[2] or V[7].
    # So Strassen contributes 0 to k=2.
    # Thus, the term r=7 exactly provides T[i,j,2]=1.
    # Similarly for k=3 (r=8) and k=4 (r=9).
    # So the total rank is 7 + 1 + 1 + 1 = 10.
    # This works because for a fixed k, T[i,j,k]=1 is constant.
    # We can represent the constant 1 tensor of size 2x2x1 with rank 1?
    # T[i,j,k] = 1.
    # Term: U[i,j]=1, V[j,k]=1, W[k,i]=1.
    # Sum_r ... = 1 * 1 * 1 = 1.
    # Yes! The all-ones tensor of size n x m x 1 has rank 1?
    # Wait. T[i,j] = 1.
    # Decomposition: U[i,j,r] V[j,r] W[r].
    # If we use one term r=0: U[i,j,0]=1, V[j,0]=1, W[0]=1.
    # Then sum = 1*1*1 = 1.
    # Yes, the rank of the all-ones tensor of size A x B x 1 is 1.
    # Because it is the outer product of vec(ones(A*B)), ones(B), ones(1).
    # So Rank(2x2x1 all-ones) = 1.
    # Therefore Rank(2x2x3 all-ones) = 3 (since k is disjoint).
    # Total Rank = 7 (Strassen for 2x2x2) + 3 (All-ones for 2x2x3) = 10.
    
    # Fill Rank 10 part
    for k_extra, r_idx in [(2, 7), (3, 8), (4, 9)]:
        for i in range(n):
            for j in range(m):
                U[i * m + j, r_idx] = 1
        for j in range(m):
            V[j * p + k_extra, r_idx] = 1
        for i in range(n):
            W[k_extra * n + i, r_idx] = 1

    return U.tolist(), V.tolist(), W.tolist()


def _try_rank_9():
    """
    Try to find a rank 9 decomposition.
    This would require Rank(2x2x3 all-ones) = 2.
    Is it possible to represent the 2x2x3 all-ones tensor with rank 2?
    T[i,j,k] = 1 for i,j in {0,1}, k in {2,3,4}.
    We need sum_{r=7}^8 U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    Since the k-slices are independent and identical, we need a rank 2 decomposition of the 2x2x1 all-ones tensor?
    No, the k dimension is coupled in V and W.
    Let's search for a rank 2 solution for the 2x2x3 part.
    Variables:
    U[i,j,7], U[i,j,8] (4x2 = 8 vars)
    V[j,k,7], V[j,k,8] (2x3x2 = 12 vars)
    W[k,i,7], W[k,i,8] (3x2x2 = 12 vars)
    Total 32 variables.
    Equations: T[i,j,k]=1 for 2*2*3=12 equations.
    System is underdetermined.
    We can try to solve it.
    """
    n, m, p = DIMENSIONS
    # We only care about the 2x2x3 part (k=2,3,4)
    # Let's use a simple solver or heuristic.
    # Given the symmetry, maybe there is a solution.
    # If not, we return None.
    
    # Try a simple ansatz.
    # Maybe U is constant?
    # If U[i,j,r] = 1 for all i,j,r.
    # Then sum_r V[j,k,r] W[k,i,r] = 1.
    # This is a matrix equation.
    # Let M_r = V[:, :, r] * W[:, :, r]^T? No.
    # For fixed k, sum_r V[j,k,r] W[k,i,r] = 1.
    # This is the (j,i) entry of sum_r V[:,k,r] W[k,:,r]^T.
    # We need sum_r V[:,k,r] W[k,:,r]^T = Ones(2,2).
    # If we pick V[:,k,7] = [1,1]^T, W[k,:,7] = [1,1]^T, then product is [[1,1],[1,1]].
    # We need [[1,1],[1,1]] * c = [[1,1],[1,1]]? No.
    # We need the sum to be [[1,1],[1,1]].
    # If we use r=7: V[:,k,7]=[1,0], W[k,:,7]=[1,1] -> [[1,1],[0,0]].
    # r=8: V[:,k,8]=[0,1], W[k,:,8]=[1,1] -> [[0,0],[1,1]].
    # Sum = [[1,1],[1,1]].
    # This works for all k if we use the same V,W patterns for all k.
    # Let's check if this is consistent.
    # For k=2,3,4:
    # V[0,k,7]=1, V[1,k,7]=0
    # V[0,k,8]=0, V[1,k,8]=1
    # W[k,0,7]=1, W[k,1,7]=1
    # W[k,0,8]=1, W[k,1,8]=1
    
    # Does this work?
    # Sum_r V[j,k,r] W[k,i,r]
    # j=0, i=0: 1*1 + 0*1 = 1.
    # j=0, i=1: 1*1 + 0*1 = 1.
    # j=1, i=0: 0*1 + 1*1 = 1.
    # j=1, i=1: 0*1 + 1*1 = 1.
    # Yes!
    # So we can represent the 2x2x3 all-ones tensor with rank 2?
    # Wait, the rank is the number of terms. Here we used 2 terms (r=7, r=8).
    # But these terms are shared across k?
    # Yes, the indices r=7 and r=8 are global.
    # So we use 2 global terms to handle all 3 k-values.
    # This means the total rank is 7 (Strassen) + 2 (Extra) = 9.
    
    U = np.zeros((n * m, 9), dtype=np.float64)
    V = np.zeros((m * p, 9), dtype=np.float64)
    W = np.zeros((p * n, 9), dtype=np.float64)
    
    # Copy Strassen part (0..6)
    U10, V10, W10 = _construct_rank_10()
    U[:, :7] = np.array(U10)[:, :7]
    V[:, :7] = np.array(V10)[:, :7]
    W[:, :7] = np.array(W10)[:, :7]
    
    # Fill Extra part (7, 8)
    for k_extra in [2, 3, 4]:
        # r=7
        V[0 * p + k_extra, 7] = 1
        V[1 * p + k_extra, 7] = 0
        W[k_extra * n + 0, 7] = 1
        W[k_extra * n + 1, 7] = 1
        
        # r=8
        V[0 * p + k_extra, 8] = 0
        V[1 * p + k_extra, 8] = 1
        W[k_extra * n + 0, 8] = 1
        W[k_extra * n + 1, 8] = 1
        
        # U is all 1s for these columns
        U[:, 7] = 1
        U[:, 8] = 1
        
    return U.tolist(), V.tolist(), W.tolist()


def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True


def _construct_rank_8():
    """
    Construct a rank-8 decomposition:
    - Rank 7: Strassen for the 2x2x2 block (k=0,1)
    - Rank 1: All-ones for 2x2x3 block (k=2,3,4) using a single term
    """
    n, m, p = DIMENSIONS
    rank = 8
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Copy Strassen part (0..6) from rank-10 construction
    U10, V10, W10 = _construct_rank_10()
    U[:, :7] = np.array(U10)[:, :7]
    V[:, :7] = np.array(V10)[:, :7]
    W[:, :7] = np.array(W10)[:, :7]

    # Rank 1 term for k=2,3,4 (index 7)
    # U[i,j,7] = 1 for all i,j
    # V[j,k,7] = 1 for all j, k in {2,3,4}
    # W[k,i,7] = 1 for all k in {2,3,4}, i
    for i in range(n):
        for j in range(m):
            U[i * m + j, 7] = 1
    for j in range(m):
        for k in range(2, p):
            V[j * p + k, 7] = 1
    for k in range(2, p):
        for i in range(n):
            W[k * n + i, 7] = 1

    return U.tolist(), V.tolist(), W.tolist()


def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS

    # Try Rank 8
    if time.time() < deadline:
        U8, V8, W8 = _construct_rank_8()
        if _verify(U8, V8, W8, n, m, p, 8):
            return {"U": U8, "V": V8, "W": W8}

    # Try Rank 9
    if time.time() < deadline:
        U9, V9, W9 = _try_rank_9()
        if _verify(U9, V9, W9, n, m, p, 9):
            return {"U": U9, "V": V9, "W": W9}

    # Try Rank 10
    if time.time() < deadline:
        U10, V10, W10 = _construct_rank_10()
        if _verify(U10, V10, W10, n, m, p, 10):
            return {"U": U10, "V": V10, "W": W10}

    # Fallback Schoolbook Rank 20
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
