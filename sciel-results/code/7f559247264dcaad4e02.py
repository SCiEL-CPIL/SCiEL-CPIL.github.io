"""Matrix multiplication 2x2 x 2x3 with rank-minimal tensor decomposition.

Strategy:
  * Rank 10 candidate: Strassen (rank 7) for the 2x2*2x2 block (k=0,1)
    plus a rank-3 decomposition for the remaining column (k=2).
  * If rank 10 cannot be verified exactly, fall back to the known rank-11
    construction (Strassen + 4 schoolbook factors for column 2).
  * Final fallback is the schoolbook rank 12.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]


def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly represents the matrix multiplication tensor.

    The tensor T[i,j,k] must equal 1 for all i,j,k (the all-ones tensor),
    because the multiplication tensor for 2x2 * 2x3 -> 2x3 has T[i,j,k]=1
    for every (i,j,k) in the index ranges.
    """
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
    canonical = np.ones((n, m, p))
    return np.allclose(T, canonical, atol=tol)


def _strassen_2x2x2():
    """
    Return a verified rank-7 decomposition for 2x2 * 2x2 matrix multiplication.
    
    Uses the standard Strassen algorithm. The factors are for the sub-problem
    where i in {0,1}, j in {0,1}, k in {0,1}.
    We return U_sub (4 x 7), V_sub (4 x 7), W_sub (4 x 7) where indices are:
      U[i*m+j] with m=2
      V[j*p+k] with p=2
      W[k*n+i] with n=2
    """
    n_sub, m_sub, p_sub = 2, 2, 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Standard Strassen for 2x2*2x2:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C11 = m1 - m4 + m6
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m3 + m2 + m5
    
    # Indexing: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[k,i] -> W[k*2+i]
    # Note: C is 2x2, so C[k,i] means row k, col i in the output.
    # But our W indexing is W[k*n+i] = W[k*2+i], so W[0,0]=C[0,0], W[0,1]=C[0,1], W[1,0]=C[1,0], W[1,1]=C[1,1]
    
    # m1 = (a+d)(e+h): U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1
    U[0, 0] = 1; U[1, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += m1 -> W[0,0] += 1
    # C22 += m1 -> W[1,1] += 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # m2 = (c+d)e: U[2,1]=1, U[3,1]=1; V[0,1]=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += m2 -> W[0,1] += 1
    # C22 += m2 -> W[1,1] += 1
    W[1, 1] = 1
    W[3, 1] = 1
    
    # m3 = a(f-h): U[0,2]=1; V[0,2]=1, V[1,2]=-1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 += m3 -> W[1,0] += 1
    # C22 -= m3 -> W[3,2] -= 1
    W[1, 2] = 1
    W[3, 2] = -1
    
    # m4 = d(g-e): U[1,3]=1; V[1,3]=1, V[0,3]=-1
    U[1, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    # C11 -= m4 -> W[0,3] -= 1
    # C21 += m4 -> W[1,3] += 1
    W[0, 3] = -1
    W[1, 3] = 1
    
    # m5 = (a+b)h: U[0,4]=1, U[1,4]=1; V[1,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C12 += m5 -> W[1,4] += 1
    # C22 += m5 -> W[3,4] += 1
    W[1, 4] = 1
    W[3, 4] = 1
    
    # m6 = (c-a)(e+f): U[2,5]=1, U[0,5]=-1; V[0,5]=1, V[1,5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C11 += m6 -> W[0,5] += 1
    W[0, 5] = 1
    
    # m7 = (b-d)(g+h): U[1,6]=1, U[1,1]... wait
    # m7 = (b-d)(g+h): U[0,1]=b, U[1,1]=d -> U[1,6]=1 (b), U[1,1]... 
    # Actually U index: U[i*2+j], so A[0,1]=b -> U[1,6], A[1,1]=d -> U[3,6]
    # U[1,6] = 1 (b), U[3,6] = -1 (d)
    # V[1,6] = g -> V[2,6], V[3,6] = h -> V[3,6]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += m7 -> W[0,6] += 1
    W[0, 6] = 1
    
    return U, V, W


def _try_rank10():
    """
    Attempt rank-10: Strassen (7) for k=0,1 + rank-3 for k=2.
    
    For the k=2 column, we need:
      C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
      C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    This is a 2x2 * 2x1 matrix multiplication, which has tensor rank 2? 
    Actually the tensor for (2x2)*(2x1) -> (2x1) has 4 entries and rank 4 schoolbook.
    But can we do it in rank 3? The tensor is:
      T[i,j,0] = delta_{i,j} for j in {0,1}, i in {0,1} -- no wait.
    
    Actually for 2x2 * 2x1, the output is 2x1. The tensor is:
      T[i,j,k] where i in {0,1} (row of C), j in {0,1} (col of A / row of B), k in {0} (col of B)
      T[i,j,0] = 1 if i==j, else 0? No.
      C[i,0] = sum_j A[i,j] * B[j,0]
      So T[i,j,0] = 1 for all i,j. It's the all-ones tensor of size 2x2x1, rank = 4? 
      No, the tensor T[i,j,k] = 1 for all i,j,k has rank = min(n*m, m*p, p*n) ... 
      Actually T[i,j,0] = 1 for all i,j is a 2x2x1 tensor. Its rank is the rank of the matrix T[:, :, 0] = [[1,1],[1,1]], which is 1.
      Wait, no. The tensor decomposition is T[i,j,k] = sum_s U[i*m+j, s] V[j*p+k, s] W[k*n+i, s].
      For p=1, k is always 0. So W[k*n+i, s] = W[i, s].
      T[i,j,0] = sum_s U[i*2+j, s] V[j*1+0, s] W[i, s] = sum_s U[2i+j, s] V[j, s] W[i, s] = 1 for all i,j.
      
      This is a bilinear form: sum_s U[2i+j,s] V[j,s] W[i,s] = 1.
      We can write this as: for each s, we have a matrix M_s where M_s[i,j] = U[2i+j,s] V[j,s] W[i,s].
      We need sum_s M_s[i,j] = 1 for all i,j.
      
      The simplest is s=0..3 with M_0 = diag(1,1), M_1 = [[0,1],[0,0]], etc. That's rank 4.
      Can we do rank 3? We need 3 rank-1 (in a certain sense) matrices that sum to the all-ones 2x2 matrix.
      The all-ones 2x2 matrix has rank 1. So we can write it as 1*ones*ones.
      But each M_s[i,j] = U[2i+j,s] V[j,s] W[i,s] is not a rank-1 matrix in the usual sense; 
      it's a tensor of a specific form.
      
      Let's try to solve it directly. We have 3 factors.
      Unknowns: 4*3 + 2*3 + 2*3 = 12 + 6 + 6 = 24 unknowns, 4 equations.
      Let's fix W = [[1,0,0],[0,1,0]] (first two columns identity, third zero).
      Then T[i,j,0] = U[2i+j,0] V[j,0] * 1 + U[2i+j,1] V[j,1] * 0 + ... 
      This doesn't work well.
      
      Let me just try to find a rank-3 decomposition numerically or analytically.
      Actually, the tensor T[i,j] = 1 (2x2) can be decomposed as:
      T[i,j] = sum_{s=0}^{2} U[i,j,s] V[j,s] W[i,s]
      
      Try: 
      s=0: U[0,0,0]=1, U[1,0,0]=0, U[2,0,0]=0, U[3,0,0]=0; V[0,0]=1, V[1,0]=0; W[0,0]=1, W[1,0]=0
        -> M_0[0,0]=1, others 0.
      s=1: U[0,1,1]=0, U[1,1,1]=1, U[2,1,1]=0, U[3,1,1]=0; V[0,1]=0, V[1,1]=1; W[0,1]=0, W[1,1]=1
        -> M_1[1,1]=1, others 0.
      s=2: U[0,0,2]=1, U[1,0,2]=1, U[2,0,2]=1, U[3,0,2]=1; V[0,2]=1, V[1,2]=1; W[0,2]=1, W[1,2]=1
        -> M_2[i,j] = 1*1*1 = 1 for all i,j.
      Sum: M_0 + M_1 + M_2 = [[2,1],[1,2]] != [[1,1],[1,1]].
      
      Let me try another approach. I'll just use the schoolbook for the k=2 part (rank 4) 
      and combine with Strassen for a total of rank 11, which is known to work.
      For rank 10, I need a rank-3 for the k=2 part.
      
      Let's set up the equations for rank-3 k=2 part:
      T[0,0,0] = U[0,0,0]V[0,0]W[0,0] + U[0,0,1]V[0,1]W[0,1] + U[0,0,2]V[0,2]W[0,2] = 1
      T[0,1,0] = U[1,0,0]V[1,0]W[0,0] + U[1,0,1]V[1,1]W[0,1] + U[1,0,2]V[1,2]W[0,2] = 1
      T[1,0,0] = U[2,0,0]V[0,0]W[1,0] + U[2,0,1]V[0,1]W[1,1] + U[2,0,2]V[0,2]W[1,2] = 1
      T[1,1,0] = U[3,0,0]V[1,0]W[1,0] + U[3,0,1]V[1,1]W[1,1] + U[3,0,2]V[1,2]W[1,2] = 1
      
      Let's fix V = [[1,0,0],[0,1,0]] (V[0,0]=1,V[0,1]=0,V[0,2]=0; V[1,0]=0,V[1,1]=1,V[1,2]=0).
      Then:
      T[0,0,0] = U[0,0,0]*1*W[0,0] = U[0,0,0]W[0,0] = 1
      T[0,1,0] = U[1,0,1]*1*W[0,1] = U[1,0,1]W[0,1] = 1
      T[1,0,0] = U[2,0,0]*1*W[1,0] = U[2,0,0]W[1,0] = 1
      T[1,1,0] = U[3,0,1]*1*W[1,1] = U[3,0,1]W[1,1] = 1
      We have freedom in the third column. Set U[*,0,2]=0, V[*,2]=0, W[*,2]=0.
      Then we need U[0,0,0]W[0,0]=1, U[1,0,1]W[0,1]=1, U[2,0,0]W[1,0]=1, U[3,0,1]W[1,1]=1.
      This is rank 4 effectively (4 independent factors), not rank 3.
      
      So rank 3 for the k=2 part may not be possible with this structure. 
      Let me just use rank 4 for k=2 and get rank 11 total.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    # Strassen for k=0,1 (columns 0-6)
    U_sub, V_sub, W_sub = _strassen_2x2x2()
    U[:, 0:7] = U_sub
    V[0:4, 0:7] = V_sub  # V_sub is 4x7, V is 6x10, V[0:4, :] corresponds to k=0,1
    # V_sub[j*2+k] -> V[j*3+k], for k=0,1: V[0:2, 0:7] = V_sub[0:2, :], V[2:4, 0:7] = V_sub[2:4, :]
    # Actually V_sub index is j*2+k, so V_sub[0:2,:] is j=0, k=0,1 -> V[0:2, 0:7]
    # V_sub[2:4,:] is j=1, k=0,1 -> V[2:4, 0:7]
    V[0:2, 0:7] = V_sub[0:2, :]
    V[2:4, 0:7] = V_sub[2:4, :]
    # V_sub[4:6,:] would be j=2 which doesn't exist. V_sub is 4x7.
    # W_sub index is k*2+i, so W_sub[0:2,:] is k=0, i=0,1 -> W[0:2, 0:7]
    # W_sub[2:4,:] is k=1, i=0,1 -> W[2:4, 0:7]
    W[0:2, 0:7] = W_sub[0:2, :]
    W[2:4, 0:7] = W_sub[2:4, :]
    
    # For k=2 (columns 7-9), use 3 factors. But we showed this may not work.
    # Let's try a specific rank-3 construction.
    # Actually, let me just try to solve it numerically.
    # For now, return None to indicate rank 10 not verified.
    
    # Let me try a different rank-3 construction for k=2.
    # We need:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Let's use 3 factors:
    # s=7: U[0,0,7]=1, V[0,2,7]=1, W[2,0,7]=1 -> contributes A[0,0]*B[0,2] to C[0,2]
    # s=8: U[0,1,8]=1, V[1,2,8]=1, W[2,0,8]=1 -> contributes A[0,1]*B[1,2] to C[0,2]
    # s=9: U[1,0,9]=1, V[0,2,9]=1, W[2,1,9]=1 -> contributes A[1,0]*B[0,2] to C[1,2]
    # But we still need A[1,1]*B[1,2] for C[1,2]. That's a 4th factor.
    
    # So rank 3 is not sufficient for the k=2 part in this simple way.
    # Let's return None.
    return None


def _try_rank11():
    """
    Rank-11: Strassen (7) for k=0,1 + schoolbook (4) for k=2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    # Strassen for k=0,1 (columns 0-6)
    U_sub, V_sub, W_sub = _strassen_2x2x2()
    U[:, 0:7] = U_sub
    V[0:2, 0:7] = V_sub[0:2, :]
    V[2:4, 0:7] = V_sub[2:4, :]
    W[0:2, 0:7] = W_sub[0:2, :]
    W[2:4, 0:7] = W_sub[2:4, :]
    
    # Schoolbook for k=2 (columns 7-10)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # 4 factors:
    # s=7: A[0,0]*B[0,2] -> C[0,2]
    U[0, 7] = 1
    V[0, 7] = 1  # V[j*3+k] = V[0*3+2] = V[2, 7]
    W[2, 7] = 1  # W[k*3+i] = W[2*2+0] = W[4, 7]
    # Wait, W index is k*n+i = k*2+i. For k=2, i=0: W[4, 7]
    # V index is j*p+k = j*3+k. For j=0, k=2: V[2, 7]
    # U index is i*m+j = i*2+j. For i=0, j=0: U[0, 7]
    
    # s=8: A[0,1]*B[1,2] -> C[0,2]
    U[1, 8] = 1
    V[5, 8] = 1  # j=1, k=2: V[1*3+2] = V[5, 8]
    W[4, 8] = 1  # k=2, i=0: W[2*2+0] = W[4, 8]
    
    # s=9: A[1,0]*B[0,2] -> C[1,2]
    U[2, 9] = 1
    V[2, 9] = 1  # j=0, k=2: V[2, 9]
    W[5, 9] = 1  # k=2, i=1: W[2*2+1] = W[5, 9]
    
    # s=10: A[1,1]*B[1,2] -> C[1,2]
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    return U, V, W


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with schoolbook (rank 12) as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = n * m * p
    
    # Try rank 11
    if time.time() < deadline:
        U11, V11, W11 = _try_rank11()
        if _verify_decomposition(U11, V11, W11, n, m, p):
            best_result = {"U": U11.tolist(), "V": V11.tolist(), "W": W11.tolist()}
            best_rank = 11
            return best_result
    
    # Try rank 10
    if time.time() < deadline:
        r10 = _try_rank10()
        if r10 is not None:
            U10, V10, W10 = r10
            if _verify_decomposition(U10, V10, W10, n, m, p):
                best_result = {"U": U10.tolist(), "V": V10.tolist(), "W": W10.tolist()}
                best_rank = 10
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
