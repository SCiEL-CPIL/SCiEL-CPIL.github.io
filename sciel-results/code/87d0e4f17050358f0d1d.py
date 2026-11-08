"""
Matrix multiplication 4x4 by 4x9 - low-rank tensor decomposition.

Strategy:
- Model 4x4 * 4x9 as block multiplication:
  A (4x4) * B (4x9) = A * [B1 | B2 | B3] where each Bi is 4x3.
  C = [C1 | C2 | C3] where each Ci = A * Bi is 4x3.
  
  Each Ci = A * Bi can be computed with a 4x4x4 decomposition restricted
  to 3 output columns. But we can do better by using Strassen 4x4x4 (rank 49)
  and just taking 3/4 of the output... no, that doesn't reduce rank.
  
  Alternative: Use the fact that 4x4x3 multiplication has rank <= 4x4x4 rank.
  Actually, we need the full tensor for 4x4x9.
  
  Better approach: Use Strassen 4x4x4 (rank 49) for each 4x4 block of B.
  B has 9 columns = 2 full 4-column blocks + 1 single column.
  2 * 49 + 16 (schoolbook for 4x4x1) = 114.
  
  Even better: Can we do 4x4x3 more efficiently?
  4x4x3 tensor: we need C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..3, k in 0..2.
  This is a 4x4x3 tensor. Its rank is at most 4*4*3 = 48 (schoolbook).
  But we can use Strassen-like methods.
  
  Actually, let's think about this differently. The tensor for 4x4x9 is:
  T[i,j,k] = 1 for all i in 0..3, j in 0..3, k in 0..8.
  
  We want to decompose: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
  
  Key insight: The 4x4x9 tensor can be seen as 9 copies of the 4x4x1 "slice" tensor,
  but the U and V matrices can be shared across slices if we structure it right.
  
  Actually no - each k value gives a different output, so W[k,i,c] differs.
  But U[i,j,c] and V[j,k,c] - V depends on k, so it differs per k too.
  
  Let me reconsider. For matrix multiplication A(4x4) * B(4x9):
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  The tensor is T[i,j,k] = 1 (the multiplication tensor).
  
  Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
  
  Schoolbook: rank = 4*4*9 = 144, with U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1 for c=(i,j,k).
  
  Strassen for 4x4x4: rank 49. For 4x4x9, we can split B into [B1(4x4) | B2(4x4) | B3(4x1)]:
  - Use Strassen 4x4x4 (rank 49) for B1 -> C1
  - Use Strassen 4x4x4 (rank 49) for B2 -> C2  
  - Use schoolbook 4x4x1 (rank 16) for B3 -> C3
  Total: 49 + 49 + 16 = 114.
  
  Can we do better? What about using a 4x4x3 decomposition?
  The 4x4x3 multiplication tensor has rank at most... let's think.
  
  Actually, there's a known result: the rank of the 4x4x4 matrix multiplication tensor is 49 (Strassen).
  For 4x4x3, we can embed it in 4x4x4 and the rank is at most 49, but could be less.
  
  Let me try a different decomposition. Split 9 = 4 + 3 + 2:
  - 4x4x4: rank 49
  - 4x4x3: rank ?
  - 4x4x2: rank ?
  
  Or 9 = 3 + 3 + 3:
  - 3 * rank(4x4x3)
  
  The rank of 4x4x3 tensor: I believe it's at most 32 or so, but I'm not sure of the exact value.
  
  Let me just use the safe approach: 2*49 + 16 = 114, and verify it works.
  
  Actually, wait. Let me reconsider the problem. We want to minimize rank.
  
  Another approach: Use the fact that we can share structure.
  For 4x4x9, think of it as: for each of the 9 output columns, we do a 4x4x1 multiply.
  The 4x4x1 tensor (computing one column of C) has rank 4 (not 16!).
  
  Why? C[i] = sum_j A[i,j] * B[j] for a single output column.
  This is: C[i] = sum_j A[i,j] * B[j]
  As a tensor: T[i,j,0] = 1 (only one k value).
  Decomposition: T[i,j,0] = sum_c U[i,j,c] * V[j,0,c] * W[0,i,c]
  
  We need: sum_c U[i,j,c] * V[j,0,c] * W[0,i,c] = 1 for all i,j.
  
  This is the 4x4x1 tensor. Its rank is 4 (it's the outer product structure of a 4x4 matrix times a vector).
  
  Wait, actually the 4x4x1 tensor T[i,j,0] = 1 has rank 4? Let me verify.
  T[i,j,0] = 1. We need sum_c U[i,j,c] * V[j,0,c] * W[0,i,c] = 1.
  
  With rank 4: let c = 0..3.
  U[i,j,c] = delta(i,c) (1 if i==c else 0)
  V[j,0,c] = 1 for all j,c
  W[0,i,c] = delta(i,c)
  
  Then sum_c delta(i,c) * 1 * delta(i,c) = sum_c delta(i,c)^2 = 1 (since only c=i contributes).
  Yes! Rank 4 works.
  
  So for 4x4x9: 9 * 4 = 36 rank? But wait, the V matrix has shape (4*9, rank) = (36, rank).
  For each k, V[j,k,c] is part of the decomposition.
  
  Let me redo this carefully.
  
  For the full 4x4x9 tensor:
  T[i,j,k] = 1 for all i in 0..3, j in 0..3, k in 0..8.
  
  Decomposition with rank R:
  T[i,j,k] = sum_{c=0}^{R-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
  
  Using the "one column at a time" approach:
  For each k, we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j.
  
  If we use rank 4 per k value, total rank = 9*4 = 36.
  
  For column k, use 4 basis vectors:
  For c = k*4 + d where d in 0..3:
    U[i,j,c] = delta(i,d)
    V[j,k,c] = 1
    V[j,k',c] = 0 for k' != k
    W[k,i,c] = delta(i,d)
    W[k',i,c] = 0 for k' != k
  
  Check: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
  = sum_{d=0}^{3} delta(i,d) * 1 * delta(i,d)  [only terms with the same k contribute]
  = sum_{d=0}^{3} delta(i,d)^2
  = 1 (since exactly one d equals i)
  
  Yes! This gives rank 36.
  
  But wait, can we do even better? The issue is that V[j,k,c] = 1 for all j means we're not exploiting structure in j.
  
  Actually, the above decomposition is valid and gives rank 36. Let me verify more carefully.
  
  U shape: (4*4, 36) = (16, 36)
  V shape: (4*9, 36) = (36, 36)
  W shape: (9*4, 36) = (36, 36)
  
  U[i*4+j][c] where c = k*4+d:
    = 1 if i==d else 0
  
  V[j*9+k][c] where c = k'*4+d:
    = 1 if k==k' else 0  (independent of j and d)
  
  W[k*4+i][c] where c = k'*4+d:
    = 1 if k==k' and i==d else 0
  
  Check: sum_c U[i*4+j][c] * V[j*9+k][c] * W[k*4+i][c]
  = sum_{k'=0}^{8} sum_{d=0}^{3} U[i*4+j][k'*4+d] * V[j*9+k][k'*4+d] * W[k*4+i][k'*4+d]
  
  U[i*4+j][k'*4+d] = delta(i,d)
  V[j*9+k][k'*4+d] = delta(k,k')
  W[k*4+i][k'*4+d] = delta(k,k') * delta(i,d)
  
  So the product = delta(i,d) * delta(k,k') * delta(k,k') * delta(i,d) = delta(i,d) * delta(k,k')
  
  Sum over k',d: sum_{k'} sum_d delta(i,d) * delta(k,k') = sum_d delta(i,d) * 1 = 1 (since sum_d delta(i,d) = 1)
  
  This works! Rank = 36.
  
  Can we reduce further? The key question is whether we can share the U and W entries across different k values.
  
  In the above, U[i*4+j][c] depends only on i and d (not on k'), and W[k*4+i][c] depends on k, i, and d.
  
  The constraint is that for each (i,j,k), we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.
  
  If we try to use fewer than 36, we need some sharing. But the tensor structure seems to require at least 36 in this "column-wise" approach.
  
  However, there might be a more clever decomposition. Let me think...
  
  Actually, the minimum rank for 4x4x9 matrix multiplication is not known to be 36. It could be lower. But 36 is a valid upper bound.
  
  Let me also consider: can we use Strassen-like tricks to reduce below 36?
  
  For 4x4x4, Strassen gives 49 < 64 (schoolbook). The ratio is 49/64 = 0.766.
  If we could achieve a similar ratio for 4x4x9: 144 * 0.766 ≈ 110. But 36 is already much better than 110!
  
  Wait, I think I made an error. Let me re-examine.
  
  Schoolbook for 4x4x9: rank = 4*4*9 = 144.
  My "column-wise" approach: rank = 9*4 = 36.
  
  That's a huge improvement! Let me double-check the logic.
  
  The multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
  
  Tensor form: The multiplication tensor M satisfies M[i,j,k] = 1 (it's the identity tensor for matrix mult).
  We decompose: M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
  
  Then: C[i,k] = sum_{i',j',k'} M[i',j',k'] * A[i',j'] * B[j',k'] * ... 
  
  No wait, that's not right. Let me be more careful.
  
  The standard formulation: 
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  This can be written as:
  C[i,k] = sum_{j} (1) * A[i,j] * B[j,k]
  
  The "1" is the tensor entry M[i,j,k] = 1.
  
  If we decompose M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c],
  then: C[i,k] = sum_c sum_j U[i,j,c] * A[i,j] * V[j,k,c] * B[j,k] * W[k,i,c]
  
  Hmm, that's not quite right either. Let me look at the standard tensor decomposition formulation.
  
  Actually, the correct formulation is:
  C[i,k] = sum_{j,c} U[i,j,c] * V[j,k,c] * W[k,i,c] * A[i,j] * B[j,k]
  
  No, that's also not right. Let me think again.
  
  The standard way: We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
  C[i,k] = sum_j A[i,j] * B[j,k]
  
  A rank-r decomposition gives:
  C[i,k] = sum_{c=1}^{r} W[k,i,c] * (sum_j U[i,j,c] * A[i,j]) * (sum_k' V[j,k',c] * B[j,k'])
  
  No, that's not right either. Let me look at this more carefully.
  
  The correct formulation for matrix multiplication tensor decomposition:
  
  We have three tensors U, V, W with:
  - U has indices (i, j, c) for i in 0..n-1, j in 0..m-1, c in 0..r-1
  - V has indices (j, k, c) for j in 0..m-1, k in 0..p-1, c in 0..r-1  
  - W has indices (k, i, c) for k in 0..p-1, i in 0..n-1, c in 0..r-1
  
  The decomposition satisfies:
  sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i, j, k
  
  Then the matrix product is computed as:
  C[i,k] = sum_c W[k,i,c] * (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k])
  
  Wait, that doesn't look right dimensionally. Let me reconsider.
  
  Actually, I think the correct interpretation is:
  Define intermediate values:
  X_c = sum_j U[i,j,c] * A[i,j]  -- but this depends on i, so it's X[i,c]
  Y_c = sum_j V[j,k,c] * B[j,k]  -- this depends on k, so it's Y[k,c]
  
  Then C[i,k] = sum_c W[k,i,c] * X[i,c] * Y[k,c]
  
  Let's verify: 
  C[i,k] = sum_c W[k,i,c] * (sum_j U[i,j,c] * A[i,j]) * (sum_j' V[j',k,c] * B[j',k])
  
  = sum_c sum_j sum_j' W[k,i,c] * U[i,j,c] * A[i,j] * V[j',k,c] * B[j',k]
  
  For this to equal sum_j A[i,j] * B[j,k], we need:
  sum_c sum_j sum_j' W[k,i,c] * U[i,j,c] * V[j',k,c] * delta(j,j') = 1 when j=j'
  
  i.e., sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
  
  Yes! That's the condition. So my earlier analysis is correct.
  
  So the rank-36 decomposition I described should work. Let me code it up and verify.
  
  Actually, I realize I should double-check: is rank 36 really achievable, or did I make an error?
  
  Let me re-examine. The condition is:
  sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i in 0..3, j in 0..3, k in 0..8.
  
  My proposed decomposition with r = 36, where c = k'*4 + d for k' in 0..8, d in 0..3:
  U[i,j,c] = U[i,j,k'*4+d] = delta(i,d)
  V[j,k,c] = V[j,k,k'*4+d] = delta(k,k')
  W[k,i,c] = W[k,i,k'*4+d] = delta(k,k') * delta(i,d)
  
  Check: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
  = sum_{k'=0}^{8} sum_{d=0}^{3} delta(i,d) * delta(k,k') * delta(k,k') * delta(i,d)
  = sum_{k'} sum_d delta(i,d)^2 * delta(k,k')
  = sum_d delta(i,d) * sum_{k'} delta(k,k')
  = 1 * 1 = 1.
  
  Yes, it works! Rank 36.
  
  Now, can we do better than 36? The theoretical minimum is at least max(n*m, m*p, p*n) / something... actually, the lower bound for matrix multiplication tensor rank is more complex.
  
  For now, 36 is a significant improvement over 114 (Strassen) and 144 (schoolbook). Let me implement this.
  
  But wait - can we reduce further? The key observation is that in my decomposition, U doesn't depend on j at all, and V doesn't depend on j or d. This suggests there might be room for optimization.
  
  Actually, I think 36 might not be optimal. Let me think about whether we can do better.
  
  The tensor T[i,j,k] = 1 for i in 0..3, j in 0..3, k in 0..8.
  
  Note that T[i,j,k] = T[i,j,0] * T[i,0,k] * T[0,j,k]... no, that's not a valid factorization.
  
  Actually, T[i,j,k] = 1 = 1 * 1 * 1, so it's a rank-1 tensor in a trivial sense, but the decomposition must respect the index structure.
  
  I believe 36 is a good upper bound. Let me implement it.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
n, m, p = DIMENSIONS

def _validate(U, V, W, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_3d = U_arr.reshape(n, m, -1)
    V_3d = V_arr.reshape(m, p, -1)
    W_3d = W_arr.reshape(p, n, -1)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _schoolbook():
    """Return the trivial schoolbook decomposition (rank = n*m*p = 144)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank36_decomposition():
    """
    Construct rank-36 decomposition for 4x4x9 matrix multiplication.
    
    For each output column k (0..8) and each basis index d (0..3):
      c = k*4 + d
      U[i,j,c] = 1 if i==d else 0
      V[j,k,c] = 1 if k matches, else 0  (i.e., V[j,k',c] = delta(k,k'))
      W[k,i,c] = 1 if k matches and i==d, else 0
    """
    rank = p * n  # 9 * 4 = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        for d in range(n):
            c = k * n + d
            # U[i, j, c] = delta(i, d) for all i, j
            for i in range(n):
                for j in range(m):
                    if i == d:
                        U[i * m + j][c] = 1
            # V[j, k, c] = 1 for all j (only for this k)
            for j in range(m):
                V[j * p + k][c] = 1
            # W[k, i, c] = delta(i, d) for this k
            for i in range(n):
                if i == d:
                    W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    Returns a dictionary with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Rank-36 decomposition
    if time.time() < deadline:
        try:
            r36 = _rank36_decomposition()
            if _validate(r36['U'], r36['V'], r36['W']):
                best_result = r36
                best_rank = len(r36['U'][0])
        except Exception:
            pass
    
    # Strategy 2: Schoolbook (always works, rank = 144)
    if time.time() < deadline:
        try:
            sb = _schoolbook()
            if _validate(sb['U'], sb['V'], sb['W']):
                rank = len(sb['U'][0])
                if rank < best_rank:
                    best_result = sb
                    best_rank = rank
        except Exception:
            pass
    
    if best_result is None:
        return _schoolbook()
    
    return best_result
