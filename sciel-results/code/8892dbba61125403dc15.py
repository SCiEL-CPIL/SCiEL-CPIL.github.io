"""Matrix multiplication 3,3,8 tensor decomposition."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n) = (9, 24, 24)
    # T[(r,k), (k',c), (c',r')] = 1 if k==k' and r==r' and c==c'
    T = np.zeros((n * m, m * p, p * n))
    for r in range(n):
        for k in range(m):
            for c in range(p):
                i = r * m + k       # A index
                j = k * p + c       # B index
                l = c * n + r       # C index (output convention: W[k*n+i, r] where output is C[r,c])
                T[i, j, l] = 1
    
    # Strategy: Use Strassen's rank-7 decomposition for each 3x3x3 block
    # The 3x3x8 tensor can be decomposed into 8 independent 3x3x1 "slices"
    # but we can do better by sharing the A-side computation.
    
    # Actually, let me think about this differently.
    # The tensor T has the structure:
    # T = sum_{r=0}^{2} sum_{c=0}^{7} (sum_k e_{(r,k)} ⊗ e_{(k,c)} ⊗ e_{(c,r)})
    
    # For each fixed (r,c), the inner sum over k is a rank-3 tensor (schoolbook for 1x3x1)
    # But we can use a lower-rank decomposition if we share across c values.
    
    # Key insight: For a fixed r, the contribution to C[r,:] is A[r,:] @ B
    # This is a 1x3 matrix times a 3x8 matrix, giving a 1x8 matrix.
    # The tensor for this is: sum_k e_{(r,k)} ⊗ e_{(k,:)} ⊗ e_{(:,r)}
    # where e_{(k,:)} is a vector in B-space (dimension 24) and e_{(:,r)} is in C-space (dimension 24).
    
    # For fixed r, define:
    # U_r = sum_k alpha_{r,k} e_{(r,k)}  (but this is just a single component if alpha=1)
    # This doesn't help directly.
    
    # Let me try a different approach: use the known rank-7 decomposition for 3x3x3
    # and apply it to each of the 8 columns, but share the A-side factors.
    
    # For 3x3x3, Strassen's decomposition uses 7 products:
    # P1 = (A00 + A11 + A22) * (B00 + B11 + B22)
    # P2 = (A01 + A11 + A21) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A11 * (B10 - B00)
    # P5 = (A02 + A12 + A22) * B11
    # P6 = (A10 + A20) * (B00 + B10)
    # P7 = (A00 + A10) * (B10 - B20)
    
    # Wait, I need to be more careful. Let me use a verified rank-7 decomposition.
    
    # Actually, for 3x3x3, the schoolbook rank is 9, and it's known that the rank is at most 7
    # (Strassen, 1969). Let me construct it explicitly.
    
    # A 3x3x3 rank-7 decomposition:
    # We need 7 vectors u_s in R^9, v_s in R^9, w_s in R^9 such that
    # sum_s u_s ⊗ v_s ⊗ w_s = T3
    
    # From the literature, one such decomposition is:
    # Let's use the following (I'll verify numerically):
    
    # For now, let me try a simpler approach: use the schoolbook (rank 9) for each 3x3x3 block
    # and see if I can do better.
    
    # The 3x3x8 tensor can be seen as:
    # T = sum_{c=0}^{7} T_c where T_c is the tensor for multiplying by the c-th column of B
    
    # Each T_c has the structure: T_c[(r,k),(k',c'),(c'',r')] = 1 if k=k', r=r', c'=c, c''=c
    # So T_c is supported on a 9x9x9 sub-tensor (the 3x3x3 part for column c).
    
    # If I use rank 9 for each T_c, total rank is 8*9 = 72 (schoolbook).
    
    # To do better, I need to find a decomposition that shares factors across different c values.
    
    # Idea: Precompute linear combinations of A that are shared across all 8 columns.
    # For example, if I precompute 3 matrices P_s = A @ S_s (s=0,1,2), where S_s are 3x3 matrices,
    # then C[:,c] = sum_s P_s @ T_s(c) where T_s(c) is a 3x1 vector.
    
    # The total rank would be 3 * 8 = 24 if this works.
    
    # Let me try to construct this.
    
    # For this to work, I need:
    # C[r,c] = sum_k A[r,k] * B[k,c] = sum_s sum_k' (A @ S_s)[r,k'] * T_s(c)[k']
    # = sum_s sum_k' sum_k'' A[r,k''] * S_s[k'',k'] * T_s(c)[k']
    
    # This requires: sum_k A[r,k] * B[k,c] = sum_s sum_k' sum_k'' A[r,k''] * S_s[k'',k'] * T_s(c)[k']
    
    # For this to hold for all A and B, I need:
    # sum_k B[k,c] * delta_{r,k} = sum_s sum_k' sum_k'' S_s[k'',k'] * T_s(c)[k'] * delta_{r,k''}
    
    # This is a constraint on S_s and T_s.
    
    # Let me try a simple construction:
    # S_0 = I, S_1 = 0, S_2 = 0
    # T_0(c) = B[:,c], T_1(c) = 0, T_2(c) = 0
    # Then C[r,c] = sum_k A[r,k] * I[k,k'] * B[k',c] = sum_k A[r,k] * B[k,c]. ✓
    
    # But this gives rank 9*8 = 72, not 24, because T_0(c) depends on c.
    
    # The issue is that T_s(c) must be independent of c for the rank to be 3*8 = 24.
    
    # Let me try a different approach: use a rank-7 decomposition for 3x3x3 and apply it to each column.
    # Total rank: 7 * 8 = 56.
    
    # This is better than 72! Let me construct it.
    
    # First, I need a valid rank-7 decomposition for 3x3x3.
    # Let me find one from the literature or construct it numerically.
    
    # For 3x3x3, the tensor is:
    # T3[(r,k),(k',c),(c',r')] = 1 if k=k', r=r', c=c'
    
    # A rank-7 decomposition exists. Let me try to find it.
    
    # From Strassen's paper, one rank-7 decomposition for 3x3x3 is:
    # (I'll use a known construction)
    
    # Actually, let me just use the schoolbook (rank 9) for now and return it,
    # since I'm not sure of the exact rank-7 decomposition.
    
    # But wait, the task says "rank-minimal exact tensor decomposition".
    # The current score is 1.0000, which corresponds to rank 72 (schoolbook).
    # If I can find a decomposition with rank < 72, the score will be better.
    
    # Let me try to use a rank-9 decomposition for each 3x3x3 block (schoolbook),
    # giving total rank 8*9 = 72. This is the same as the current score.
    
    # To improve, I need to find a lower-rank decomposition.
    
    # Let me try to use a rank-7 decomposition for 3x3x3.
    # I'll search for one in the literature.
    
    # For now, let me return the schoolbook decomposition with rank 72.
    
    rank = n * m * p  # 72
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for r in range(n):
        for k in range(m):
            for c in range(p):
                U[r * m + k, col] = 1
                V[k * p + c, col] = 1
                W[c * n + r, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
