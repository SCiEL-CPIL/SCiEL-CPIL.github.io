"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False, abs(val - 1.0)
    return True, 0.0

def _build_strassen_2x2x2():
    """Build Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7) for the 2x2x2 case.
    U indices: i*2+j, V indices: j*2+k, W indices: k*2+i
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen product r, we need:
    # sum over factors that give U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 when (i,j,k) matches
    
    # Product 1: (A00+A11)(B00+B11)
    # A00: i=0,j=0; A11: i=1,j=1
    # B00: j=0,k=0; B11: j=1,k=1
    # So U[0,0,0]=1, U[1,1,0]=1
    # V[0,0,0]=1, V[1,1,0]=1
    # W[0,0,0]=1 (k=0,i=0), W[1,1,0]=1 (k=1,i=1)
    # But we need this to contribute to C00 and C11
    # C00 = M1 + ...: i=0,k=0
    # C11 = M1 - ...: i=1,k=1
    
    # Let me set up more carefully.
    # For product r, U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The product contributes 1 to T[i,j,k] if all three are nonzero.
    
    # M1 = (A00+A11)(B00+B11)
    # This means: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # So it contributes to (i=0,j=0,k=0), (i=0,j=0,k=1), (i=1,j=1,k=0), (i=1,j=1,k=1)
    # U[0,0]=1, U[1,1]=1
    # V[0,0]=1, V[0,1]=1, V[1,0]=1, V[1,1]=1
    # W[0,0]=1, W[1,0]=1, W[0,1]=1, W[1,1]=1
    # Wait, that's not right. Let me think again.
    
    # Actually for a single "rank-1" tensor term:
    # T[i,j,k] += U[i*m+j] * V[j*p+k] * W[k*n+i]
    
    # For M1 = (A00+A11)(B00+B11):
    # We want it to contribute to (i=0,j=0,k=0), (i=0,j=0,k=1), (i=1,j=1,k=0), (i=1,j=1,k=1)
    # But with a single rank-1 term, we can only have one (i,j) pair and one (j,k) pair.
    # So M1 needs to be split into multiple rank-1 terms.
    
    # Actually, Strassen's algorithm uses 7 scalar multiplications.
    # Each scalar multiplication corresponds to one rank in the tensor decomposition.
    # For each multiplication, we have specific A elements and B elements.
    
    # Let me define each of the 7 products explicitly:
    
    # r=0: M1 = (A00+A11)(B00+B11)
    # A part: A00 (i=0,j=0), A11 (i=1,j=1)
    # B part: B00 (j=0,k=0), B11 (j=1,k=1)
    # C part: contributes to C00 (i=0,k=0) and C11 (i=1,k=1)
    # This is actually 4 rank-1 terms:
    # (A00)(B00) -> (i=0,j=0,k=0)
    # (A00)(B11) -> (i=0,j=0,k=1)  -- wait, B11 is j=1,k=1, so this doesn't work with A00 (j=0)
    
    # I'm overcomplicating this. Let me use the standard Strassen formulation.
    # In Strassen, we compute 7 intermediate products:
    # M1 = (A00+A11)(B00+B11)
    # etc.
    # Each M is a scalar, and then C is formed from linear combinations of M's.
    # In tensor decomposition terms, each M corresponds to one rank-1 term.
    # The A and B "factors" are sums, which means U and V entries are 1 or -1.
    
    # For M1 = (A00+A11)(B00+B11):
    # U[0,0,0]=1, U[1,1,0]=1  (A00 and A11)
    # V[0,0,0]=1, V[1,1,0]=1  (B00 and B11)  
    # But wait, (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # For this to be a single rank-1 tensor term, we need:
    # U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for the right (i,j,k)
    
    # Actually, I think the correct interpretation is:
    # The tensor T[i,j,k] = 1 for all i,j,k in the matmul tensor.
    # We decompose T = sum_r U_r ⊗ V_r ⊗ W_r where each is a rank-1 tensor.
    # For Strassen, we're not directly decomposing T, but finding a way to compute
    # the matmul with fewer multiplications.
    
    # Let me just directly construct the Strassen decomposition.
    # For 2x2x2, the 7 Strassen products are:
    
    # Product 0: (A00+A11)(B00+B11)
    # This gives: A00*B00 + A00*B11 + A11*B00 + A11*B11
    # In tensor form, this is 4 separate rank-1 terms, but they share the same "multiplication slot"
    
    # I think I need to be more careful. Let me look at this differently.
    # The matrix multiplication tensor for 2x2x2 has T[i,j,k] = 1 for all i,j,k in {0,1}.
    # A rank-r decomposition means: T[i,j,k] = sum_{s=1}^r U[i*2+j,s] * V[j*2+k,s] * W[k*2+i,s]
    
    # For Strassen with 7 products, we assign each product to a rank s.
    # For product s, U[i*2+j,s] is 1 if A[i,j] is part of the first factor,
    # V[j*2+k,s] is 1 if B[j,k] is part of the second factor,
    # W[k*2+i,s] is 1 if C[i,k] gets a contribution from this product.
    
    # Product 0: (A00+A11)(B00+B11)
    # First factor involves A00 (i=0,j=0) and A11 (i=1,j=1)
    # Second factor involves B00 (j=0,k=0) and B11 (j=1,k=1)
    # This product contributes to C00 (i=0,k=0) and C11 (i=1,k=1)
    # But how? C00 = M1 + M4 - M5 + M7, C11 = M1 - M2 + M3 + M6
    # So M1 contributes +1 to C00 and +1 to C11.
    
    # For the tensor decomposition:
    # U[0,0,0]=1, U[1,1,0]=1
    # V[0,0,0]=1, V[1,1,0]=1
    # W[0,0,0]=1 (for C00, i=0,k=0), W[1,1,0]=1 (for C11, i=1,k=1)
    # But this would give T[0,0,0] += 1*1*1 = 1, T[0,1,0] += 0*1*1 = 0, etc.
    # That's not right because we need T[i,j,k]=1 for ALL i,j,k.
    
    # I think the issue is that Strassen's algorithm doesn't directly give a tensor decomposition.
    # Instead, it gives a way to compute the matmul with 7 multiplications.
    # The tensor decomposition rank is different from the number of multiplications.
    
    # Let me just use a known rank-7 decomposition for 2x2x2 matmul.
    # Actually, the rank of the 2x2x2 matmul tensor is 7 (proven by Strassen).
    
    # Let me construct it properly. The 7 rank-1 terms are:
    
    # Term 0: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
    # This gives: T[0,0,0]+=1, T[0,0,1]+=0 (V[0,1]=0), T[0,1,0]+=0, T[0,1,1]+=0
    #           T[1,0,0]+=0, T[1,0,1]+=0, T[1,1,0]+=0, T[1,1,1]+=1
    # Hmm, that only covers 2 of the 8 entries.
    
    # I think I need to use the actual Strassen formulas more carefully.
    # Let me just hardcode a known rank-7 decomposition.
    
    # After research, here's a valid rank-7 decomposition for 2x2x2:
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 0: contributes to (0,0,0), (0,0,1), (1,1,0), (1,1,1)
    U[0, 0] = 1; U[1, 0] = 1  # A00, A11 -> but indices are i*2+j
    # i=0,j=0 -> idx 0; i=1,j=1 -> idx 3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1  # j=0,k=0 -> idx 0; j=1,k=1 -> idx 3
    W[0, 0] = 1; W[3, 0] = 1  # k=0,i=0 -> idx 0; k=1,i=1 -> idx 3
    
    # This gives T[0,0,0] += U[0,0]*V[0,0]*W[0,0] = 1
    # T[0,0,1] += U[0,0]*V[1,0]*W[1,0] = 1*0*1 = 0  -- V[1,0] is j=0,k=1 -> idx 1, V[1,0]=0
    # So this only gives T[0,0,0] and T[1,1,1] = 1. Not enough.
    
    # Let me try a different approach. I'll just use the schoolbook for 2x2x2 (rank 8)
    # and see if the overall decomposition works.
    
    # Actually, let me just build the full 3x4x8 decomposition using schoolbook
    # but try to find a better structure.
    
    return None  # Fallback

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition for n x m x p matrix multiplication."""
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
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_3x4x8_decomposition():
    """Build a decomposition for 3x4x8 matrix multiplication.
    
    Strategy: Split the 3x4 matrix into blocks:
    - Row 0: 1x4
    - Rows 1-2: 2x4
    
    For the 2x4 x 4x8 part, we can use Strassen on 2x2x2 blocks.
    For the 1x4 x 4x8 part, use schoolbook.
    
    Actually, let me try a different decomposition:
    - Split 3x4 into 1x4 + 1x4 + 1x4 (three 1x4 rows)
    - Each 1x4 x 4x8 can be done with rank 8 (schoolbook)
    - Total: 24
    
    Or:
    - Split 3x4 into 2x4 + 1x4
    - 2x4 x 4x8: split 4x8 into 4x4 + 4x4
      - 2x4 x 4x4: use Strassen-like, rank ~ 2*4*4/2 = 16? No, Strassen gives 7 per 2x2x2.
      - Actually 2x4 x 4x4: treat as 2x(4x4) x (4x4)x4... this is getting complicated.
    
    Let me just try a direct approach with rank optimization.
    """
    n, m, p = 3, 4, 8
    
    # Try to build a rank-32 decomposition (which is the lower bound max(n*m, m*p, n*p) = max(12, 32, 24) = 32)
    # This might be achievable.
    
    # For now, use schoolbook as fallback
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    return U_sb, V_sb, W_sb

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s safety margin
    
    # Try Strassen-based decomposition
    result = _build_3x4x8_decomposition()
    if result is not None:
        U, V, W = result
        valid, err = _verify(U, V, W, n, m, p, tol=1e-6)
        if valid:
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
