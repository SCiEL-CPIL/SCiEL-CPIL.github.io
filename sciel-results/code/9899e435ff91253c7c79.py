"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from fractions import Fraction

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    The decomposition requires: for each rank index c,
    U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] sums to 1 for each (i,j,k)
    
    Actually the tensor for matrix multiplication M_{n,m,p} has entries:
    T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    # Build the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    u = U[i*m+j][c] if isinstance(U, list) else U[i*m+j, c]
                    v = V[j*p+k][c] if isinstance(V, list) else V[j*p+k, c]
                    w = W[k*n+i][c] if isinstance(W, list) else W[k*n+i, c]
                    s += u * v * w
                if s != 1:
                    return False
    return True

def _verify_with_numpy(n, m, p, U, V, W):
    """Faster verification using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j]
                v_val = V_arr[j*p + k]
                w_val = W_arr[k*n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    U_l = U.tolist()
    V_l = V.tolist()
    W_l = W.tolist()
    return U_l, V_l, W_l

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # The naive rank is n*m*p = 60
    # We know from the knowledge that rank 55 is achievable using Strassen on 2x2 blocks
    # Let's try to find rank 55 or lower.
    
    # Strategy: The matrix multiplication tensor for 2x3x10 can be decomposed
    # by splitting the 3 columns of A into a 2-column part and a 1-column part.
    # For the 2x2x10 part, use Strassen-like decomposition.
    # For the 2x1x10 part, use trivial multiplication.
    
    # Actually, let me think about this more carefully.
    # We're computing C = A * B where A is 2x3, B is 3x10, C is 2x10.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # The tensor T has indices (i,j,k) with i in {0,1}, j in {0,1,2}, k in {0,...,9}
    # T[i,j,k] = 1 for all valid (i,j,k)
    
    # Naive: rank = 2*3*10 = 60
    
    # Block approach: Split j into {0,1} and {2}
    # Part 1: i in {0,1}, j in {0,1}, k in {0,...,9} -> 2x2x10 tensor
    # Part 2: i in {0,1}, j in {2}, k in {0,...,9} -> 2x1x10 tensor
    
    # For 2x2x10: We can use Strassen on the 2x2 block for each pair of k values
    # Strassen gives rank 7 for 2x2x2 instead of 8
    # So for 2x2x10, we can group k into 5 pairs, each using Strassen: 5*7 = 35
    # Plus the 2x1x10 part: rank 2*1*10 = 20
    # Total: 35 + 20 = 55
    
    # Let's construct this decomposition.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*10 = 30 rows
    W = [[0] * rank for _ in range(p * n)]  # 10*2 = 20 rows
    
    col = 0
    
    # Part 1: 2x2x10 using Strassen on pairs of k
    # For each pair (k0, k1), Strassen computes:
    # C00 = A00*B00 - A01*(B10-B00)  -> but this is for 2x2*2x2=2x2
    # Actually Strassen for 2x2*2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # In our tensor notation for the 2x2x2 sub-block (rows i=0,1; cols j=0,1; output k=0,1):
    # We need to express T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1} as sum of 7 rank-1 terms.
    
    # The Strassen decomposition for 2x2 matrix multiplication:
    # Each M term corresponds to a rank-1 decomposition where:
    # U factors: coefficients on A entries
    # V factors: coefficients on B entries  
    # W factors: coefficients on C entries
    
    # Let me define the 7 Strassen terms for 2x2x2:
    # Term 0 (M1): U=[1,0,0,1] (A00+A11), V=[1,0,0,1] (B00+B11), W=[1,0,0,1] (C00+C11)
    # Wait, I need to be more careful with the indexing.
    
    # For the 2x2x2 sub-tensor with indices (i,j,k) where i,j,k in {0,1}:
    # T[i,j,k] = 1
    
    # Strassen rank-1 terms (U on A[i,j], V on B[j,k], W on C[i,k]):
    # Using the standard Strassen:
    # M1 = (A00+A11)*(B00+B11): U_A = [1,0,0,1], U_B = [1,0,0,1], U_C = [1,0,0,1]
    # M2 = (A10+A11)*B00: U_A = [0,0,1,1], U_B = [1,0,0,0], U_C = [0,0,1,0]
    # M3 = A00*(B01-B11): U_A = [1,0,0,0], U_B = [0,1,0,-1], U_C = [0,1,0,0]
    # M4 = A11*(B10-B00): U_A = [0,0,0,1], U_B = [0,0,1,-1], U_C = [0,0,0,1]
    # M5 = (A00+A01)*B11: U_A = [1,1,0,0], U_B = [0,0,0,1], U_C = [0,1,0,0]
    # M6 = (A10-A00)*(B00+B01): U_A = [0,0,1,-1], U_B = [1,1,0,0], U_C = [0,0,1,0]
    # M7 = (A01-A11)*(B10+B11): U_A = [0,1,0,-1], U_B = [0,0,1,1], U_C = [0,1,0,1]
    
    # Where A is indexed as [A00, A01, A10, A11] and similarly for B, C.
    
    # Let me verify: C00 should be M1 - M4 + M6
    # C00 from M1: 1, from M4: -1, from M6: 1 -> total 1. But we need C00 = A00*B00 + A01*B10
    # Hmm, the W factor gives the coefficient for each C entry.
    
    # Actually, let me reconsider. The tensor T[i,j,k]=1 means:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So the "output" side is C[i,k], and we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    # For the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}):
    # U is indexed by (i,j) -> 4 entries
    # V is indexed by (j,k) -> 4 entries
    # W is indexed by (k,i) -> 4 entries
    
    # Strassen terms for 2x2x2:
    # Each term c has U_c (4 values), V_c (4 values), W_c (4 values)
    # such that sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all (i,j,k)
    
    # Standard Strassen gives:
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # So the W factors (coefficients on C entries) are:
    # M1: W = [1, 0, 0, 1] (C00 + C11)
    # M2: W = [0, 0, 1, 0] (C10)
    # M3: W = [0, 1, 0, 0] (C01)
    # M4: W = [-1, 0, 1, 0] (-C00 + C10)
    # M5: W = [0, 1, 0, 0] (C01)
    # M6: W = [1, 0, 1, 0] (C00 + C10)
    # M7: W = [0, 0, 0, 1] (C11)
    
    # Wait, let me re-derive. C00 = M1 - M4 + M6, so:
    # M1 contributes +1 to C00
    # M4 contributes -1 to C00
    # M6 contributes +1 to C00
    
    # C01 = M3 + M5:
    # M3 contributes +1 to C01
    # M5 contributes +1 to C01
    
    # C10 = M2 + M4:
    # M2 contributes +1 to C10
    # M4 contributes +1 to C10
    
    # C11 = M1 - M3 + M7:
    # M1 contributes +1 to C11
    # M3 contributes -1 to C11
    # M7 contributes +1 to C11
    
    # So W factors:
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 0]
    # M3: [0, 1, -1, 0]
    # M4: [-1, 0, 1, 0]
    # M5: [0, 1, 0, 0]
    # M6: [1, 0, 1, 0]
    # M7: [0, 0, 0, 1]
    
    # U factors (coefficients on A[i,j], indexed as [A00, A01, A10, A11]):
    # M1: A00+A11 -> [1, 0, 0, 1]
    # M2: A10+A11 -> [0, 0, 1, 1]
    # M3: A00 -> [1, 0, 0, 0]
    # M4: A11 -> [0, 0, 0, 1]
    # M5: A00+A01 -> [1, 1, 0, 0]
    # M6: A10-A00 -> [0, 0, 1, -1]  # Wait, A10 is at index 2, A00 at index 0
    # M7: A01-A11 -> [0, 1, 0, -1]
    
    # V factors (coefficients on B[j,k], indexed as [B00, B01, B10, B11]):
    # M1: B00+B11 -> [1, 0, 0, 1]
    # M2: B00 -> [1, 0, 0, 0]
    # M3: B01-B11 -> [0, 1, 0, -1]
    # M4: B10-B00 -> [0, 0, 1, -1]  # Wait, B10 is at index 2, B00 at index 0
    # M5: B11 -> [0, 0, 0, 1]
    # M6: B00+B01 -> [1, 1, 0, 0]
    # M7: B10+B11 -> [0, 0, 1, 1]
    
    # Now let's verify: for (i,j,k) = (0,0,0), we need sum_c U_c[0,0] * V_c[0,0] * W_c[0,0] = 1
    # U[0,0] values: M1=1, M2=0, M3=1, M4=0, M5=1, M6=0, M7=0
    # V[0,0] values: M1=1, M2=1, M3=0, M4=0, M5=0, M6=1, M7=0
    # W[0,0] values: M1=1, M2=0, M3=0, M4=-1, M5=0, M6=1, M7=0
    # Sum: 1*1*1 + 0*1*0 + 1*0*0 + 0*0*(-1) + 1*0*0 + 0*1*1 + 0*0*0 = 1. Correct!
    
    # For (i,j,k) = (0,1,0): C00 = A00*B00 + A01*B10, so T[0,1,0] = 1
    # U[0,1] values: M1=0, M2=0, M3=0, M4=0, M5=1, M6=0, M7=1
    # V[1,0] values: M1=0, M2=0, M3=1, M4=1, M5=0, M6=0, M7=1
    # W[0,0] values: M1=1, M2=0, M3=0, M4=-1, M5=0, M6=1, M7=0
    # Sum: 0*0*1 + 0*0*0 + 0*1*0 + 0*1*(-1) + 1*0*0 + 0*0*1 + 1*1*0 = 0. Wrong!
    
    # I think I have an error. Let me reconsider the indexing.
    
    # The issue is that V is indexed by (j,k), so V[j*p+k] in the full problem.
    # For the 2x2x2 sub-block, V is indexed by (j,k) where j in {0,1}, k in {0,1}.
    # So V[0,0] = B00, V[0,1] = B01, V[1,0] = B10, V[1,1] = B11.
    
    # And U is indexed by (i,j), so U[0,0] = A00, U[0,1] = A01, U[1,0] = A10, U[1,1] = A11.
    
    # And W is indexed by (k,i), so W[0,0] = C00, W[0,1] = C10, W[1,0] = C01, W[1,1] = C11.
    
    # Wait, W[k*n+i]. For n=2, W[k*2+i]. So W[0,0]=C[0,0], W[0,1]=C[1,0], W[1,0]=C[0,1], W[1,1]=C[1,1].
    # Hmm, that's W indexed by (k,i) -> W[k*2+i].
    
    # So W[0,0] corresponds to k=0,i=0 -> C[0,0]
    # W[0,1] corresponds to k=0,i=1 -> C[1,0]
    # W[1,0] corresponds to k=1,i=0 -> C[0,1]
    # W[1,1] corresponds to k=1,i=1 -> C[1,1]
    
    # Let me redo the W factors with this indexing:
    # C00 is at W[0,0], C10 is at W[0,1], C01 is at W[1,0], C11 is at W[1,1]
    
    # C00 = M1 - M4 + M6: W[0,0] gets +1 from M1, -1 from M4, +1 from M6
    # C01 = M3 + M5: W[1,0] gets +1 from M3, +1 from M5
    # C10 = M2 + M4: W[0,1] gets +1 from M2, +1 from M4
    # C11 = M1 - M3 + M7: W[1,1] gets +1 from M1, -1 from M3, +1 from M7
    
    # W factors (indexed as [W[0,0], W[0,1], W[1,0], W[1,1]] = [C00, C10, C01, C11]):
    # M1: [1, 0, 0, 1]
    # M2: [0, 1, 0, 0]
    # M3: [0, 0, 1, -1]
    # M4: [-1, 1, 0, 0]
    # M5: [0, 0, 1, 0]
    # M6: [1, 0, 0, 0]
    # M7: [0, 0, 0, 1]
    
    # Let me verify (i,j,k) = (0,1,0): T[0,1,0] = 1
    # U[0,1] = A01: M1=0, M2=0, M3=0, M4=0, M5=1, M6=0, M7=1
    # V[1,0] = B10: M1=0, M2=0, M3=0, M4=1, M5=0, M6=0, M7=1
    # W[0,0] = C00: M1=1, M2=0, M3=0, M4=-1, M5=0, M6=1, M7=0
    # Sum: 0*0*1 + 0*0*0 + 0*0*0 + 0*1*(-1) + 1*0*0 + 0*0*1 + 1*1*0 = 0. Still wrong!
    
    # Hmm, I think the issue is that for T[0,1,0], we're computing the contribution to C[0,0] from A[0,1]*B[1,0].
    # So this should contribute to C[0,0], which is W[0,0].
    # But the Strassen formula says C00 = M1 - M4 + M6, and none of these involve A01*B10 directly.
    # M6 = (A10-A00)(B00+B01) - no A01 or B10.
    # 
    # I think I'm confusing myself. Let me start over with a cleaner approach.
    
    # The key insight: for the 2x2x2 tensor, Strassen's algorithm gives a rank-7 decomposition.
    # Let me look up the exact form.
    
    # Actually, let me just try to verify the known result: rank(2x2x2) = 7.
    # And use a known decomposition.
    
    # From literature, the Strassen decomposition for 2x2 matrix mult:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 - M4 + M6
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M3 + M7
    
    # Using 1-based indexing: a11=A[0,0], a12=A[0,1], a21=A[1,0], a22=A[1,1]
    # b11=B[0,0], b12=B[0,1], b21=B[1,0], b22=B[1,1]
    # c11=C[0,0], c12=C[0,1], c21=C[1,0], c22=C[1,1]
    
    # U factors (on A, indexed [A00, A01, A10, A11]):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 1]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [-1, 0, 1, 0]
    # M7: [0, 1, 0, -1]
    
    # V factors (on B, indexed [B00, B01, B10, B11]):
    # M1: [1, 0, 0, 1]
    # M2: [1, 0, 0, 0]
    # M3: [0, 1, 0, -1]
    # M4: [-1, 0, 1, 0]
    # M5: [0, 0, 0, 1]
    # M6: [1, 1, 0, 0]
    # M7: [0, 0, 1, 1]
    
    # W factors (on C, indexed [C00, C10, C01, C11] since W[k*n+i]):
    # c11 = C[0,0] = M1 - M4 + M6 -> W[0,0]: M1=1, M4=-1, M6=1
    # c12 = C[0,1] = M3 + M5 -> W[1,0]: M3=1, M5=1
    # c21 = C[1,0] = M2 + M4 -> W[0,1]: M2=1, M4=1
    # c22 = C[1,1] = M1 - M3 + M7 -> W[1,1]: M1=1, M3=-1, M7=1
    
    # W factors:
    # M1: [1, 0, 0, 1]
    # M2: [0, 1, 0, 0]
    # M3: [0, 0, 1, -1]
    # M4: [-1, 1, 0, 0]
    # M5: [0, 0, 1, 0]
    # M6: [1, 0, 0, 0]
    # M7: [0, 0, 0, 1]
    
    # Verify (i,j,k) = (0,0,0): T=1
    # U[0,0]=A00: M1=1, M2=0, M3=1, M4=0, M5=1, M6=-1, M7=0
    # V[0,0]=B00: M1=1, M2=1, M3=0, M4=-1, M5=0, M6=1, M7=0
    # W[0,0]=C00: M1=1, M2=0, M3=0, M4=-1, M5=0, M6=1, M7=0
    # Sum: 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*(-1) + 1*0*0 + (-1)*1*1 + 0*0*0 = 1 + 0 + 0 + 0 + 0 - 1 + 0 = 0. WRONG!
    
    # I'm making errors. Let me be very careful.
    
    # T[0,0,0] means i=0, j=0, k=0. This is the coefficient of A[0,0]*B[0,0] in C[0,0].
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # So T[0,0,0] = 1.
    
    # In the decomposition: sum_c U[0,0]_c * V[0,0]_c * W[0,0]_c = 1
    # where U is indexed by (i,j), V by (j,k), W by (k,i).
    
    # U[0,0]_c is the coefficient of A[0,0] in the c-th Strassen product.
    # V[0,0]_c is the coefficient of B[0,0] in the c-th Strassen product.
    # W[0,0]_c is the coefficient of C[0,0] in the c-th Strassen product.
    
    # From Strassen:
    # M1 = (A00+A11)(B00+B11): A00 coeff=1, B00 coeff=1, C00 coeff=1 (from c11=M1-...)
    # M2 = (A10+A11)B00: A00 coeff=0, B00 coeff=1, C00 coeff=0
    # M3 = A00(B01-B11): A00 coeff=1, B00 coeff=0, C00 coeff=0
    # M4 = A11(B10-B00): A00 coeff=0, B00 coeff=-1, C00 coeff=-1 (from c11=...-M4+...)
    # M5 = (A00+A01)B11: A00 coeff=1, B00 coeff=0, C00 coeff=0
    # M6 = (A10-A00)(B00+B01): A00 coeff=-1, B00 coeff=1, C00 coeff=1 (from c11=...+M6)
    # M7 = (A01-A11)(B10+B11): A00 coeff=0, B00 coeff=0, C00 coeff=0
    
    # Sum for (0,0,0): 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*(-1) + 1*0*0 + (-1)*1*1 + 0*0*0
    # = 1 + 0 + 0 + 0 + 0 - 1 + 0 = 0. Still wrong!
    
    # The problem is that M4 has B00 coeff = -1 (from B10-B00), and C00 coeff = -1.
    # So the product is 0 * (-1) * (-1) = 0. That's fine.
    # M6 has A00 coeff = -1 (from A10-A00), B00 coeff = 1, C00 coeff = 1.
    # Product: (-1)*1*1 = -1.
    # M1: 1*1*1 = 1.
    # Total: 1 - 1 = 0. But it should be 1!
    
    # I see the issue. The Strassen formula computes c11 = M1 - M4 + M6, but the individual
    # products M1, M4, M6 each contribute to multiple c entries. The W factor should capture
    # ALL contributions, not just to c11.
    
    # M1 = (A00+A11)(B00+B11) contributes to c11 AND c22.
    # c11 = M1 - M4 + M6
    # c22 = M1 - M3 + M7
    # So M1 contributes +1 to c11 and +1 to c22.
    # W[0,0] (c11) for M1 = 1
    # W[1,1] (c22) for M1 = 1
    
    # M4 = A11(B10-B00) contributes to c11 and c21.
    # c11 = ... - M4 ...
    # c21 = M2 + M4
    # So M4 contributes -1 to c11 and +1 to c21.
    # W[0,0] for M4 = -1
    # W[0,1] for M4 = 1
    
    # M6 = (A10-A00)(B00+B01) contributes to c11 and c21.
    # c11 = ... + M6
    # c21 = M2 + M4 (M6 not in c21!)
    
    # Wait, c21 = M2 + M4. M6 is not in c21. So M6 only contributes to c11.
    # W[0,0] for M6 = 1, W[0,1] for M6 = 0.
    
    # Let me redo:
    # M1: c11 += 1, c22 += 1. W = [1, 0, 0, 1]
    # M2: c21 += 1. W = [0, 1, 0, 0]
    # M3: c12 += 1, c22 -= 1. W = [0, 0, 1, -1]
    # M4: c11 -= 1, c21 += 1. W = [-1, 1, 0, 0]
    # M5: c12 += 1. W = [0, 0, 1, 0]
    # M6: c11 += 1. W = [1, 0, 0, 0]
    # M7: c22 += 1. W = [0, 0, 0, 1]
    
    # Now verify (0,0,0):
    # U[0,0]: M1=1, M2=0, M3=1, M4=0, M5=1, M6=-1, M7=0
    # V[0,0]: M1=1, M2=1, M3=0, M4=-1, M5=0, M6=1, M7=0
    # W[0,0]: M1=1, M2=0, M3=0, M4=-1, M5=0, M6=1, M7=0
    # Sum: 1*1*1 + 0*1*0 + 1*0*0 + 0*(-1)*(-1) + 1*0*0 + (-1)*1*1 + 0*0*0
    # = 1 + 0 + 0 + 0 + 0 - 1 + 0 = 0. STILL WRONG!
    
    # I think the fundamental issue is that my U and V factors are wrong.
    # Let me re-derive from scratch.
    
    # M1 = (A00+A11)(B00+B11)
    # This is a product of a linear combination of A entries and a linear combination of B entries.
    # As a rank-1 tensor: U1[i,j] * V1[j,k] where:
    # U1[0,0] = 1 (coeff of A00), U1[1,1] = 1 (coeff of A11), others 0
    # V1[0,0] = 1 (coeff of B00), V1[1,1] = 1 (coeff of B11), others 0
    
    # So for (i,j,k) = (0,0,0): U1[0,0] * V1[0,0] = 1*1 = 1. Good.
    # For (i,j,k) = (0,0,1): U1[0,0] * V1[0,1] = 1*0 = 0. Good (B00 doesn't have B01).
    # For (i,j,k) = (1,1,0): U1[1,1] * V1[1,0] = 1*0 = 0. Good.
    # For (i,j,k) = (1,1,1): U1[1,1] * V1[1,1] = 1*1 = 1. Good.
    
    # So the U and V factors for M1 are correct. The issue must be in how I'm summing.
    
    # Let me check all 8 entries of the 2x2x2 tensor:
    # (0,0,0): M1:1*1*1=1, M2:0*1*0=0, M3:1*0*0=0, M4:0*(-1)*(-1)=0, M5:1*0*0=0, M6:-1*1*1=-1, M7:0*0*0=0. Sum=0. WRONG.
    
    # The problem is clear: M6 = (A10-A00)(B00+B01).
    # U6[0,0] = -1 (coeff of A00 in A10-A00)
    # V6[0,0] = 1 (coeff of B00 in B00+B01)
    # W6[0,0] = 1 (M6 contributes +1 to c11)
    # Product: -1*1*1 = -1.
    
    # But M1 also gives +1. So they cancel. But T[0,0,0] should be 1, not 0.
    
    # I think the issue is that Strassen's algorithm doesn't directly give a tensor decomposition
    # in the way I'm thinking. The products M1...M7 are intermediate values, and the final
    # C entries are linear combinations of these M values. The tensor decomposition should
    # express T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i].
    
    # Let me think about it differently. Each "term" in the decomposition is a rank-1 tensor
    # U_c ⊗ V_c ⊗ W_c. The sum of these rank-1 tensors should equal the matrix multiplication tensor.
    
    # For Strassen, we have 7 products. Each product M_c = (sum_{i,j} u_c[i,j] * A[i,j]) * (sum_{j,k} v_c[j,k] * B[j,k])
    # and then C[i,k] = sum_c w_c[i,k] * M_c.
    
    # So the tensor is: T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[i,k]
    # Note: w is indexed by (i,k), not (k,i). But in our problem, W is indexed by (k,i) as W[k*n+i].
    # So w_c[i,k] = W_c[k*n+i].
    
    # This is the same as what I had. So why is it not working?
    
    # Let me just try a different approach: use the known fact that rank(2x2x2)=7 and
    # construct the decomposition numerically using optimization, then round to integers.
    
    # Actually, let me just hardcode the Strassen decomposition more carefully.
    
    # I'll define the 7 terms explicitly and verify each of the 8 tensor entries.
    
    # Let me use a completely different approach: just use the naive decomposition for now
    # and try to optimize. Given the time constraints, let me use the known rank-55 construction.
    
    # Actually, the simplest approach: use the naive rank-60 decomposition, which is guaranteed correct.
    # Then try to improve if time permits.
    
    # Let's go with the naive decomposition first to ensure correctness.
    rank = n * m * p  # 60
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify
    if not _verify_with_numpy(n, m, p, U, V, W):
        # This should never happen
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
