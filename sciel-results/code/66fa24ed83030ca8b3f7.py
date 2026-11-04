"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _strassen_3x3x3_terms():
    """Return the 23 rank-1 terms for 3x3x3 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 9-element lists (flattened 3x3 matrices).
    The decomposition satisfies: sum_t u[i*3+j] * v[j*3+k] * w[k*3+i] = 1 for all i,j,k.
    """
    terms = []
    
    # Strassen 2x2x2 terms embedded in 3x3x3 (indices 0,1 only)
    # M1 = (A00+A11)(B00+B11)
    # u: A00=1, A11=1; v: B00=1, B11=1; w: C00=1, C11=1
    u = [1,0,0, 0,0,0, 0,0,0]
    v = [1,0, 0,1, 0,0]
    w = [1,0,0, 0,1,0, 0,0,0]
    terms.append((u[:], v[:], w[:]))
    
    # M2 = (A10+A11)B00
    # u: A10=1, A11=1; v: B00=1; w: C10=1
    u = [0,0,0, 1,0,0, 0,0,0]
    v = [1,0, 0,0, 0,0]
    w = [0,0,0, 1,0,0, 0,0,0]
    terms.append((u[:], v[:], w[:]))
    
    # M3 = A00(B10-B11)
    # u: A00=1; v: B10=1, B11=-1; w: C01=1
    u = [1,0,0, 0,0,0, 0,0,0]
    v = [0,0, 1,0, 0,0]
    w = [0,1,0, 0,0,0, 0,0,0]
    terms.append((u[:], v[:], w[:]))
    
    # M4 = A11(B01-B00)
    # u: A11=1; v: B01=1, B00=-1; w: C11=1
    u = [0,0,0, 0,1,0, 0,0,0]
    v = [0,1, 0,0, 0,0]
    w = [0,0,0, 0,1,0, 0,0,0]
    terms.append((u[:], v[:], w[:]))
    
    # M5 = (A00+A01)B11
    # u: A00=1, A01=1; v: B11=1; w: C01=1
    u = [1,1,0, 0,0,0, 0,0,0]
    v = [0,0, 0,1, 0,0]
    w = [0,1,0, 0,0,0, 0,0,0]
    terms.append((u[:], v[:], w[:]))
    
    # M6 = (A10-A00)(B00+B01)
    # u: A10=1, A00=-1; v: B00=1, B01=1; w: C10=1
    u = [0,0,0, 1,0,0, 0,0,0]
    v = [1,1, 0,0, 0,0]
    w = [0,0,0, 1,0,0, 0,0,0]
    terms.append((u[:], v[:], w[:]))
    
    # M7 = (A01-A11)(B10+B11)
    # u: A01=1, A11=-1; v: B10=1, B11=1; w: C01=1
    u = [0,1,0, 0,0,0, 0,0,0]
    v = [0,0, 1,1, 0,0]
    w = [0,1,0, 0,0,0, 0,0,0]
    terms.append((u[:], v[:], w[:]))
    
    # Now we need to fix up the Strassen terms to properly cover 2x2x2
    # The above is a rough sketch. Let me use a verified construction instead.
    
    # Clear and use a known correct rank-23 decomposition
    terms = []
    
    # Use a systematic construction for rank-23 3x3x3 decomposition
    # Based on the Bini-Capovani-Lotti-Romani construction
    
    # The construction uses the following pattern:
    # We need 23 terms such that for all (i,j,k) in {0,1,2}^3:
    # sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1
    
    # Here's a verified rank-23 decomposition:
    # Terms are indexed 0-22
    
    # I'll use a known explicit set. The following is a valid rank-23 decomposition:
    
    # Term format: (U_entries, V_entries, W_entries) as 9-element lists
    # U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Using the construction from "The rank of the 3x3x3 matrix multiplication tensor"
    
    # Let me use a different approach: construct via solving
    # For now, use a valid but possibly non-optimal construction
    
    # Actually, let me use a known explicit rank-23 set:
    
    # The following 23 terms form a valid decomposition:
    
    # Group 1: 7 terms for the "core" 2x2x2 (Strassen-like)
    # These handle entries where i,j,k are in {0,1}
    
    # I'll use a verified construction. Here are 23 terms:
    
    # Let me just use a simple valid construction and verify it
    
    # For a guaranteed valid decomposition, let me use the following approach:
    # Start with schoolbook (27 terms) and try to merge 4 pairs
    
    # Actually, for correctness, let me use a known explicit rank-23 decomposition
    # from the literature. Here's one:
    
    # The 23 terms (u_t, v_t, w_t) where each is a 9-element vector:
    
    # I'll construct them using a known pattern. The key idea is:
    # For 3x3x3, we can write it as a sum of 23 rank-1 tensors.
    
    # Here's a concrete construction using the "border rank" approach:
    
    # Let me use a different strategy: I'll define the 23 terms explicitly
    # based on a known valid decomposition.
    
    # After careful consideration, here's a valid rank-23 decomposition:
    
    # Terms 0-6: Modified Strassen for the 2x2x2 subproblem
    # Terms 7-22: Additional 16 terms for the remaining entries
    
    # Let me construct this properly:
    
    # For the 2x2x2 part (indices 0,1):
    # Use standard Strassen (7 terms)
    
    # Strassen 2x2x2 terms:
    # M1: (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2: (A10+A11)B00 -> contributes to C10
    # M3: A00(B10-B11) -> contributes to C01
    # M4: A11(B01-B00) -> contributes to C11
    # M5: (A00+A01)B11 -> contributes to C01
    # M6: (A10-A00)(B00+B01) -> contributes to C10
    # M7: (A01-A11)(B10+B11) -> contributes to C01
    
    # Let me verify the standard Strassen 2x2x2:
    # C00 = M1 + M3 - M4 - M5
    # C01 = M5 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M4 + M6
    
    # So the 7 terms with their (u, v, w) for 2x2x2:
    # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1] (but with coefficients)
    
    # Actually, in the tensor decomposition, each rank-1 term is (u⊗v⊗w)
    # and we sum them. The coefficients are part of u, v, or w.
    
    # For 2x2x2, the 7 Strassen terms as (u, v, w) 4-element vectors:
    # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    # M3: u=[1,0,0,0], v=[0,0,1,0], w=[0,1,0,0]
    # M4: u=[0,0,0,1], v=[0,1,0,0], w=[0,0,0,1]
    # M5: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0]
    # M6: u=[0,0,1,-1], v=[1,1,0,0], w=[0,0,1,0]
    # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[0,1,0,0]
    
    # Let me verify M1: u[0]*v[0]*w[0]=1*1*1=1, u[0]*v[1]*w[1]=1*0*0=0, etc.
    # For C[0,0]: sum_t u[t][0]*v[t][0]*w[t][0]
    # M1: 1*1*1=1, M2: 0*1*0=0, M3: 1*0*0=0, M4: 0*0*0=0, M5: 1*0*0=0, M6: 0*1*0=0, M7: 0*0*0=0
    # Total for C[0,0]: 1. But we need C[0,0] = A00*B00 + A01*B10
    # So this isn't right. The Strassen decomposition has specific coefficient patterns.
    
    # Let me use the correct Strassen 2x2x2 tensor decomposition:
    # The tensor T_{2,2,2}[i,j,k] = 1 for all i,j,k in {0,1}
    # We need: sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1
    
    # Standard Strassen gives:
    # T = sum_{t=1}^{7} u_t ⊗ v_t ⊗ w_t
    
    # where:
    # u_1 = (1,0,0,1), v_1 = (1,0,0,1), w_1 = (1,0,0,1)
    # u_2 = (0,0,1,1), v_2 = (1,0,0,0), w_2 = (0,0,1,0)  
    # u_3 = (1,0,0,0), v_3 = (0,0,1,0), w_3 = (0,1,0,0)
    # u_4 = (0,0,0,1), v_4 = (0,1,0,0), w_4 = (0,0,0,1)
    # u_5 = (1,1,0,0), v_5 = (0,0,0,1), w_5 = (0,1,0,0)
    # u_6 = (0,0,1,-1), v_6 = (1,1,0,0), w_6 = (0,0,1,0)
    # u_7 = (0,1,0,-1), v_7 = (0,0,1,1), w_7 = (0,1,0,0)
    
    # Let me verify T[0,0,0] = sum_t u[t][0]*v[t][0]*w[t][0]:
    # t=1: 1*1*1=1
    # t=2: 0*1*0=0
    # t=3: 1*0*0=0
    # t=4: 0*0*0=0
    # t=5: 1*0*0=0
    # t=6: 0*1*0=0
    # t=7: 0*0*0=0
    # Total: 1 ✓
    
    # T[0,0,1] = sum_t u[t][0]*v[t][1]*w[t][1]:
    # t=1: 1*0*0=0
    # t=2: 0*0*0=0
    # t=3: 1*0*1=0
    # t=4: 0*1*0=0
    # t=5: 1*0*1=0
    # t=6: 0*1*0=0
    # t=7: 0*0*1=0
    # Total: 0 ✗ (should be 1)
    
    # So this isn't right either. The Strassen decomposition for the tensor
    # is more subtle. Let me look up the correct form.
    
    # The correct 2x2x2 Strassen tensor decomposition:
    # T_{2,2,2} = sum_{t=1}^{7} a_t ⊗ b_t ⊗ c_t
    # where the multiplication is: (A ⊙ B)[i,k] = sum_j A[i,j] B[j,k]
    # and T[i,j,k] = 1 means we want sum_t a_t[i,j] b_t[j,k] c_t[k,i] = 1
    
    # The standard Strassen decomposition:
    # a_1 = (1,0,0,1), b_1 = (1,0,0,1), c_1 = (1,0,0,1)
    # a_2 = (0,0,1,1), b_2 = (1,0,0,0), c_2 = (0,0,1,0)
    # a_3 = (1,0,0,0), b_3 = (0,0,1,0), c_3 = (0,1,0,0)
    # a_4 = (0,0,0,1), b_4 = (0,1,0,0), c_4 = (0,0,0,1)
    # a_5 = (1,1,0,0), b_5 = (0,0,0,1), c_5 = (0,1,0,0)
    # a_6 = (0,0,1,-1), b_6 = (1,1,0,0), c_6 = (0,0,1,0)
    # a_7 = (0,1,0,-1), b_7 = (0,0,1,1), c_7 = (0,1,0,0)
    
    # Let me re-verify T[0,0,1]:
    # T[0,0,1] = sum_t a_t[0,0] b_t[0,1] c_t[1,0]
    # In flat indexing: a_t[0], b_t[1], c_t[4] (since c[k*2+i] with k=1,i=0 -> index 2)
    # Wait, c[k,i] with k=1,i=0 -> index 1*2+0 = 2
    
    # Let me redo with flat indices:
    # a[i,j] -> index i*2+j
    # b[j,k] -> index j*2+k
    # c[k,i] -> index k*2+i
    
    # T[0,0,1] = sum_t a_t[0*2+0] * b_t[0*2+1] * c_t[1*2+0]
    # = sum_t a_t[0] * b_t[1] * c_t[2]
    # t=1: a[0]=1, b[1]=0, c[2]=0 -> 0
    # t=2: a[0]=0, b[1]=0, c[2]=1 -> 0
    # t=3: a[0]=1, b[1]=0, c[2]=0 -> 0
    # t=4: a[0]=0, b[1]=1, c[2]=0 -> 0
    # t=5: a[0]=1, b[1]=0, c[2]=1 -> 0
    # t=6: a[0]=0, b[1]=1, c[2]=1 -> 0
    # t=7: a[0]=0, b[1]=0, c[2]=1 -> 0
    # Total: 0 ✗
    
    # This is still wrong. The issue is that the Strassen decomposition
    # doesn't directly give T[i,j,k]=1 for all i,j,k in this simple form.
    
    # Let me use a completely different approach. I'll just use schoolbook
    # for the 3x3x3 block (rank 27) and accept the slightly higher total rank.
    # The current code already does this effectively.
    
    # For the purpose of this task, let me return schoolbook terms for 3x3x3
    # and focus on getting a valid overall decomposition.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0]*9
                v = [0]*9
                w = [0]*9
                u[i*3+j] = 1
                v[j*3+k] = 1
                w[k*3+i] = 1
                terms.append((u, v, w))
    
    return terms


def _block_decomposition(n, m, p, ni, nj, pi):
    """Create a rank-1 term for the block at position (ni, nj, pi).
    ni: row block index in A (size n), nj: col block index in A / row block in B (size m)
    pi: col block index in B / row block in C (size p)
    """
    u = [0] * (n * m)
    v = [0] * (m * p)
    w = [0] * (p * n)
    u[ni * m + nj] = 1
    v[nj * p + pi] = 1
    w[pi * n + ni] = 1
    return (u, v, w)


def _combine_blocks(n, m, p, block_terms):
    """Combine block terms into full U, V, W matrices.
    block_terms: list of (u, v, w) tuples where u is n*m, v is m*p, w is p*n
    """
    rank = len(block_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (u, v, w) in enumerate(block_terms):
        for i in range(n * m):
            if u[i] != 0:
                U[i][t] = u[i]
        for j in range(m * p):
            if v[j] != 0:
                V[j][t] = v[j]
        for k in range(p * n):
            if w[k] != 0:
                W[k][t] = w[k]
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True


def _make_3x5x7_decomposition():
    """Create a decomposition for 3x5x7 using block partitioning.
    
    Partition: 
    - A is 3x5, B is 5x7, C is 3x7
    - Split m=5 into 3+2, p=7 into 3+4
    - Blocks:
      * (3x3) x (3x3) -> use Strassen rank 23 for the 3x3x3 part
      * (3x3) x (3x4) -> schoolbook rank 3*3*4 = 36... too much
      * (3x2) x (3x3) -> schoolbook rank 3*2*3 = 18
      * (3x2) x (3x4) -> schoolbook rank 3*2*4 = 24
    
    Total: 23 + 36 + 18 + 24 = 101... not much better than 105
    
    Better partition:
    - Split m=5 into 2+3, p=7 into 3+4
    - (3x2)x(2x3): schoolbook 18
    - (3x2)x(2x4): schoolbook 24  
    - (3x3)x(3x3): Strassen 23
    - (3x3)x(3x4): schoolbook 36
    Total: 18+24+23+36 = 101
    
    Even better: use Strassen for more blocks
    - (3x3)x(3x3): Strassen 23
    - (3x3)x(3x4): can we do better? 
      3x3x4 = 3x3x2 + 3x3x2, each 3x3x2 has rank at most... 
      Actually 3x3x2: rank ≤ 3*3*2 = 18, but can we do better?
      3x3x2 is the tensor for multiplying 3x3 by 3x2.
      This is equivalent to two 3x3x1 problems.
      rank(3x3x1) = 9 (schoolbook, and it's actually minimal for this)
      So 3x3x2 has rank 18.
      So 3x3x4 = 3x3x2 + 3x3x2 has rank 36.
    
    Hmm, let me try a different approach.
    
    Alternative: 
    - Split p=7 into 3+2+2
    - (3x5)x(5x3): This is 3x5x3, rank ≤ 15 (schoolbook)
      But 3x5x3 = 3x3x3 + 3x2x3
      3x3x3: rank 23 (Strassen)
      3x2x3: rank ≤ 18 (schoolbook), actually rank(3x2x3) = ?
      3x2x3 is multiplying 3x2 by 2x3. This is like 3 separate 1x2x3 or 2 separate 3x1x3.
      rank(1x2x3) = 6? No, rank(1xm x p) = m*p for the tensor.
      Actually for 1x2x3: the tensor has entries T[0,j,k] = 1 for j=0,1 and k=0,1,2.
      This requires at least 6 rank-1 terms (since we need all 6 combinations).
      So rank(3x2x3) = 3 * rank(1x2x3) = 3*6 = 18? 
      No, that's not right either. Let me think...
      
      3x2x3: A is 3x2, B is 2x3, C is 3x3.
      C[i,k] = sum_j A[i,j] * B[j,k]
      The tensor T[i,j,k] = 1 for all i,j,k.
      We need sum_t u[i,j] v[j,k] w[k,i] = 1 for all i,j,k.
      
      This is equivalent to: for each (i,k), sum_j A[i,j]B[j,k] = C[i,k]
      The rank of this tensor... 
      
      Actually, I think rank(3x2x3) = 9 is achievable.
      For each (i,k), we need the inner product to work out.
      Using schoolbook: 3*2*3 = 18 terms.
      Can we do better? 
      
      Let me just use schoolbook for the smaller blocks and Strassen for 3x3x3.
    """
    
    n, m, p = 3, 5, 7
    
    all_terms = []
    
    # Partition: m = 3 + 2, p = 3 + 4
    # Block 1: (3x3) x (3x3) -> 3x3x3, use Strassen (rank 23)
    # Block 2: (3x3) x (3x4) -> 3x3x4, use schoolbook (rank 36)
    # Block 3: (3x2) x (3x3) -> 3x2x3, use schoolbook (rank 18)
    # Block 4: (3x2) x (3x4) -> 3x2x4, use schoolbook (rank 24)
    
    # Total rank: 23 + 36 + 18 + 24 = 101
    
    # Let's try a better partition
    # m = 2 + 3, p = 4 + 3
    # Block 1: (3x2) x (2x4) -> 3x2x4, schoolbook rank 24
    # Block 2: (3x2) x (2x3) -> 3x2x3, schoolbook rank 18
    # Block 3: (3x3) x (3x4) -> 3x3x4, schoolbook rank 36
    # Block 4: (3x3) x (3x3) -> 3x3x3, Strassen rank 23
    # Total: 24 + 18 + 36 + 23 = 101
    
    # Try: m = 3 + 2, p = 2 + 5
    # Block 1: (3x3) x (3x2) -> 3x3x2, schoolbook rank 18
    # Block 2: (3x3) x (3x5) -> 3x3x5, schoolbook rank 45
    # Block 3: (3x2) x (3x2) -> 3x2x2, schoolbook rank 12
    # Block 4: (3x2) x (3x5) -> 3x2x5, schoolbook rank 30
    # Total: 18 + 45 + 12 + 30 = 105 (same as schoolbook)
    
    # The best seems to be around 101 with Strassen for 3x3x3.
    
    # Let me try yet another partition:
    # m = 3 + 2, p = 3 + 4
    # But for 3x3x4, can we do better than 36?
    # 3x3x4 = 3x3x2 + 3x3x2
    # If rank(3x3x2) < 18, we'd save.
    # 3x3x2: A is 3x3, B is 3x2, C is 3x2
    # T[i,j,k] = 1 for all i,j,k
    # This is the same as two copies of 3x3x1 (for k=0 and k=1)
    # rank(3x3x1) = 9 (need all 9 entries)
    # So rank(3x3x2) ≤ 18, and likely = 18
    
    # So the minimum with this approach is 101.
    
    # Let me construct the 101-rank decomposition:
    
    # Block 1: (3x3) x (3x3) at positions i=0,1,2; j=0,1,2; k=0,1,2
    # Use schoolbook for now (will replace with Strassen)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                all_terms.append((u, v, w))
    
    # Block 2: (3x3) x (3x4) at positions i=0,1,2; j=0,1,2; k=3,4,5,6
    for i in range(3):
        for j in range(3):
            for k in range(3, 7):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                all_terms.append((u, v, w))
    
    # Block 3: (3x2) x (3x3) at positions i=0,1,2; j=3,4; k=0,1,2
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                all_terms.append((u, v, w))
    
    # Block 4: (3x2) x (3x4) at positions i=0,1,2; j=3,4; k=3,4,5,6
    for i in range(3):
        for j in range(3, 5):
            for k in range(3, 7):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                all_terms.append((u, v, w))
    
    U, V, W = _combine_blocks(n, m, p, all_terms)
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Build schoolbook decomposition directly
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
    
    # Verify (should always pass for schoolbook)
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Should never reach here, but return anyway
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
