"""
Strategy: Strassen-based algebraic construction exploiting block structure.

For C = A(2x2) * B(2x11), we use the Strassen decomposition for the 2x2 part.
The key insight: we can decompose the 2x2x11 tensor by:
1. Using the known rank-7 decomposition for 2x2x2 (Strassen)
2. Extending it to handle the 11 columns via linear combinations

The Strassen decomposition for 2x2x2 uses 7 rank-1 terms:
M1 = (A00+A11)(B00+B11)
M2 = (A10+A11)B00
M3 = A00(B10-B11)
M4 = A11(B01-B00)
M5 = (A00+A01)B11
M6 = (A10-A00)(B00+B10)
M7 = (A01-A11)(B01+B11)

C00 = M1 + M4 - M5 + M7
C01 = M3 + M5
C10 = M2 + M4
C11 = M1 - M2 + M3 + M6

For 2x2x11, we treat each column k independently but share the Strassen structure.
Actually, a better approach: since the 2x2 multiplication structure is fixed,
we can use 7 "Strassen multiplications" per pair of columns, but we need to be more careful.

Let me think differently. The tensor T[i,j,k] = 1 for all i,j,k with the constraint
that C[i,k] = sum_j A[i,j] B[j,k].

A direct algebraic construction:
For each k in 0..10, we need to compute C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
and C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]

Using Strassen on the 2x2 matrix product where we view B as having 11 "columns":
We can group the 11 columns. But a simpler valid construction:

Rank-44 is schoolbook. Let's find a better one.

Actually, the correct approach for 2x2xn:
The bilinear complexity of 2x2xn matrix multiplication is 2n+2 for n>=2? No.
For 2x2x2 it's 7. For 2x2x3 it's 11. For 2x2xn it's 2n+3 for n>=2? 
Let me check: 2*2+3=7 for n=2. 2*3+3=9 for n=3? But it's known to be 11.

Actually, the rank of 2x2xn is known:
- 2x2x2: 7
- 2x2x3: 11  
- 2x2x4: 15
- General: 2n+3 for n even? No, 2*2+3=7, 2*3+3=9≠11.

Let me just construct a valid decomposition directly.

Strategy: Use the fact that we can write:
C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]

Define auxiliary variables:
P_k = A[0,0]*B[0,k]  (for each k)
Q_k = A[0,1]*B[1,k]  (for each k)
R_k = A[1,0]*B[0,k]  (for each k)
S_k = A[1,1]*B[1,k]  (for each k)

That's 44 rank-1 terms (schoolbook).

To reduce rank, use Strassen-like combinations. For the 2x2 block:
Let's define for pairs of columns (2k, 2k+1):
M1_k = (A[0,0]+A[1,1])*(B[0,2k]+B[1,2k+1])
M2_k = (A[1,0]+A[1,1])*B[0,2k]
M3_k = A[0,0]*(B[1,2k]-B[1,2k+1])
M4_k = A[1,1]*(B[0,2k+1]-B[0,2k])
M5_k = (A[0,0]+A[0,1])*B[1,2k+1]
M6_k = (A[1,0]-A[0,0])*(B[0,2k]+B[1,2k])
M7_k = (A[0,1]-A[1,1])*(B[0,2k+1]+B[1,2k+1])

This is getting complex. Let me use a simpler valid construction.

Simplest correct approach: Use the known fact that the tensor rank of 2x2xn 
matrix multiplication can be achieved by a construction based on the 2x2x2 rank-7
decomposition applied recursively or via linear combinations.

For a guaranteed correct answer within time, I'll construct rank 39 using the following:
- 11 columns, each requiring 2 outputs (C[0,k] and C[1,k])
- Each output is a sum of 2 products
- Total 44 products, but we can share some via Strassen

Actually, let me just verify the schoolbook rank 44 and try to find a smaller one
via a direct algebraic construction that I can verify.

For 2x2x11, a valid rank-39 construction:
Use the identity:
C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]

Define for each k:
u_k = A[0,0] + A[0,1] + A[1,0] + A[1,1]  (scalar)
This doesn't directly help.

Let me use a different strategy: construct the decomposition explicitly and verify it.
I'll use a rank-44 schoolbook decomposition as the safe fallback, and attempt to find
a lower rank via a structured search in the algebraic space.
"""

import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Schoolbook rank: n*m*p = 44
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (22, rank)  
    # W shape: (p*n, rank) = (22, rank)
    
    # Index conventions:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # The tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try to find a rank < 44 decomposition using Strassen-based construction.
    
    # Strassen decomposition for 2x2x2 (rank 7):
    # Terms indexed by (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1}
    # U indices: i*2+j in {0,1,2,3}
    # V indices: j*2+k in {0,1,2,3}
    # W indices: k*2+i in {0,1,2,3}
    
    # For 2x2x11, we can use a block construction.
    # Let's try rank 39: use 7 terms for the "core" 2x2x2 structure repeated/extended.
    
    # Actually, let me construct a valid rank-44 solution first, then try to optimize.
    
    def _make_schoolbook():
        rank = n * m * p  # 44
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def _verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try Strassen-based construction for rank 39.
    # The idea: for 2x2xn, we can use a construction based on the 2x2x2 rank-7
    # decomposition. For n=11, we need to extend this.
    
    # A known result: the rank of 2x2xn matrix multiplication is at most 2n + 7 for n >= 2.
    # For n=2: 2*2+7 = 11? No, it's 7.
    # For n=3: 2*3+7 = 13? But it's 11.
    # Let me just try various ranks and see if I can construct valid decompositions.
    
    # Strategy: Use a linear algebra approach.
    # For a given rank R, we have R unknowns in each of U, V, W.
    # The constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # This is a system of 44 equations in 48R unknowns (but nonlinear).
    
    # For small R, we can try to solve this via a structured approach.
    
    # Let me try a specific construction for rank 39:
    # Use 11 "blocks" of 3 terms each + 6 extra terms = 39? 
    # Or 7 terms for the 2x2x2 core + 32 for the remaining 9 columns?
    
    # Actually, let me try a different approach: 
    # The 2x2x11 tensor can be seen as 11 copies of the 2x2x1 "slice" 
    # but they share the A matrix.
    
    # For each k, we need:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # Using Strassen on the 2x2 part for each pair of k values:
    # For columns (2t, 2t+1), define:
    # S1_t = (A[0,0]+A[1,1])
    # S2_t = (A[1,0]+A[1,1])
    # S3_t = A[0,0]
    # S4_t = A[1,1]
    # S5_t = (A[0,0]+A[0,1])
    # S6_t = (A[1,0]-A[0,0])
    # S7_t = (A[0,1]-A[1,1])
    
    # T1_t = B[0,2t]+B[1,2t+1]
    # T2_t = B[0,2t]
    # T3_t = B[1,2t]-B[1,2t+1]
    # T4_t = B[0,2t+1]-B[0,2t]
    # T5_t = B[1,2t+1]
    # T6_t = B[0,2t]+B[1,2t]
    # T7_t = B[0,2t+1]+B[1,2t+1]
    
    # M1_t = S1_t * T1_t
    # M2_t = S2_t * T2_t
    # M3_t = S3_t * T3_t
    # M4_t = S4_t * T4_t
    # M5_t = S5_t * T5_t
    # M6_t = S6_t * T6_t
    # M7_t = S7_t * T7_t
    
    # C[0,2t] = M1_t + M4_t - M5_t + M7_t
    # C[0,2t+1] = M3_t + M5_t
    # C[1,2t] = M2_t + M4_t
    # C[1,2t+1] = M1_t - M2_t + M3_t + M6_t
    
    # For 11 columns, we have 5 full pairs (0-1, 2-3, 4-5, 6-7, 8-9) and 1 single (10).
    # 5 pairs * 7 terms = 35 terms + 2 terms for the single column = 37 terms.
    # But wait, for the single column 10, we need:
    # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
    # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
    # That's 4 products, but we can use 2 terms if we're clever? No, we need at least 2 terms.
    # Actually for a single column, we need 2 rank-1 terms minimum (one for each output).
    # But each term contributes to both outputs potentially.
    
    # Let me reconsider. For the single column 10:
    # We need 2 rank-1 terms:
    # Term A: U[A[0,0]] * V[B[0,10]] * W[C[0,10]]
    # Term B: U[A[0,1]] * V[B[1,10]] * W[C[0,10]]
    # Term C: U[A[1,0]] * V[B[0,10]] * W[C[1,10]]
    # Term D: U[A[1,1]] * V[B[1,10]] * W[C[1,10]]
    # That's 4 terms for the single column.
    
    # Total: 5*7 + 4 = 39 terms.
    
    # Let me construct this explicitly.
    
    def _make_strassen_rank39():
        rank = 39
        U = np.zeros((4, rank), dtype=float)
        V = np.zeros((22, rank), dtype=float)
        W = np.zeros((22, rank), dtype=float)
        
        col = 0
        # 5 pairs of columns: (0,1), (2,3), (4,5), (6,7), (8,9)
        for t in range(5):
            k0 = 2 * t
            k1 = 2 * t + 1
            
            # Term 1: M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
            # U: A[0,0] -> idx 0, A[1,1] -> idx 3
            # V: B[0,k0] -> idx 0*11+k0 = k0, B[1,k1] -> idx 1*11+k1 = 11+k1
            # W: C[0,k0] -> idx k0*2+0 = 2*k0, C[0,k1] -> idx 2*k1, C[1,k0] -> idx 2*k0+1, C[1,k1] -> idx 2*k1+1
            # M1 contributes to: C[0,k0] (+1), C[0,k1] (-1 from -M5? No, let me re-derive)
            
            # Let me re-derive the Strassen formulas carefully.
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
            # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
            # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
            
            # Strassen:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for pair (k0, k1):
            # C[0,k0] = M1 + M4 - M5 + M7
            # C[0,k1] = M3 + M5
            # C[1,k0] = M2 + M4
            # C[1,k1] = M1 - M2 + M3 + M6
            
            # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
            U[0, col] = 1; U[3, col] = 1
            V[k0, col] = 1; V[11 + k1, col] = 1
            W[2*k0, col] = 1; W[2*k0+1, col] = -1; W[2*k1, col] = 0; W[2*k1+1, col] = 1
            # C[0,k0] gets +1, C[1,k0] gets -1, C[1,k1] gets +1
            col += 1
            
            # M2 = (A[1,0]+A[1,1]) * B[0,k0]
            U[2, col] = 1; U[3, col] = 1
            V[k0, col] = 1
            W[2*k0+1, col] = 1; W[2*k1+1, col] = -1
            # C[1,k0] gets +1, C[1,k1] gets -1
            col += 1
            
            # M3 = A[0,0] * (B[1,k0]-B[1,k1])
            U[0, col] = 1
            V[11 + k0, col] = 1; V[11 + k1, col] = -1
            W[2*k1, col] = 1; W[2*k1+1, col] = 1
            # C[0,k1] gets +1, C[1,k1] gets +1
            col += 1
            
            # M4 = A[1,1] * (B[0,k1]-B[0,k0])
            U[3, col] = 1
            V[k1, col] = 1; V[k0, col] = -1
            W[2*k0, col] = 1; W[2*k0+1, col] = 1
            # C[0,k0] gets +1, C[1,k0] gets +1
            col += 1
            
            # M5 = (A[0,0]+A[0,1]) * B[1,k1]
            U[0, col] = 1; U[1, col] = 1
            V[11 + k1, col] = 1
            W[2*k0, col] = -1; W[2*k1, col] = 1
            # C[0,k0] gets -1, C[0,k1] gets +1
            col += 1
            
            # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[1,k0])
            U[2, col] = 1; U[0, col] = -1
            V[k0, col] = 1; V[11 + k0, col] = 1
            W[2*k1+1, col] = 1
            # C[1,k1] gets +1
            col += 1
            
            # M7 = (A[0,1]-A[1,1]) * (B[0,k1]+B[1,k1])
            U[1, col] = 1; U[3, col] = -1
            V[k1, col] = 1; V[11 + k1, col] = 1
            W[2*k0, col] = 1
            # C[0,k0] gets +1
            col += 1
        
        # Single column 10: need 4 terms
        k = 10
        # Term: A[0,0] * B[0,10] -> C[0,10]
        U[0, col] = 1
        V[k, col] = 1
        W[2*k, col] = 1
        col += 1
        
        # Term: A[0,1] * B[1,10] -> C[0,10]
        U[1, col] = 1
        V[11 + k, col] = 1
        W[2*k, col] = 1
        col += 1
        
        # Term: A[1,0] * B[0,10] -> C[1,10]
        U[2, col] = 1
        V[k, col] = 1
        W[2*k + 1, col] = 1
        col += 1
        
        # Term: A[1,1] * B[1,10] -> C[1,10]
        U[3, col] = 1
        V[11 + k, col] = 1
        W[2*k + 1, col] = 1
        col += 1
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        return U, V, W, rank
    
    # Try the Strassen rank-39 construction
    U, V, W, rank = _make_strassen_rank39()
    
    if _verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails, fall back to schoolbook rank 44
    U, V, W, rank = _make_schoolbook()
    
    if _verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Should not reach here
    raise RuntimeError("Failed to construct valid decomposition")
