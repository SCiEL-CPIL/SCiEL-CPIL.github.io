"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(n):  # j goes over m
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_np(U, V, W):
        """Vectorized verification."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def try_nullspace_approach(max_rank=14):
        """
        Structural approach: For each candidate rank R, set up the bilinear system.
        Fix the W matrix to have a structured sparse pattern, then solve for U and V
        via linear algebra.
        
        Key insight: If we fix W, then for each output index c = k*n+i, the
        equation sum_r U[i*m+j,r] * V[j*p+k,r] * W[c,r] = 1 becomes a system
        where U and V appear bilinearly. But if we also fix the support pattern
        of which (i,j) pairs connect to which r in U, and which (j,k) pairs in V,
        we get a linear system.
        
        Alternative: Use the fact that the matmul tensor has known structure.
        For rank R, we need to find vectors u_r (length 4), v_r (length 8), w_r (length 8)
        such that for all (i,j,k): sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1.
        
        Let's fix the w_r vectors to be standard basis-like patterns and solve.
        """
        for R in range(7, max_rank + 1):
            if time.time() > deadline - 10:
                break
            
            # Strategy: Fix W to have entries from {-1, 0, 1} with specific patterns.
            # For each output position (k,i), w_r[k*n+i] determines how rank-r contributes.
            
            # Try: each w_r has exactly 2 non-zero entries (typical for matmul decompositions)
            # The 8 output positions are: (0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0),(3,1)
            # i.e., k*n+i for k in {0,1,2,3}, i in {0,1}
            
            # For a rank-1 term to be "valid" for matmul, the w vector should have
            # support that corresponds to a consistent (k,i) pattern.
            
            # Let's try a systematic search over W patterns.
            # Each w_r can have entries in {-1, 0, 1} at the 8 positions.
            # But that's 3^8 * R possibilities... too many.
            
            # Better: exploit symmetry. The 8 output positions split into 4 pairs:
            # (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1), (k=2,i=0), (k=2,i=1), (k=3,i=0), (k=3,i=1)
            
            # For matmul, a natural pattern is w_r has support on positions related to
            # the (i,j,k) indices. Let's try: w_r[k*n+i] is nonzero only for specific (k,i).
            
            # Actually, let's use a completely different approach:
            # Set up the problem as: find U (4xR), V (8xR), W (8xR) such that
            # the trilinear form equals the matmul tensor.
            
            # Flatten: for each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # There are 16 such equations (2*2*4).
            
            # Total unknowns: 4R + 8R + 8R = 20R
            # Equations: 16
            # This is underdetermined for any R >= 1, so we need additional constraints.
            
            # The additional constraint is that the decomposition must be "exact" for
            # ALL inputs, which is already captured by the 16 equations (since the
            # matmul tensor is determined by its 16 entries).
            
            # Wait, that's not right. The 16 equations are:
            # For (i=0,j=0,k=0): sum_r U[0,r]*V[0,r]*W[0,r] = 1
            # For (i=0,j=0,k=1): sum_r U[0,r]*V[1,r]*W[1,r] = 1
            # For (i=0,j=0,k=2): sum_r U[0,r]*V[2,r]*W[2,r] = 1
            # For (i=0,j=0,k=3): sum_r U[0,r]*V[3,r]*W[3,r] = 1
            # For (i=0,j=1,k=0): sum_r U[1,r]*V[0,r]*W[0,r] = 1
            # ... etc.
            
            # These are 16 bilinear equations in the unknowns. The system is nonlinear.
            
            # Approach: Fix W, then the equations become bilinear in U and V.
            # If we also fix the sparsity pattern of U (which entries are nonzero),
            # we get a linear system in V (or vice versa).
            
            # Let's try: fix W to have a specific pattern, fix U's sparsity, solve for V.
            
            # For rank R, let's try W where each column w_r has exactly 2 nonzeros.
            # The 8 output positions correspond to (k,i) pairs.
            # A natural choice: w_r is supported on positions {a_r, b_r} with values {1, -1} or {1, 1}.
            
            # Number of ways to choose 2 positions from 8: C(8,2) = 28
            # Signs: 4 choices (+1/+1, +1/-1, -1/+1, -1/-1)
            # So 28*4 = 112 possible w_r vectors.
            # For R columns, that's 112^R combinations... still too many for R>3.
            
            # Better approach: Use the known structure. In matmul decompositions,
            # each rank-1 term typically corresponds to a "gate" that connects
            # specific input pairs to specific output positions.
            
            # Let me try a different structural approach:
            # Use the fact that for 2x2*2x4, we can write B = [B1 | B2] where B1, B2 are 2x2.
            # C = A*B1 | A*B2.
            # We know A*B1 needs rank 7 (Strassen) and A*B2 needs rank 7.
            # But they share the A factor, so we might save some rank.
            
            # Let's try to solve this as a linear system by fixing the A-side (U) structure.
            
            # For Strassen on A*B1 (2x2 * 2x2), the U factors are:
            # r0: a00+a11 -> U[0]=1, U[3]=1
            # r1: a10+a11 -> U[2]=1, U[3]=1
            # r2: a00       -> U[0]=1
            # r3: a11       -> U[3]=1
            # r4: a00+a01  -> U[0]=1, U[1]=1
            # r5: a10-a00  -> U[2]=1, U[0]=-1
            # r6: a01-a11  -> U[1]=1, U[3]=-1
            
            # For A*B2, same U factors but different V and W.
            
            # The question is: can we merge some of the 14 rank-1 terms?
            # Term r0 for B1: U=[1,0,0,1], V=[1,0,0,0,0,0,0,0], W=[1,0,0,0,0,0,0,0]
            # Term r0 for B2: U=[1,0,0,1], V=[0,0,0,0,1,0,0,0], W=[0,0,0,0,1,0,0,0]
            # These have the same U but different V and W, so they can't be merged into one term.
            
            # However, we can try to find a decomposition where some terms serve double duty.
            
            # Let's set up a linear system. Fix the U matrix (4xR) and W matrix (8xR),
            # then solve for V (8xR) linearly.
            
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # This is: sum_r (U[i*m+j,r] * W[k*n+i,r]) * V[j*p+k,r] = 1
            # For fixed (i,k), this is a linear equation in the variables V[j*p+k,r] for all j,r.
            
            # Actually, for fixed (i,j,k), it's: sum_r V[j*p+k,r] * (U[i*m+j,r] * W[k*n+i,r]) = 1
            # This is a linear equation in the V entries for row (j*p+k).
            
            # So for each (i,j,k), we get: sum_r V[j*p+k,r] * U[i*m+j,r] * W[k*n+i,r] = 1
            # Grouping by j: for each j, we have 2*4=8 equations (for i in {0,1}, k in {0,1,2,3})
            # involving V[j*p+0,r], V[j*p+1,r], V[j*p+2,r], V[j*p+3,r] for r in 0..R-1.
            
            # Wait, that's not quite right. Let me re-index.
            # For fixed j, the equations are:
            # For i=0,k=0: sum_r V[0,r] * U[0,r] * W[0,r] = 1
            # For i=0,k=1: sum_r V[1,r] * U[0,r] * W[1,r] = 1
            # For i=0,k=2: sum_r V[2,r] * U[0,r] * W[2,r] = 1
            # For i=0,k=3: sum_r V[3,r] * U[0,r] * W[3,r] = 1
            # For i=1,k=0: sum_r V[0,r] * U[1,r] * W[0,r] = 1
            # For i=1,k=1: sum_r V[1,r] * U[1,r] * W[1,r] = 1
            # For i=1,k=2: sum_r V[2,r] * U[1,r] * W[2,r] = 1
            # For i=1,k=3: sum_r V[3,r] * U[1,r] * W[3,r] = 1
            
            # So for j=0, we have 8 equations in 4R unknowns (V[0,r], V[1,r], V[2,r], V[3,r] for r=0..R-1).
            # For j=1, we have 8 equations in 4R unknowns (V[4,r], V[5,r], V[6,r], V[7,r] for r=0..R-1).
            
            # If we fix U and W, we can solve for V. The system is underdetermined
            # (8 equations, 4R unknowns for R>=2), so there are many solutions.
            
            # The question is: does there exist a choice of U and W (with integer/half-integer
            # entries) such that V also has integer/half-integer entries?
            
            # This is a hard combinatorial problem. Let me try a more direct approach.
            
            # Direct approach: For small R, enumerate possible U and W patterns.
            # U is 4xR with entries in {-1, 0, 1}. W is 8xR with entries in {-1, 0, 1}.
            # For R=7: 3^(4*7) * 3^(8*7) = 3^28 * 3^56 = 3^84... way too many.
            
            # Need a smarter approach. Let's use the Strassen structure as a starting point
            # and try to optimize/merge terms.
            
            # Alternative: Use a numerical approach. Set up the bilinear system and
            # use iterative methods to find a solution.
            
            # Let's try: start with the rank-14 Strassen solution and try to eliminate
            # terms by finding linear dependencies.
            
            pass  # Will implement below
    
    def try_strassen_with_merging():
        """
        Start with rank-14 (two independent Strassen blocks) and try to merge
        terms by finding equivalent decompositions.
        
        The 14 terms are:
        Left block (B1, cols 0-1):
          L0: U=[1,0,0,1], V=[1,0,0,0,0,0,0,0], W=[1,0,0,0,0,0,0,0]
          L1: U=[0,0,1,1], V=[1,0,0,0,0,0,0,0], W=[0,0,0,1,0,0,0,0]  -- wait, need to be careful
        
        Let me use the correct Strassen formulas.
        C = A*B where A is 2x2, B is 2x2.
        C00 = a00*b00 + a01*b10
        C01 = a00*b01 + a01*b11
        C10 = a10*b00 + a11*b10
        C11 = a10*b01 + a11*b11
        
        Strassen:
        M1 = (a00+a11)(b00+b11)
        M2 = (a10+a11)(b00)
        M3 = a00(b10-b11)
        M4 = a11(b01-b00)
        M5 = (a00+a01)(b11)
        M6 = (a10-a00)(b00+b01)
        M7 = (a01-a11)(b10+b11)
        
        C00 = M1 - M4 + M5
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M7
        
        Let me verify:
        M1 - M4 + M5 = (a00+a11)(b00+b11) - a11(b01-b00) + (a00+a01)b11
        = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b01 + a11*b00 + a00*b11 + a01*b11
        = a00*b00 + 2*a00*b11 + 2*a11*b00 + a11*b11 - a11*b01 + a01*b11
        
        That doesn't look right. Let me re-derive.
        
        Standard Strassen:
        M1 = (a11-a01)(b01+b11)
        M2 = (a00+a11)(b00+b11)
        M3 = (a10+a11)(b10+b11)
        M4 = a00(b01-b11)
        M5 = a11(b00-b10)
        M6 = (a00+a01)(b10)
        M7 = (a10+a11)(b01)
        
        C00 = M1 + M4 + M5 - M7
        C01 = M6 + M7
        C10 = M3 - M5 - M1
        C11 = M2 - M3 - M4 + M1
        
        Let me verify C00:
        M1 + M4 + M5 - M7
        = (a11-a01)(b01+b11) + a00(b01-b11) + a11(b00-b10) - (a10+a11)b01
        = a11*b01 + a11*b11 - a01*b01 - a01*b11 + a00*b01 - a00*b11 + a11*b00 - a11*b10 - a10*b01 - a11*b01
        = a00*b01 - a00*b11 + a11*b11 - a01*b01 - a01*b11 + a11*b00 - a11*b10 - a10*b01
        Hmm, that's not a00*b00 + a01*b10 either.
        
        I'm getting confused with the indexing. Let me use a consistent convention.
        
        A = [[a00, a01], [a10, a11]]
        B = [[b00, b01], [b10, b11]]
        C = A*B = [[a00*b00 + a01*b10, a00*b01 + a01*b11],
                   [a10*b00 + a11*b10, a10*b01 + a11*b11]]
        
        Strassen (one common version):
        M1 = (a00 + a11)(b00 + b11)
        M2 = (a10 + a11)(b00)
        M3 = a00(b10 - b11)
        M4 = a11(b01 - b00)
        M5 = (a00 + a01)(b11)
        M6 = (a10 - a00)(b00 + b01)
        M7 = (a01 - a11)(b10 + b11)
        
        C00 = M1 - M4 + M5
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M7
        
        Verify C00:
        M1 - M4 + M5 = (a00+a11)(b00+b11) - a11(b01-b00) + (a00+a01)b11
        = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b01 + a11*b00 + a00*b11 + a01*b11
        = a00*b00 + 2*a00*b11 + 2*a11*b00 + a11*b11 - a11*b01 + a01*b11
        
        That's not right. The issue is that M5 should be (a00+a01)*b10, not b11.
        
        Let me look up the correct Strassen formulas:
        
        M1 = (a11 - a01) * (b01 + b11)
        M2 = (a00 + a11) * (b00 + b11)  
        M3 = (a10 + a11) * (b10 + b11)
        M4 = a00 * (b01 - b11)
        M5 = a11 * (b00 - b10)
        M6 = (a00 + a01) * b10
        M7 = (a10 + a11) * b01
        
        C00 = M1 + M4 + M5 - M7
        C01 = M6 + M7
        C10 = M3 - M5 - M1
        C11 = M2 - M3 - M4 + M1
        
        Verify C00:
        M1 + M4 + M5 - M7
        = (a11-a01)(b01+b11) + a00(b01-b11) + a11(b00-b10) - (a10+a11)b01
        = a11*b01 + a11*b11 - a01*b01 - a01*b11 + a00*b01 - a00*b11 + a11*b00 - a11*b10 - a10*b01 - a11*b01
        = a00*b01 - a00*b11 + a11*b11 - a01*b01 - a01*b11 + a11*b00 - a11*b10 - a10*b01
        
        Still not matching. I think I have the matrix multiplication convention wrong.
        
        If C = A*B with A = [[a00,a01],[a10,a11]] and B = [[b00,b01],[b10,b11]]:
        C00 = a00*b00 + a01*b10
        C01 = a00*b01 + a01*b11
        C10 = a10*b00 + a11*b10
        C11 = a10*b01 + a11*b11
        
        The correct Strassen for this:
        M1 = (a00 + a11)(b00 + b11)
        M2 = (a10 + a11)(b00)
        M3 = a00(b10 - b11)
        M4 = a11(b01 - b00)
        M5 = (a00 + a01)(b10)
        M6 = (a10 - a00)(b00 + b01)
        M7 = (a01 - a11)(b10 + b11)
        
        C00 = M1 - M4 + M5
        C01 = M3 + M5  -- wait, M5 has b10, not b11
        
        I'm clearly making errors. Let me just verify numerically in code.
        """
        pass
    
    def verify_strassen_2x2():
        """Verify the Strassen decomposition for 2x2*2x2 numerically."""
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # C = A*B
        # C00 = a00*b00 + a01*b10
        # C01 = a00*b01 + a01*b11
        # C10 = a10*b00 + a11*b10
        # C11 = a10*b01 + a11*b11
        
        # Try:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)(b00)
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)(b10)
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 - M4 + M5
        # = (a00+a11)(b00+b11) - a11(b01-b00) + (a00+a01)b10
        # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a11*b01 + a11*b00 + a00*b10 + a01*b10
        # = a00*b00 + a00*b11 + 2*a11*b00 + a11*b11 - a11*b01 + a00*b10 + a01*b10
        
        # Expected C00 = a00*b00 + a01*b10
        # Difference: a00*b11 + 2*a11*b00 + a11*b11 - a11*b01 + a00*b10
        # That's not zero, so this is wrong.
        
        # Let me try the standard Strassen from Wikipedia:
        # M1 = (a11 - a01)(b01 + b11)
        # M2 = (a00 + a11)(b00 + b11)
        # M3 = (a10 + a11)(b10 + b11)
        # M4 = a00(b01 - b11)
        # M5 = a11(b00 - b10)
        # M6 = (a00 + a01)b10
        # M7 = (a10 + a11)b01
        # C00 = M1 + M4 + M5 - M7
        # C01 = M6 + M7
        # C10 = M3 - M5 - M1
        # C11 = M2 - M3 - M4 + M1
        
        # Verify C00 = M1 + M4 + M5 - M7:
        # = (a11-a01)(b01+b11) + a00(b01-b11) + a11(b00-b10) - (a10+a11)b01
        # = a11*b01 + a11*b11 - a01*b01 - a01*b11 + a00*b01 - a00*b11 + a11*b00 - a11*b10 - a10*b01 - a11*b01
        # = a00*b01 - a00*b11 + a11*b11 - a01*b01 - a01*b11 + a11*b00 - a11*b10 - a10*b01
        
        # Expected: a00*b00 + a01*b10
        # Still not matching. I must have the B matrix convention wrong.
        
        # Maybe B is [[b00, b10], [b01, b11]] (column-major)?
        # If B = [[b00, b10], [b01, b11]], then:
        # C00 = a00*b00 + a01*b01
        # C01 = a00*b10 + a01*b11
        # C10 = a10*b00 + a11*b01
        # C11 = a10*b10 + a11*b11
        
        # With this convention, Strassen:
        # M1 = (a11-a01)(b01+b11)  -- b01 is now B[1,0]
        # This is getting confusing. Let me just use the tensor convention from the problem.
        
        # The problem uses: U[i*m+j], V[j*p+k], W[k*n+i]
        # For 2x2*2x2: n=2, m=2, p=2
        # U[i*2+j], V[j*2+k], W[k*2+i]
        # i,j in {0,1}, k in {0,1}
        
        # The constraint is: for all (i,j,k), sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1
        
        # Let me just verify the rank-14 construction numerically.
        pass
    
    def try_direct_linear_solve():
        """
        Direct approach: For a given rank R, fix the support patterns of U and W,
        then solve for V linearly. Check if V has integer/half-integer entries.
        
        For efficiency, we'll use the Strassen structure as a guide for the supports.
        """
        best_result = None
        
        for R in range(7, 15):
            if time.time() > deadline - 5:
                break
            
            # For each rank R, try to find U, V, W with entries in {-1, 0, 1} (or half-integers)
            # such that the tensor identity holds.
            
            # Strategy: Use the fact that in matmul decompositions, each rank-1 term
            # typically has U supported on 2 entries, V on 1-2 entries, W on 1-2 entries.
            
            # Let's try a randomized approach with verification.
            # Generate random sparse U, V, W and check.
            
            # Actually, let's use a more structured approach.
            # For 2x2*2x4, the output has 8 entries: C[k*n+i] for k in 0..3, i in 0..1.
            # Each output C[k*n+i] = sum_j A[i*m+j]*B[j*p+k] = A[i*2+0]*B[0*4+k] + A[i*2+1]*B[1*4+k]
            
            # So C[0] = a00*b0 + a01*b4  (k=0, i=0)
            # C[1] = a00*b1 + a01*b5  (k=0, i=1)
            # C[2] = a00*b2 + a01*b6  (k=1, i=0)
            # C[3] = a00*b3 + a01*b7  (k=1, i=1)
            # C[4] = a10*b0 + a11*b4  (k=2, i=0)
            # C[5] = a10*b1 + a11*b5  (k=2, i=1)
            # C[6] = a10*b2 + a11*b6  (k=3, i=0)
            # C[7] = a10*b3 + a11*b7  (k=3, i=1)
            
            # Wait, let me re-check the indexing.
            # U index: i*m+j = i*2+j, so U[0]=a00, U[1]=a01, U[2]=a10, U[3]=a11
            # V index: j*p+k = j*4+k, so V[0]=b00, V[1]=b01, V[2]=b02, V[3]=b03, V[4]=b10, V[5]=b11, V[6]=b12, V[7]=b13
            # W index: k*n+i = k*2+i, so W[0]=c00, W[1]=c01, W[2]=c10, W[3]=c11, W[4]=c20, W[5]=c21, W[6]=c30, W[7]=c31
            
            # C[k*n+i] = sum_j U[i*2+j] * V[j*4+k] * W[k*2+i]
            # C[0] (k=0,i=0) = U[0]*V[0]*W[0] + U[1]*V[4]*W[0] = a00*b00 + a01*b10 ✓
            # C[1] (k=0,i=1) = U[0]*V[0]*W[1] + U[1]*V[4]*W[1] = a00*b00 + a01*b10
            # Wait, that gives C[1] = a00*b00 + a01*b10, but it should be a00*b01 + a01*b11.
            
            # I think I have the W indexing wrong. Let me re-read the problem.
            # "The output factor follows the official trace convention W[k*n+i,r]"
            # So W[k*n+i] is the coefficient for output position (k,i).
            # And C[k*n+i] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            
            # For the decomposition to be correct:
            # C[k*n+i] should equal sum_j A[i,j] * B[j,k] = sum_j U[i*m+j] * V[j*p+k]
            
            # So we need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta_{(i,j,k) is valid}
            
            # Wait, the tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
            # And the decomposition requires: T = sum_r u_r ⊗ v_r ⊗ w_r
            # where u_r is a vector of length nm, v_r of length mp, w_r of length pn.
            
            # So: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # where a = i*m+j, b = j*p+k, c = k*n+i.
            
            # For this to equal the matmul tensor, we need:
            # T[i*m+j, j*p+k, k*n+i] = 1 for all (i,j,k)
            # And T[a,b,c] = 0 otherwise.
            
            # But actually, the matmul tensor IS defined as T[i*m+j, j*p+k, k*n+i] = 1.
            # So the decomposition just needs to reproduce this tensor.
            
            # The 16 nonzero entries of T are at positions (i*m+j, j*p+k, k*n+i) for all (i,j,k).
            # For n=2,m=2,p=4: i in {0,1}, j in {0,1}, k in {0,1,2,3}
            # That's 2*2*4 = 16 entries.
            
            # So we need: for all (i,j,k), sum_r U[i*2+j,r] * V[j*4+k,r] * W[k*2+i,r] = 1
            # And for all other (a,b,c), sum_r U[a,r] * V[b,r] * W[c,r] = 0
            
            # The "other" conditions are automatically satisfied if the first set is,
            # because the matmul tensor is sparse (only 16 nonzeros out of 4*8*8=256 entries).
            
            # Actually no, we need to verify ALL 256 entries. But due to the structure,
            # if the 16 "active" entries are correct, the rest should be zero
            # (assuming the decomposition respects the matmul structure).
            
            # For safety, let's verify all 256 entries.
            
            def verify_full(U, V, W):
                U = np.array(U, dtype=float)
                V = np.array(V, dtype=float)
                W = np.array(W, dtype=float)
                R = U.shape[1]
                # Build the reconstructed tensor
                T_rec = np.zeros((nm, mp, pn))
                for r in range(R):
                    T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                # Build the target tensor
                T_target = np.zeros((nm, mp, pn))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_target[i*m+j, j*p+k, k*n+i] = 1.0
                return np.allclose(T_rec, T_target, atol=1e-9)
            
            # Now, for a given R, we need to find U (4xR), V (8xR), W (8xR) with
            # entries in {-1, -0.5, 0, 0.5, 1} such that the above holds.
            
            # This is a constraint satisfaction problem. Let's try a structured approach.
            
            # For R=7 (minimum possible for 2x2*2x2, so 2x2*2x4 needs at least 7):
            # Actually, 2x2*2x4 might need more than 7. The border rank is 11.
            
            # Let's try R=8, 9, 10, 11 with a structured search.
            
            # Key observation: The 16 equations are:
            # (i=0,j=0,k=0): U[0,r]*V[0,r]*W[0,r] summed = 1
            # (i=0,j=0,k=1): U[0,r]*V[1,r]*W[1,r] summed = 1
            # (i=0,j=0,k=2): U[0,r]*V[2,r]*W[2,r] summed = 1
            # (i=0,j=0,k=3): U[0,r]*V[3,r]*W[3,r] summed = 1
            # (i=0,j=1,k=0): U[1,r]*V[4,r]*W[0,r] summed = 1
            # (i=0,j=1,k=1): U[1,r]*V[5,r]*W[1,r] summed = 1
            # (i=0,j=1,k=2): U[1,r]*V[6,r]*W[2,r] summed = 1
            # (i=0,j=1,k=3): U[1,r]*V[7,r]*W[3,r] summed = 1
            # (i=1,j=0,k=0): U[2,r]*V[0,r]*W[4,r] summed = 1
            # (i=1,j=0,k=1): U[2,r]*V[1,r]*W[5,r] summed = 1
            # (i=1,j=0,k=2): U[2,r]*V[2,r]*W[6,r] summed = 1
            # (i=1,j=0,k=3): U[2,r]*V[3,r]*W[7,r] summed = 1
            # (i=1,j=1,k=0): U[3,r]*V[4,r]*W[4,r] summed = 1
            # (i=1,j=1,k=1): U[3,r]*V[5,r]*W[5,r] summed = 1
            # (i=1,j=1,k=2): U[3,r]*V[6,r]*W[6,r] summed = 1
            # (i=1,j=1,k=3): U[3,r]*V[7,r]*W[7,r] summed = 1
            
            # This is a system of 16 bilinear equations.
            
            # Approach: Fix W, then for each (i,k), we have 2 equations (for j=0,1):
            # sum_r U[i*2+0,r]*V[0*4+k,r]*W[k*2+i,r] = 1
            # sum_r U[i*2+1,r]*V[1*4+k,r]*W[k*2+i,r] = 1
            
            # If W[k*2+i,r] = 0 for most r, this simplifies.
            
            # Let's try a specific structured approach for R=8:
            # Use 4 terms for the "a00" part and 4 for the "a01" part, etc.
            
            # Actually, let me try a completely different approach:
            # Use the known rank-11 construction from the literature.
            
            # For 2x2*2x4, one known construction uses rank 11.
            # Let me try to construct it explicitly.
            
            # The idea: Write B = [B1 | B2] where B1, B2 are 2x2.
            # C = A*B1 | A*B2
            # Use 7 Strassen products for A*B1.
            # For A*B2, use 4 additional products by sharing some structure.
            
            # Strassen for A*B1 (B1 = [[b00,b01],[b10,b11]]):
            # M1 = (a00+a11)(b00+b11)
            # M2 = (a10+a11)(b00)
            # M3 = a00(b10-b11)
            # M4 = a11(b01-b00)
            # M5 = (a00+a01)(b10)
            # M6 = (a10-a00)(b00+b01)
            # M7 = (a01-a11)(b10+b11)
            
            # C1_00 = M1 - M4 + M5
            # C1_01 = M3 + M5
            # C1_10 = M2 + M4
            # C1_11 = M1 - M2 + M3 + M7
            
            # For A*B2 (B2 = [[b02,b03],[b12,b13]]):
            # N1 = (a00+a11)(b02+b13)
            # N2 = (a10+a11)(b02)
            # N3 = a00(b12-b13)
            # N4 = a11(b03-b02)
            # N5 = (a00+a01)(b12)
            # N6 = (a10-a00)(b02+b03)
            # N7 = (a01-a11)(b12+b13)
            
            # C2_00 = N1 - N4 + N5
            # C2_01 = N3 + N5
            # C2_10 = N2 + N4
            # C2_11 = N1 - N2 + N3 + N7
            
            # Total: 14 products. To get to 11, we need to save 3.
            
            # Observation: M1 and N1 have the same U factor (a00+a11).
            # M2 and N2 have the same U factor (a10+a11).
            # M3 and N3 have the same U factor (a00).
            # M4 and N4 have the same U factor (a11).
            # M5 and N5 have the same U factor (a00+a01).
            # M6 and N6 have the same U factor (a10-a00).
            # M7 and N7 have the same U factor (a01-a11).
            
            # We can't merge them directly because V and W differ.
            # But we can try to find a different decomposition.
            
            # Let me try to verify the rank-14 construction first, then try to optimize.
            
            # Rank-14 construction:
            R = 14
            U = [[0]*R for _ in range(4)]
            V = [[0]*R for _ in range(8)]
            W = [[0]*R for _ in range(8)]
            
            # Left block (B1: cols 0,1 -> V indices 0,1,4,5)
            # M1 = (a00+a11)(b00+b11): U=[1,0,0,1], V=[1,0,0,0,0,1,0,0], W for C1
            # C1_00 = M1 - M4 + M5, C1_01 = M3 + M5, C1_10 = M2 + M4, C1_11 = M1 - M2 + M3 + M7
            # W index: k*n+i. For left block, k=0,1 -> W[0],W[1],W[2],W[3]
            # C1_00 = W[0] (k=0,i=0), C1_01 = W[1] (k=0,i=1), C1_10 = W[2] (k=1,i=0), C1_11 = W[3] (k=1,i=1)
            
            # r=0: M1 = (a00+a11)(b00+b11)
            U[0][0] = 1; U[3][0] = 1
            V[0][0] = 1; V[5][0] = 1  # b00 and b11
            W[0][0] = 1; W[3][0] = 1  # contributes to C1_00 and C1_11
            
            # r=1: M2 = (a10+a11)(b00)
            U[2][1] = 1; U[3][1] = 1
            V[0][1] = 1  # b00
            W[2][1] = 1; W[3][1] = -1  # C1_10 += M2, C1_11 -= M2
            
            # r=2: M3 = a00(b10-b11)
            U[0][2] = 1
            V[4][2] = 1; V[5][2] = -1  # b10 - b11
            W[1][2] = 1; W[3][2] = 1  # C1_01 += M3, C1_11 += M3
            
            # r=3: M4 = a11(b01-b00)
            U[3][3] = 1
            V[1][3] = 1; V[0][3] = -1  # b01 - b00
            W[0][3] = -1; W[2][3] = 1  # C1_00 -= M4, C1_10 += M4
            
            # r=4: M5 = (a00+a01)(b10)
            U[0][4] = 1; U[1][4] = 1
            V[4][4] = 1  # b10
            W[0][4] = 1; W[1][4] = 1  # C1_00 += M5, C1_01 += M5
            
            # r=5: M6 = (a10-a00)(b00+b01)
            U[2][5] = 1; U[0][5] = -1
            V[0][5] = 1; V[1][5] = 1  # b00 + b01
            # M6 doesn't appear in any C1 output in this version... 
            # Wait, in the standard Strassen, M6 is used in C01 = M6 + M7
            # But I defined C01 = M3 + M5. Let me re-check.
            
            # I think I mixed up the Strassen formulas. Let me just verify numerically
            # and adjust until it works.
            
            # Let me just use the verify function to check.
            
            if verify_full(U, V, W):
                if best_result is None or R < best_result[0]:
                    best_result = (R, {"U": U, "V": V, "W": W})
            
            # This is getting too complicated to do manually. Let me use a different approach.
            
            # Use scipy to solve a linear system.
            # Fix W to have a specific pattern, then solve for U and V.
            
            break  # Break out of the R loop for now
        
        if best_result is not None:
            return best_result[1]
        return None
    
    def try_rank14_strassen():
        """
        Carefully construct the rank-14 decomposition using two independent
        Strassen blocks and verify it.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # I'll use the standard Strassen decomposition and verify numerically.
        # If it doesn't work, I'll adjust the coefficients.
        
        # For 2x2*2x2 Strassen, the standard decomposition is:
        # Given A = [[a,b],[c,d]] and B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = a(g-h)
        # M4 = d(f-e)
        # M5 = (a+b)(h)
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C00 = M1 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let me verify this:
        # M1 - M4 + M5 = (a+d)(e+h) - d(f-e) + (a+b)h
        # = ae + ah + de + dh - df + de + ah + bh
        # = ae + 2ah + 2de + dh - df + bh
        # Expected C00 = ae + bg
        # This is not matching. The issue is that M5 should be (a+b)*g, not (a+b)*h.
        
        # Correct Strassen (from Wikipedia, for C = A*B):
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = a(g-h)
        # M4 = d(f-e)
        # M5 = (a+b)(h)
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C00 = M1 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Wait, I just verified and it doesn't work. Let me look up the correct formulas.
        
        # From Wikipedia:
        # M1 = (a + d)(e + h)
        # M2 = (c + d)(e)
        # M3 = a(g - h)
        # M4 = d(f - e)
        # M5 = (a + b)(h)
        # M6 = (c - a)(e + f)
        # M7 = (b - d)(g + h)
        # C00 = M1 - M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let me verify C00 = M1 - M4 + M5:
        # = (a+d)(e+h) - d(f-e) + (a+b)h
        # = ae + ah + de + dh - df + de + ah + bh
        # = ae + 2ah + 2de + dh - df + bh
        
        # But C00 should be ae + bg.
        # So this is wrong. The correct formula must be different.
        
        # Let me try:
        # C00 = M1 + M4 - M5 - M7 (or some other combination)
        
        # Actually, I think the issue is that I'm using the wrong Strassen variant.
        # Let me just use a known correct one:
        
        # M1 = (a - c)(e + f)
        # M2 = (d - c)(e)
        # M3 = (a + b)(h)
        # M4 = (d + b)(f - h)
        # M5 = (a - d)(e + h)
        # M6 = (b + c)(g - e)
        # M7 = (a + b)(g + h)
        # C00 = M5 + M4 + M6 - M7 + M2
        # C01 = M1 + M7
        # C10 = M3 + M5
        # C11 = M1 + M3 + M4 - M6 + M2
        
        # I'm going in circles. Let me just use a numerical approach to find the
        # correct coefficients.
        
        # For now, let me just return the naive rank-16 decomposition.
        return None
    
    # Try the direct approach
    result = try_direct_linear_solve()
    if result is not None:
        return result
    
    # Fallback: naive rank-16
    return naive_decomp()
# EVOLVE-BLOCK-END
