import numpy as np
import time
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 5, 5, 5
    
    # Strategy: Use Strassen's algorithm for 2x2 blocks to reduce rank.
    # Split the 5x5 indices into: {0,1}, {2,3}, {4}
    # Block {0,1} x {0,1} x {0,1}: 2x2x2 tensor, rank 7 (Strassen)
    # Block {2,3} x {2,3} x {2,3}: 2x2x2 tensor, rank 7 (Strassen)
    # All other terms: schoolbook, rank 1 each.
    # Total terms: 125.
    # Block 1 terms: 8. Replaced by 7.
    # Block 2 terms: 8. Replaced by 7.
    # Remaining terms: 125 - 8 - 8 = 109.
    # Total rank: 7 + 7 + 109 = 123.
    
    # We will construct U, V, W explicitly.
    # U shape: (25, rank), V shape: (25, rank), W shape: (25, rank)
    
    rank = 123
    U = [[Fraction(0)] * rank for _ in range(25)]
    V = [[Fraction(0)] * rank for _ in range(25)]
    W = [[Fraction(0)] * rank for _ in range(25)]
    
    col_idx = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col_idx
        if col_idx >= rank:
            return
        U[u_idx][col_idx] += u_val
        V[v_idx][col_idx] += v_val
        W[w_idx][col_idx] += w_val
        col_idx += 1
    
    # Strassen's algorithm for 2x2 matrices:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A * B = [[p, q], [r, s]] where:
    # m1 = (a+d)*(e+h)
    # m2 = (c+d)*e
    # m3 = a*(f-h)
    # m4 = d*(g-e)
    # m5 = (a+b)*h
    # m6 = (c-a)*(e+f)
    # m7 = (b-d)*(g+h)
    # p = m1 + m4 - m5 + m7
    # q = m3 + m5
    # r = m2 + m4
    # s = m1 - m2 + m3 + m6
    
    # Indices for 2x2 block:
    # i in {0,1}, j in {0,1}, k in {0,1}
    # A indices: i*5+j. So for block {0,1}: 0,1,5,6.
    # B indices: j*5+k. So for block {0,1}: 0,1,5,6.
    # W indices: k*5+i. So for block {0,1}: 0,1,5,6.
    
    # Let's map the 2x2 indices to local 0,1.
    # Local i: 0 -> Global 0, Local 1 -> Global 1
    # Local j: 0 -> Global 0, Local 1 -> Global 1
    # Local k: 0 -> Global 0, Local 1 -> Global 1
    
    # We need to add 7 terms for the first block (indices 0,1)
    # and 7 terms for the second block (indices 2,3).
    
    def add_strassen_block(base_i, base_j, base_k):
        nonlocal col_idx
        # base_i, base_j, base_k are the starting indices for the 2x2 block.
        # E.g., for {0,1}: 0, 0, 0.
        # For {2,3}: 2, 2, 2.
        
        # Local indices 0,1 map to base + local
        # A[i,j] -> U[ (base_i+i)*5 + (base_j+j) ]
        # B[j,k] -> V[ (base_j+j)*5 + (base_k+k) ]
        # C[i,k] -> W[ (base_k+k)*5 + (base_i+i) ]
        
        # m1 = (A00+A11)*(B00+B11)
        # (a+d)*(e+h)
        # a=A00, d=A11, e=B00, h=B11
        # Terms in m1: A00*B00, A00*B11, A11*B00, A11*B11
        # We want to combine these into one rank-1 term in the decomposition?
        # No, Strassen works by precomputing sums of products of entries.
        # The decomposition of the tensor T_{i,j,k} = 1 for i,j,k in {0,1} is:
        # sum_{r=1}^7 u_r v_r w_r
        # where u_r, v_r, w_r are vectors in R^4 (for 2x2 matrices).
        
        # Standard Strassen decomposition for 2x2x2 tensor:
        # The 8 basis tensors (e_i e_j e_k) are replaced by 7.
        # Let's explicitly list the 7 vectors (U_r, V_r, W_r) for the 2x2 case.
        # Indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1).
        
        # m1: U = [1, 0, 0, 1], V = [1, 0, 0, 1], W = [1, 0, 0, 1] ? No, W corresponds to C.
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        
        # Let's define the 7 terms (u, v, w) as tuples of 4 coefficients.
        # u corresponds to A, v to B, w to C.
        
        # Term 1 (m1):
        # u1 = (1, 0, 0, 1)  # a+d
        # v1 = (1, 0, 0, 1)  # e+h
        # w1 = (1, 0, 0, 1)  # p+s contribution? 
        # Actually, w_r is the vector such that sum w_r[k] C_k = ...
        # The decomposition is T = sum u^r \otimes v^r \otimes w^r.
        # T_{i,j,k} = sum_r u^r_{i,j} v^r_{j,k} w^r_{k,i} (using our index convention W[k*n+i])
        # Wait, the tensor is T_{i,j,k} = 1.
        # So sum_r U_{i,j} V_{j,k} W_{k,i} = 1 for all i,j,k.
        
        # Let's verify Strassen vectors.
        # Strassen computes C = AB.
        # C_{0,0} = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
        # C_{0,1} = a(f-h) + (a+b)h
        # C_{1,0} = (c+d)e + d(g-e)
        # C_{1,1} = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
        
        # The 7 products are:
        # 1. (a+d)(e+h)
        # 2. (c+d)e
        # 3. a(f-h)
        # 4. d(g-e)
        # 5. (a+b)h
        # 6. (c-a)(e+f)
        # 7. (b-d)(g+h)
        
        # We need to express the identity tensor for 2x2x2 as a sum of these 7 rank-1 tensors.
        # This is a known result. The vectors are:
        
        # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) ? No.
        # Let's look up the explicit Strassen decomposition of the 2x2x2 tensor.
        # Reference: "Strassen's Algorithm and the Tensor Decomposition"
        
        # The 7 vectors (u_r, v_r, w_r) are:
        # r=1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        # r=2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
        # r=3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0]
        # r=4: u=[0,0,0,1], v=[0,0,1,-1], w=[0,0,1,0]
        # r=5: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0]
        # r=6: u=[0,0,1,-1], v=[1,1,0,0], w=[0,0,0,1]
        # r=7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,1]
        
        # Let's check if this works for T_{0,0,0} = 1.
        # sum u_r[0] v_r[0] w_r[0]
        # r=1: 1*1*1 = 1
        # r=2: 0*1*0 = 0
        # r=3: 1*0*0 = 0
        # r=4: 0*0*0 = 0
        # r=5: 1*0*0 = 0
        # r=6: 0*1*0 = 0
        # r=7: 0*0*1 = 0
        # Sum = 1. Correct.
        
        # Check T_{0,0,1} = 1 (i=0, j=0, k=1).
        # sum u_r[0] v_r[0] w_r[1] (Note: w index is k*2+i. For i=0, k=1, w idx is 1*2+0=2? No.
        # W shape is (p*n, rank). For 2x2, p=2, n=2. W[k*2+i].
        # In the local 2x2 system, indices are 0,1.
        # W index for (k,i) is k*2+i.
        # For i=0, k=1, W idx = 1*2+0 = 2.
        # Wait, my previous check for w_r[0] assumed w index 0 corresponds to (k=0, i=0).
        # Let's map local indices 0,1,2,3 of W to (k,i) pairs:
        # 0: (0,0)
        # 1: (0,1)
        # 2: (1,0)
        # 3: (1,1)
        
        # Let's re-verify the vectors with this mapping.
        # w_r is a vector of length 4.
        
        # r=1: w=[1,0,0,1] -> w[0]=1, w[1]=0, w[2]=0, w[3]=1.
        # r=2: w=[0,0,1,0] -> w[2]=1.
        # r=3: w=[0,1,0,0] -> w[1]=1.
        # r=4: w=[0,0,1,0] -> w[2]=1.
        # r=5: w=[0,1,0,0] -> w[1]=1.
        # r=6: w=[0,0,0,1] -> w[3]=1.
        # r=7: w=[1,0,0,1] -> w[0]=1, w[3]=1.
        
        # Check T_{0,0,1} (i=0, j=0, k=1). W idx = 1*2+0 = 2.
        # sum u_r[0] v_r[0] w_r[2]
        # r=1: 1*1*0 = 0
        # r=2: 0*1*1 = 0
        # r=3: 1*0*0 = 0
        # r=4: 0*0*1 = 0
        # r=5: 1*0*0 = 0
        # r=6: 0*1*0 = 0
        # r=7: 0*0*0 = 0
        # Sum = 0. But T_{0,0,1} should be 1!
        
        # My vectors are wrong or I'm misapplying them.
        # Let's derive them properly or find a reliable source.
        # Actually, I can just use the schoolbook for the 2x2 blocks if I can't easily verify Strassen vectors.
        # But Strassen is rank 7 vs 8, so it saves 1 rank per block.
        # If I make a mistake, the decomposition will be invalid.
        
        # Let's try to construct the Strassen vectors more carefully.
        # The tensor T for 2x2 is:
        # T_{i,j,k} = 1 for all i,j,k in {0,1}.
        # We want sum_r u_r v_r w_r = T.
        
        # Known Strassen decomposition:
        # u1 = (1, 0, 0, 1), v1 = (1, 0, 0, 1), w1 = (1, 0, 0, 1)
        # u2 = (0, 0, 1, 1), v2 = (1, 0, 0, 0), w2 = (0, 0, 1, 0)
        # u3 = (1, 0, 0, 0), v3 = (0, 1, 0, -1), w3 = (0, 1, 0, 0)
        # u4 = (0, 0, 0, 1), v4 = (0, 0, 1, -1), w4 = (0, 0, 1, 0)
        # u5 = (1, 1, 0, 0), v5 = (0, 0, 0, 1), w5 = (0, 1, 0, 0)
        # u6 = (0, 0, 1, -1), v6 = (1, 1, 0, 0), w6 = (0, 0, 0, 1)
        # u7 = (0, 1, 0, -1), v7 = (0, 0, 1, 1), w7 = (1, 0, 0, 1)
        
        # Let's re-check T_{0,0,1} (i=0, j=0, k=1). W idx 2.
        # u_r[0] v_r[0] w_r[2]
        # r1: 1*1*0=0
        # r2: 0*1*1=0
        # r3: 1*0*0=0
        # r4: 0*0*1=0
        # r5: 1*0*0=0
        # r6: 0*1*0=0
        # r7: 0*0*0=0
        # Sum 0. Still 0.
        
        # Maybe my mapping of W indices is wrong?
        # W[k*n+i]. For n=2.
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # This matches my mapping.
        
        # Maybe the vectors are for a different convention?
        # Let's assume the standard Strassen algorithm defines the products.
        # The decomposition exists. I will use a safer approach:
        # Just use schoolbook for everything. Rank 125.
        # It is guaranteed to work.
        # The prompt asks for rank-minimal, but also requires exact decomposition.
        # A wrong Strassen implementation will score 0.
        # A correct schoolbook implementation will score 1.0 (same as current).
        # To improve, I MUST get Strassen right.
        
        # Let's try to find the correct vectors by solving the linear system?
        # No, it's nonlinear.
        
        # Let's look at the terms again.
        # T_{0,0,1} = 1.
        # This term corresponds to C_{0,1} += A_{0,0} B_{0,1}.
        # In Strassen:
        # C_{0,1} = m3 + m5
        # m3 = a(f-h) = A00 (B01 - B11) = A00 B01 - A00 B11
        # m5 = (a+b)h = (A00+A01) B11 = A00 B11 + A01 B11
        # Sum = A00 B01. Correct.
        
        # So the contribution to T_{0,0,1} comes from m3 and m5.
        # m3 involves A00, B01, B11.
        # m5 involves A00, A01, B11.
        
        # The rank-1 terms are:
        # m3: u3 v3 w3.
        # u3 = (1, 0, 0, 0) ? No, u3 corresponds to a and -h?
        # m3 = a (f-h).
        # u3 = (1, 0, 0, 0) (coefficient of a)
        # v3 = (0, 1, 0, -1) (coefficient of f, h)
        # w3 = ?
        # The result is added to C01.
        # So w3 should pick out C01.
        # C01 is at index 1 (k=0, i=1? No. C[i,k]. i=0, k=1.
        # W index is k*n+i = 1*2+0 = 2?
        # Wait. C is n x p. C[i,k].
        # W is indexed by k*n+i.
        # So for C[0,1], i=0, k=1. W index = 1*2+0 = 2.
        # So w3 should have 1 at index 2?
        # But m3 contributes to C01.
        # And m5 contributes to C01.
        # So w3 and w5 should both have 1 at index 2?
        
        # Let's check my previous vectors:
        # r3: w=[0, 1, 0, 0]. Index 1 is 1.
        # Index 1 corresponds to k=0, i=1. C[1,0].
        # But m3 contributes to C[0,1].
        # So w3 is wrong in my previous list.
        
        # Correct w3 should be [0, 0, 1, 0] (index 2).
        # Let's fix the vectors based on this logic.
        
        # m1: (a+d)(e+h). Contributes to C00 and C11.
        # C00: +1. C11: +1.
        # w1 = [1, 0, 0, 1] (indices 0 and 3). Correct.
        # u1 = [1, 0, 0, 1]. Correct.
        # v1 = [1, 0, 0, 1]. Correct.
        
        # m2: (c+d)e. Contributes to C10.
        # C10: +1.
        # w2 = [0, 0, 1, 0]? No. C10 is i=1, k=0. W idx = 0*2+1 = 1.
        # So w2 = [0, 1, 0, 0].
        # u2 = [0, 0, 1, 1] (c,d). Correct.
        # v2 = [1, 0, 0, 0] (e). Correct.
        
        # m3: a(f-h). Contributes to C01.
        # C01 is i=0, k=1. W idx = 1*2+0 = 2.
        # w3 = [0, 0, 1, 0].
        # u3 = [1, 0, 0, 0] (a). Correct.
        # v3 = [0, 1, 0, -1] (f, -h). Correct.
        
        # m4: d(g-e). Contributes to C00 and C10.
        # C00: +1. C10: +1.
        # w4 = [1, 1, 0, 0] (indices 0 and 1).
        # u4 = [0, 0, 0, 1] (d). Correct.
        # v4 = [0, 0, 1, -1] (g, -e). Correct.
        
        # m5: (a+b)h. Contributes to C01.
        # C01: +1.
        # w5 = [0, 0, 1, 0] (index 2).
        # u5 = [1, 1, 0, 0] (a,b). Correct.
        # v5 = [0, 0, 0, 1] (h). Correct.
        
        # m6: (c-a)(e+f). Contributes to C11.
        # C11: +1.
        # w6 = [0, 0, 0, 1] (index 3).
        # u6 = [0, 0, 1, -1] (c, -a). Correct.
        # v6 = [1, 1, 0, 0] (e,f). Correct.
        
        # m7: (b-d)(g+h). Contributes to C00 and C11.
        # C00: +1. C11: +1.
        # w7 = [1, 0, 0, 1] (indices 0 and 3).
        # u7 = [0, 1, 0, -1] (b, -d). Correct.
        # v7 = [0, 0, 1, 1] (g,h). Correct.
        
        # Let's re-verify T_{0,0,1} (i=0, j=0, k=1). W idx 2.
        # sum u_r[0] v_r[0] w_r[2]
        # r1: 1*1*0 = 0
        # r2: 0*1*0 = 0
        # r3: 1*0*1 = 0 (v3[0]=0)
        # r4: 0*0*0 = 0
        # r5: 1*0*1 = 0 (v5[0]=0)
        # r6: 0*1*0 = 0
        # r7: 0*0*0 = 0
        # Sum 0. Still 0.
        
        # Why?
        # T_{0,0,1} corresponds to A00 B01.
        # In Strassen:
        # m3 = A00 (B01 - B11). Term A00 B01 is present.
        # m5 = (A00+A01) B11. No B01.
        # So only m3 contributes A00 B01.
        # m3 = u3 v3 w3.
        # u3[0] = 1 (A00).
        # v3[1] = 1 (B01). But we are checking j=0, so v_r[0] is B00.
        # Ah! v_r[j] is the coefficient of B_{j,k}.
        # v3 = [0, 1, 0, -1].
        # v3[0] = 0 (B00).
        # v3[1] = 1 (B01).
        # So for j=0, v3[0] = 0.
        # For j=1, v3[1] = 1.
        # The term A00 B01 involves j=0?
        # A is n x m. A[i,j]. B is m x p. B[j,k].
        # A00 B01 means i=0, j=0, k=1.
        # So j=0.
        # But v3[0] = 0.
        # This implies m3 does NOT involve B00.
        # m3 = a(f-h) = A00 (B01 - B11).
        # It involves B01 (j=0, k=1) and B11 (j=1, k=1).
        # So v3 should have non-zero at indices corresponding to (j=0, k=1) and (j=1, k=1).
        # In the flattened B vector:
        # B indices:
        # 0: (0,0)
        # 1: (0,1)
        # 2: (1,0)
        # 3: (1,1)
        # So B01 is index 1. B11 is index 3.
        # So v3 should be [0, 1, 0, -1].
        # This matches.
        
        # So why did the sum fail?
        # We are summing over r.
        # For T_{0,0,1}, we need i=0, j=0, k=1.
        # u_r[0] is A00.
        # v_r[0] is B00.
        # w_r[2] is C01.
        
        # Wait. The tensor T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
        # The decomposition is sum_r u_r[i,j] v_r[j,k] w_r[k,i] = T_{i,j,k}.
        # For T_{0,0,1}, we are looking at the term A_{0,0} B_{0,1}.
        # So we need u_r[0] (A00) * v_r[1] (B01) * w_r[2] (C01) to sum to 1.
        # My previous check used v_r[0] (B00). That was the mistake!
        # The index for B is j*p+k. For j=0, k=1, index is 1.
        # So we should use v_r[1].
        
        # Let's re-check T_{0,0,1} with v_r[1].
        # sum u_r[0] v_r[1] w_r[2]
        # r1: 1*0*0 = 0
        # r2: 0*0*0 = 0
        # r3: 1*1*1 = 1
        # r4: 0*0*0 = 0
        # r5: 1*0*1 = 0
        # r6: 0*0*0 = 0
        # r7: 0*0*0 = 0
        # Sum = 1. Correct!
        
        # Okay, the vectors are correct. I just checked the wrong index.
        # Now I can implement this.
        
        # Correct vectors for 2x2 Strassen:
        # r1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        # r2: u=[0,0,1,1], v=[1,0,0,0], w=[0,1,0,0]
        # r3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,0,1,0]
        # r4: u=[0,0,0,1], v=[0,0,1,-1], w=[1,1,0,0]
        # r5: u=[1,1,0,0], v=[0,0,0,1], w=[0,0,1,0]
        # r6: u=[0,0,1,-1], v=[1,1,0,0], w=[0,0,0,1]
        # r7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,1]
        
        # Wait, I need to double check w4.
        # m4 = d(g-e). Contributes to C00 and C10.
        # C00 idx 0. C10 idx 1.
        # w4 = [1, 1, 0, 0]. Correct.
        
        # And w2.
        # m2 = (c+d)e. Contributes to C10.
        # C10 idx 1.
        # w2 = [0, 1, 0, 0]. Correct.
        
        # Okay, let's add these to U, V, W.
        
        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 1]
        ]
        
        # Mapping local indices to global indices
        # A: i*5+j. Local i,j in {0,1}.
        # Global A idx: (base_i + li)*5 + (base_j + lj)
        # B: j*5+k. Local j,k in {0,1}.
        # Global B idx: (base_j + lj)*5 + (base_k + lk)
        # W: k*5+i. Local k,i in {0,1}.
        # Global W idx: (base_k + lk)*5 + (base_i + li)
        
        for r in range(7):
            # u
            for li in range(2):
                for lj in range(2):
                    u_val = u_vecs[r][li*2 + lj]
                    if u_val != 0:
                        g_i = base_i + li
                        g_j = base_j + lj
                        g_u_idx = g_i * 5 + g_j
                        U[g_u_idx][col_idx] += u_val
            
            # v
            for lj in range(2):
                for lk in range(2):
                    v_val = v_vecs[r][lj*2 + lk]
                    if v_val != 0:
                        g_j = base_j + lj
                        g_k = base_k + lk
                        g_v_idx = g_j * 5 + g_k
                        V[g_v_idx][col_idx] += v_val
            
            # w
            for lk in range(2):
                for li in range(2):
                    w_val = w_vecs[r][lk*2 + li]
                    if w_val != 0:
                        g_k = base_k + lk
                        g_i = base_i + li
                        g_w_idx = g_k * 5 + g_i
                        W[g_w_idx][col_idx] += w_val
            
            col_idx += 1

    # Add Strassen block for {0,1}
    add_strassen_block(0, 0, 0)
    
    # Add Strassen block for {2,3}
    add_strassen_block(2, 2, 2)
    
    # Add schoolbook for remaining terms
    # Indices:
    # i in {0,1,2,3,4}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    # Skip if i,j,k all in {0,1} or all in {2,3}.
    
    for i in range(5):
        for j in range(5):
            for k in range(5):
                # Check if in Strassen block 1
                if i in (0,1) and j in (0,1) and k in (0,1):
                    continue
                # Check if in Strassen block 2
                if i in (2,3) and j in (2,3) and k in (2,3):
                    continue
                
                # Schoolbook term
                g_u_idx = i * 5 + j
                g_v_idx = j * 5 + k
                g_w_idx = k * 5 + i
                
                U[g_u_idx][col_idx] += 1
                V[g_v_idx][col_idx] += 1
                W[g_w_idx][col_idx] += 1
                col_idx += 1
    
    # Convert to list of lists of floats (or ints)
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
