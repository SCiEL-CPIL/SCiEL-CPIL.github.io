import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    n, m, p = 2, 3, 3
    num_outputs = p * n  # 6
    num_inputs_a = n * m  # 6
    num_inputs_b = m * p  # 9
    
    def make_tensor_fast(U, V, W, rank):
        """Compute the multiplication tensor from factorization using vectorized ops"""
        # U: (6, rank), V: (9, rank), W: (6, rank)
        # tensor[ia, ib, out] = sum_r U[ia,r] * V[ib,r] * W[out,r]
        # = sum_r (U[:,r] * W[:,r]) * V[ib,r]
        # For each r: outer product U[:,r] x W[:,r] gives (6,6) matrix, scaled by V[:,r]
        tensor = np.zeros((num_inputs_a, num_inputs_b, num_outputs))
        for r in range(rank):
            if np.all(U[:, r] == 0) or np.all(V[:, r] == 0) or np.all(W[:, r] == 0):
                continue
            # For each ib, tensor[:, ib, :] += U[:,r] * V[ib,r] * W[:,r]
            # = V[ib,r] * outer(U[:,r], W[:,r])
            outer = np.outer(U[:, r], W[:, r])  # (6, 6)
            tensor[:, :, :] += outer[:, np.newaxis, :] * V[:, r][np.newaxis, :, np.newaxis]
        return tensor
    
    def target_tensor():
        T = np.zeros((num_inputs_a, num_inputs_b, num_outputs))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ia = i * m + j
                    ib = j * p + k
                    out_idx = k * n + i
                    T[ia, ib, out_idx] = 1.0
        return T
    
    def verify(U, V, W, rank, target):
        computed = make_tensor_fast(U, V, W, rank)
        return np.max(np.abs(computed - target)) < 1e-10
    
    target = target_tensor()
    
    # Try rank-16 construction first
    # Strategy: Strassen (7) for 2x2x2 + 9 for remaining
    # The remaining 10 terms need to be compressed to 9.
    # 
    # Remaining terms after Strassen handles A[:,:2]*B[:2,:2]:
    # C00 += A02*B20  (term 1)
    # C10 += A12*B20  (term 2)  
    # C01 += A02*B21  (term 3)
    # C11 += A12*B21  (term 4)
    # C02 += A00*B02  (term 5)
    # C02 += A01*B12  (term 6)
    # C02 += A02*B22  (term 7)
    # C12 += A10*B02  (term 8)
    # C12 += A11*B12  (term 9)
    # C12 += A12*B22  (term 10)
    #
    # Idea: Combine terms 1&2 (both use B20) and 3&4 (both use B21) using a shared structure.
    # Terms 1&2: A02*B20 -> C00, A12*B20 -> C10. This is a 2x1x1 x 1x1x2 = rank 2 (can't reduce)
    # Terms 5,6,7,8,9,10: C02 and C12 computations
    #   C02 = A00*B02 + A01*B12 + A02*B22
    #   C12 = A10*B02 + A11*B12 + A12*B22
    # These 6 terms form a 2x3x3 "diagonal" pattern. Can we do it in 5?
    # Using the identity: for vectors x,y,z of length 3, x·z and y·z can be computed
    # with rank 5 using: (x+y)·z, (x-y)·z, and then we still need 3 terms for z...
    # Actually no. x·z = sum xi*zi, y·z = sum yi*zi. 
    # Let u = x+y, v = x-y. Then x·z = (u·z + v·z)/2, y·z = (u·z - v·z)/2.
    # But u·z and v·z each need 3 terms, so total 6 terms. No savings.
    #
    # Alternative: Maybe we can share a term between the B20/B21 block and the B02/B12/B22 block.
    # Let's try: create a term that contributes to both C00 (via A02*B20) and C02 (via A02*B22)
    # Term: A02 * (B20 + B22) -> contributes to C00 and C02
    # Then we need to subtract A02*B22 from C02 and A02*B20 from... wait this gets complicated.
    
    # Let me try a direct rank-16 construction:
    rank16 = 16
    U16 = np.zeros((num_inputs_a, rank16))
    V16 = np.zeros((num_inputs_b, rank16))
    W16 = np.zeros((num_outputs, rank16))
    
    # Indices:
    # A: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # B: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # C: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1), 4:(0,2), 5:(1,2)
    
    # Strassen for A[:,:2] * B[:2,:2] -> C[:,:2]
    # r=0: (A00+A11)(B00+B11)
    U16[0, 0] = 1; U16[4, 0] = 1
    V16[0, 0] = 1; V16[4, 0] = 1
    W16[0, 0] = 1; W16[3, 0] = 1
    
    # r=1: (A10+A11)B00
    U16[3, 1] = 1; U16[4, 1] = 1
    V16[0, 1] = 1
    W16[1, 1] = 1; W16[3, 1] = -1
    
    # r=2: A00(B01-B11)
    U16[0, 2] = 1
    V16[1, 2] = 1; V16[4, 2] = -1
    W16[2, 2] = 1; W16[3, 2] = 1
    
    # r=3: A11(B10-B00)
    U16[4, 3] = 1
    V16[3, 3] = 1; V16[0, 3] = -1
    W16[0, 3] = 1; W16[1, 3] = 1
    
    # r=4: (A00+A01)B11
    U16[0, 4] = 1; U16[1, 4] = 1
    V16[4, 4] = 1
    W16[0, 4] = -1; W16[2, 4] = 1
    
    # r=5: (A10-A00)(B00+B01)
    U16[3, 5] = 1; U16[0, 5] = -1
    V16[0, 5] = 1; V16[1, 5] = 1
    W16[3, 5] = 1
    
    # r=6: (A01-A11)(B10+B11)
    U16[1, 6] = 1; U16[4, 6] = -1
    V16[3, 6] = 1; V16[4, 6] = 1
    W16[0, 6] = 1
    
    # Now for the remaining 9 terms (r=7..15):
    # We need: A02*B20->C00, A12*B20->C10, A02*B21->C01, A12*B21->C11
    #          A00*B02->C02, A01*B12->C02, A02*B22->C02
    #          A10*B02->C12, A11*B12->C12, A12*B22->C12
    # That's 10 terms. Need to merge one.
    
    # Try: r=7: A02*(B20+B22) -> C00 + C02
    U16[2, 7] = 1
    V16[6, 7] = 1; V16[8, 7] = 1
    W16[0, 7] = 1; W16[4, 7] = 1
    
    # r=8: A12*(B20+B22) -> C10 + C12
    U16[5, 8] = 1
    V16[6, 8] = 1; V16[8, 8] = 1
    W16[1, 8] = 1; W16[5, 8] = 1
    
    # r=9: A02*B21 -> C01
    U16[2, 9] = 1
    V16[7, 9] = 1
    W16[2, 9] = 1
    
    # r=10: A12*B21 -> C11
    U16[5, 10] = 1
    V16[7, 10] = 1
    W16[3, 10] = 1
    
    # Now we need to fix: C02 has extra A02*B22, C12 has extra A12*B22
    # And we still need: A00*B02->C02, A01*B12->C02, A10*B02->C12, A11*B12->C12
    # So remaining: 4 terms for A00*B02, A01*B12, A10*B02, A11*B12
    # Plus we need to subtract A02*B22 from C02 and A12*B22 from C12
    # But we can't subtract with positive rank terms easily unless we use negative weights.
    
    # r=11: A00*B02 -> C02
    U16[0, 11] = 1
    V16[2, 11] = 1
    W16[4, 11] = 1
    
    # r=12: A01*B12 -> C02
    U16[1, 12] = 1
    V16[5, 12] = 1
    W16[4, 12] = 1
    
    # r=13: A10*B02 -> C12
    U16[3, 13] = 1
    V16[2, 13] = 1
    W16[5, 13] = 1
    
    # r=14: A11*B12 -> C12
    U16[4, 14] = 1
    V16[5, 14] = 1
    W16[5, 14] = 1
    
    # r=15: -A02*B22 -> -C02, -A12*B22 -> -C12 (correction term)
    U16[2, 15] = -1; U16[5, 15] = -1
    V16[8, 15] = 1
    W16[4, 15] = -1; W16[5, 15] = -1
    
    # Verify rank-16
    if verify(U16, V16, W16, rank16, target):
        return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Try rank-15: attempt to find a valid decomposition
    # Strategy: Try different combinations for the "cross" terms
    # The 10 remaining terms after Strassen are:
    # A02*B20->C00, A12*B20->C10, A02*B21->C01, A12*B21->C11
    # A00*B02->C02, A01*B12->C02, A02*B22->C02
    # A10*B02->C12, A11*B12->C12, A12*B22->C12
    
    # Try rank-15 with various strategies
    best_rank = 16
    best_result = None
    
    # Strategy 1: Try to merge terms more aggressively
    # Use (A02+A12)*(B20+B22) and (A02-A12)*(B20-B22) type combinations
    # Plus handle B21 terms separately
    
    rank15 = 15
    U15 = np.zeros((num_inputs_a, rank15))
    V15 = np.zeros((num_inputs_b, rank15))
    W15 = np.zeros((num_outputs, rank15))
    
    # Strassen part (r=0..6)
    U15[0, 0] = 1; U15[4, 0] = 1
    V15[0, 0] = 1; V15[4, 0] = 1
    W15[0, 0] = 1; W15[3, 0] = 1
    
    U15[3, 1] = 1; U15[4, 1] = 1
    V15[0, 1] = 1
    W15[1, 1] = 1; W15[3, 1] = -1
    
    U15[0, 2] = 1
    V15[1, 2] = 1; V15[4, 2] = -1
    W15[2, 2] = 1; W15[3, 2] = 1
    
    U15[4, 3] = 1
    V15[3, 3] = 1; V15[0, 3] = -1
    W15[0, 3] = 1; W15[1, 3] = 1
    
    U15[0, 4] = 1; U15[1, 4] = 1
    V15[4, 4] = 1
    W15[0, 4] = -1; W15[2, 4] = 1
    
    U15[3, 5] = 1; U15[0, 5] = -1
    V15[0, 5] = 1; V15[1, 5] = 1
    W15[3, 5] = 1
    
    U15[1, 6] = 1; U15[4, 6] = -1
    V15[3, 6] = 1; V15[4, 6] = 1
    W15[0, 6] = 1
    
    # r=7: (A02+A12)*(B20+B22) -> contributes to C00+C10+C02+C12
    U15[2, 7] = 1; U15[5, 7] = 1
    V15[6, 7] = 1; V15[8, 7] = 1
    W15[0, 7] = 1; W15[1, 7] = 1; W15[4, 7] = 1; W15[5, 7] = 1
    
    # r=8: (A02-A12)*(B20-B22) -> correction for C00,C10,C02,C12
    U15[2, 8] = 1; U15[5, 8] = -1
    V15[6, 8] = 1; V15[8, 8] = -1
    W15[0, 8] = 1; W15[1, 8] = -1; W15[4, 8] = 1; W15[5, 8] = -1
    
    # r=9: A02*B21 -> C01
    U15[2, 9] = 1
    V15[7, 9] = 1
    W15[2, 9] = 1
    
    # r=10: A12*B21 -> C11
    U15[5, 10] = 1
    V15[7, 10] = 1
    W15[3, 10] = 1
    
    # r=11: A00*B02 -> C02
    U15[0, 11] = 1
    V15[2, 11] = 1
    W15[4, 11] = 1
    
    # r=12: A01*B12 -> C02
    U15[1, 12] = 1
    V15[5, 12] = 1
    W15[4, 12] = 1
    
    # r=13: A10*B02 -> C12
    U15[3, 13] = 1
    V15[2, 13] = 1
    W15[5, 13] = 1
    
    # r=14: A11*B12 -> C12
    U15[4, 14] = 1
    V15[5, 14] = 1
    W15[5, 14] = 1
    
    # Check: C00 gets: from r7: A02*B20 + A02*B22, from r8: A02*B20 - A02*B22
    # Total: 2*A02*B20. That's wrong, we need just A02*B20.
    # Let me fix: use 0.5 coefficients
    U15[2, 7] = 0.5; U15[5, 7] = 0.5
    V15[6, 7] = 1; V15[8, 7] = 1
    W15[0, 7] = 1; W15[1, 7] = 1; W15[4, 7] = 1; W15[5, 7] = 1
    
    U15[2, 8] = 0.5; U15[5, 8] = -0.5
    V15[6, 8] = 1; V15[8, 8] = -1
    W15[0, 8] = 1; W15[1, 8] = -1; W15[4, 8] = 1; W15[5, 8] = -1
    
    # Now C00: 0.5*A02*B20 + 0.5*A02*B22 + 0.5*A02*B20 - 0.5*A02*B22 = A02*B20 ✓
    # C10: 0.5*A12*B20 + 0.5*A12*B22 + (-0.5*A12*B20) - (-0.5*A12*B22) = 0.5*A12*B20 + 0.5*A12*B22 + 0.5*A12*B20 + 0.5*A12*B22 = A12*B20 + A12*B22
    # That's wrong for C10, we need just A12*B20.
    
    # This approach is getting complicated. Let me try a different rank-15 strategy.
    
    # Alternative rank-15: Use the fact that we can share B02 and B12 terms
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # Let x = A00+A10, y = A00-A10, z = A01+A11, w = A01-A11
    # Then A00*B02 + A10*B02 = x*B02, and A00*B02 - A10*B02 = y*B02
    # Similarly for B12 terms.
    # But this doesn't reduce rank since we still need separate terms for C02 and C12.
    
    # Let me just try to verify if our rank-16 works and return it if so
    if verify(U16, V16, W16, rank16, target):
        return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Fallback: rank-17 (known to work)
    rank17 = 17
    U17 = np.zeros((num_inputs_a, rank17))
    V17 = np.zeros((num_inputs_b, rank17))
    W17 = np.zeros((num_outputs, rank17))
    
    # Same Strassen part
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Schoolbook for remaining 10 terms
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1
    
    if verify(U17, V17, W17, rank17, target):
        return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Ultimate fallback: full schoolbook rank 18
    rank18 = 18
    U18 = np.zeros((num_inputs_a, rank18))
    V18 = np.zeros((num_inputs_b, rank18))
    W18 = np.zeros((num_outputs, rank18))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ia = i * m + j
                ib = j * p + k
                out_idx = k * n + i
                U18[ia, r] = 1
                V18[ib, r] = 1
                W18[out_idx, r] = 1
                r += 1
    return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
