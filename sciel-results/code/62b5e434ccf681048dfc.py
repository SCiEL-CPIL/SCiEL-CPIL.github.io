import time
import sys
import itertools

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Build the standard rank-28 block-diagonal Strassen solution
    def build_rank28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, start_r):
            a11 = A_r0 * m + A_c0
            a12 = A_r0 * m + A_c0 + 1
            a21 = (A_r0 + 1) * m + A_c0
            a22 = (A_r0 + 1) * m + A_c0 + 1
            b11 = B_r0 * p + B_c0
            b12 = B_r0 * p + B_c0 + 1
            b21 = (B_r0 + 1) * p + B_c0
            b22 = (B_r0 + 1) * p + B_c0 + 1
            c11_idx = C_c0 * n + C_r0
            c12_idx = (C_c0 + 1) * n + C_r0
            c21_idx = C_c0 * n + (C_r0 + 1)
            c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
            r = start_r
            # M1 = (A11+A22)(B11+B22)
            U[a11][r] += 1; U[a22][r] += 1; V[b11][r] += 1; V[b22][r] += 1
            W[c11_idx][r] += 1; W[c22_idx][r] += 1; r += 1
            # M2 = (A21+A22)B11
            U[a21][r] += 1; U[a22][r] += 1; V[b11][r] += 1
            W[c21_idx][r] += 1; W[c22_idx][r] += -1; r += 1
            # M3 = A11(B12-B22)
            U[a11][r] += 1; V[b12][r] += 1; V[b22][r] += -1
            W[c12_idx][r] += 1; W[c22_idx][r] += 1; r += 1
            # M4 = A22(B21-B11)
            U[a22][r] += 1; V[b21][r] += 1; V[b11][r] += -1
            W[c11_idx][r] += 1; W[c21_idx][r] += 1; r += 1
            # M5 = (A11+A12)B22
            U[a11][r] += 1; U[a12][r] += 1; V[b22][r] += 1
            W[c11_idx][r] += -1; W[c12_idx][r] += 1; r += 1
            # M6 = (A21-A11)(B11+B12)
            U[a21][r] += 1; U[a11][r] += -1; V[b11][r] += 1; V[b12][r] += 1
            W[c22_idx][r] += 1; r += 1
            # M7 = (A12-A22)(B21+B22)
            U[a12][r] += 1; U[a22][r] += -1; V[b21][r] += 1; V[b22][r] += 1
            W[c11_idx][r] += 1; r += 1
            return r
        
        r = add_strassen(0, 0, 0, 0, 0, 0, 0)
        r = add_strassen(0, 2, 2, 0, 0, 0, r)
        r = add_strassen(0, 0, 0, 2, 0, 2, r)
        r = add_strassen(0, 2, 2, 2, 0, 2, r)
        assert r == 28
        return U, V, W
    
    U28, V28, W28 = build_rank28()
    
    # Verify rank-28 solution
    def verify(U, V, W, rank):
        """Check that sum_r U[i,r]*V[j,r] = W[k,r]*delta_{k, i+j mod ...} for matmul tensor"""
        # The matmul tensor: T[a_idx, b_idx, c_idx] = 1 if c_idx corresponds to a*b
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # Tensor element T[a, b, c] should equal sum_r U[a,r]*V[b,r]*W[c,r]
        # For matmul: T[a_idx, b_idx, c_idx] = 1 iff a_idx, b_idx, c_idx form valid product
        # a_idx = i*m + j (i in [0,n), j in [0,m))
        # b_idx = j*p + k (j in [0,m), k in [0,p))  -- must match same j
        # c_idx = k*n + i (k in [0,p), i in [0,n))   -- W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    expected = 1
                    actual = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
                    if abs(actual - expected) > 1e-9:
                        return False
                    # Also check that non-matching triples give 0
                # Check non-matching: different j
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k2 in range(p):
                        a_idx = i * m + j
                        b_idx = j2 * p + k2
                        c_idx = k2 * n + i
                        expected = 0
                        actual = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
                        if abs(actual) > 1e-9:
                            return False
        return True
    
    assert verify(U28, V28, W28, 28), "Rank-28 verification failed"
    
    # Now search for rank-27: try merging one Strassen term from P1 with one from P2 (both write to C_L)
    # or P3 with P4 (both write to C_R)
    # Strategy: remove term t1 from P1 and term t2 from P2, add one shared term that covers both
    
    # First, let's get the individual Strassen terms for each product
    def get_strassen_terms(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0):
        """Return list of 7 (U_dict, V_dict, W_dict) for one Strassen product"""
        a11 = A_r0 * m + A_c0
        a12 = A_r0 * m + A_c0 + 1
        a21 = (A_r0 + 1) * m + A_c0
        a22 = (A_r0 + 1) * m + A_c0 + 1
        b11 = B_r0 * p + B_c0
        b12 = B_r0 * p + B_c0 + 1
        b21 = (B_r0 + 1) * p + B_c0
        b22 = (B_r0 + 1) * p + B_c0 + 1
        c11_idx = C_c0 * n + C_r0
        c12_idx = (C_c0 + 1) * n + C_r0
        c21_idx = C_c0 * n + (C_r0 + 1)
        c22_idx = (C_c0 + 1) * n + (C_r0 + 1)
        
        terms = []
        # M1 = (A11+A22)(B11+B22)
        terms.append(({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11_idx: 1, c22_idx: 1}))
        # M2 = (A21+A22)B11
        terms.append(({a21: 1, a22: 1}, {b11: 1}, {c21_idx: 1, c22_idx: -1}))
        # M3 = A11(B12-B22)
        terms.append(({a11: 1}, {b12: 1, b22: -1}, {c12_idx: 1, c22_idx: 1}))
        # M4 = A22(B21-B11)
        terms.append(({a22: 1}, {b21: 1, b11: -1}, {c11_idx: 1, c21_idx: 1}))
        # M5 = (A11+A12)B22
        terms.append(({a11: 1, a12: 1}, {b22: 1}, {c11_idx: -1, c12_idx: 1}))
        # M6 = (A21-A11)(B11+B12)
        terms.append(({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22_idx: 1}))
        # M7 = (A12-A22)(B21+B22)
        terms.append(({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11_idx: 1}))
        return terms
    
    # Get terms for each product
    P1_terms = get_strassen_terms(0, 0, 0, 0, 0, 0)  # A_L * B_11 -> C_L
    P2_terms = get_strassen_terms(0, 2, 2, 0, 0, 0)  # A_R * B_21 -> C_L
    P3_terms = get_strassen_terms(0, 0, 0, 2, 0, 2)  # A_L * B_12 -> C_R
    P4_terms = get_strassen_terms(0, 2, 2, 2, 0, 2)  # A_R * B_22 -> C_R
    
    # Try to find a rank-27 decomposition by merging terms from same-output products
    # P1 and P2 both write to C_L (cols 0-1)
    # P3 and P4 both write to C_R (cols 2-3)
    
    def try_merge(p1_terms, p2_terms, p1_idx, p2_idx):
        """Try to replace term p1_idx from P1 and term p2_idx from P2 with one shared term.
        Returns (U_dict, V_dict, W_dict) for the shared term, or None."""
        U1, V1, W1 = p1_terms[p1_idx]
        U2, V2, W2 = p2_terms[p2_idx]
        
        # The shared term must satisfy:
        # For all (a,b,c) in the matmul support:
        #   U1[a]*V1[b]*W1[c] + U2[a]*V2[b]*W2[c] = S_U[a]*S_V[b]*S_W[c]
        # where S is the shared term
        # This is a system of equations. Let's try a simple approach:
        # Assume the shared term has the same structure as one of them, or is a combination.
        
        # Collect all relevant indices
        all_a = set(U1.keys()) | set(U2.keys())
        all_b = set(V1.keys()) | set(V2.keys())
        all_c = set(W1.keys()) | set(W2.keys())
        
        # For the matmul tensor, the support is:
        # (a_idx, b_idx, c_idx) where a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
        # with consistent j
        support = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    support.append((a_idx, b_idx, c_idx))
        
        # For each support point, compute the required value
        # sum = U1[a]*V1[b]*W1[c] + U2[a]*V2[b]*W2[c]
        # We need S_U[a]*S_V[b]*S_W[c] = sum for all support points
        # This is a rank-1 approximation of the "residual" tensor
        
        # Build the residual tensor
        # R[a,b,c] = U1[a]*V1[b]*W1[c] + U2[a]*V2[b]*W2[c]
        # We need R to be rank-1 (i.e., factorizable as S_U*S_V*S_W)
        
        # Get unique indices
        a_list = sorted(set(a for a, b, c in support))
        b_list = sorted(set(b for a, b, c in support))
        c_list = sorted(set(c for a, b, c in support))
        
        # Build R as a 3D tensor
        R = {}
        for a, b, c in support:
            val = U1.get(a, 0) * V1.get(b, 0) * W1.get(c, 0)
            val += U2.get(a, 0) * V2.get(b, 0) * W2.get(c, 0)
            R[(a, b, c)] = val
        
        # Check if R is rank-1: for all (a1,b1,c1) and (a2,b2,c2):
        # R[a1,b1,c1]*R[a2,b2,c2] = R[a1,b2,c1]*R[a2,b1,c2]
        # (This is the condition for a 3-tensor to be rank 1)
        
        # But we only care about the support points. Let's check consistency.
        # If R is rank-1 on the support, we can extract S_U, S_V, S_W
        
        # Try to factor: pick a "pivot" point where R != 0
        pivot = None
        for key in R:
            if R[key] != 0:
                pivot = key
                break
        
        if pivot is None:
            # R is zero everywhere on support -> trivial merge (use zero term)
            return ({}, {}, {})
        
        pa, pb, pc = pivot
        R_pivot = R[pivot]
        
        # S_U[a] * S_V[b] * S_W[c] = R[a,b,c]
        # S_U[a] = R[a, pb, pc] / (S_V[pb] * S_W[pc])
        # S_V[b] = R[pa, b, pc] / (S_U[pa] * S_W[pc])
        # S_W[c] = R[pa, pb, c] / (S_U[pa] * S_V[pb])
        
        # We have freedom to scale. Set S_U[pa] = 1, then:
        # S_V[pb] * S_W[pc] = R_pivot
        # Let's set S_V[pb] = 1, S_W[pc] = R_pivot
        # Then S_U[a] = R[a, pb, pc] / R_pivot
        # S_V[b] = R[pa, b, pc] / R_pivot  (since S_U[pa]=1, S_W[pc]=R_pivot)
        # S_W[c] = R[pa, pb, c] / R_pivot  (since S_U[pa]=1, S_V[pb]=1)
        
        S_U = {}
        S_V = {}
        S_W = {}
        
        # Compute S_U
        for a in a_list:
            if (a, pb, pc) in R:
                S_U[a] = R[(a, pb, pc)] / R_pivot
            else:
                S_U[a] = 0
        
        # Compute S_V
        for b in b_list:
            if (pa, b, pc) in R:
                S_V[b] = R[(pa, b, pc)] / R_pivot
            else:
                S_V[b] = 0
        
        # Compute S_W
        for c in c_list:
            if (pa, pb, c) in R:
                S_W[c] = R[(pa, pb, c)] / R_pivot
            else:
                S_W[c] = 0
        
        # Verify: for all support points, S_U[a]*S_V[b]*S_W[c] == R[a,b,c]
        for a, b, c in support:
            expected = R.get((a, b, c), 0)
            actual = S_U.get(a, 0) * S_V.get(b, 0) * S_W.get(c, 0)
            if abs(expected - actual) > 1e-9:
                return None  # Not rank-1
        
        # Clean up: remove zeros
        S_U = {k: v for k, v in S_U.items() if abs(v) > 1e-12}
        S_V = {k: v for k, v in S_V.items() if abs(v) > 1e-12}
        S_W = {k: v for k, v in S_W.items() if abs(v) > 1e-12}
        
        return (S_U, S_V, S_W)
    
    # Try all pairs of terms from P1/P2 and P3/P4
    best_result = None
    
    for pair_name, (t1_list, t2_list) in [("C_L", (P1_terms, P2_terms)), ("C_R", (P3_terms, P4_terms))]:
        for i in range(7):
            for j in range(7):
                if time.time() > deadline:
                    break
                shared = try_merge(t1_list, t2_list, i, j)
                if shared is None:
                    continue
                SU, SV, SW = shared
                if not SU or not SV or not SW:
                    # Degenerate: one of the factors is empty, meaning the term is zero
                    # This would mean we can just drop both terms (rank 26?) - unlikely but check
                    continue
                
                # Build the rank-27 candidate
                rank = 27
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                # Fill in all terms except the merged ones
                # Terms 0-6 from P1, 0-6 from P2, 0-6 from P3, 0-6 from P4
                # But skip term i from P1 and term j from P2, add shared term at position 26
                
                r = 0
                for pi in range(7):
                    if pi == i:
                        continue
                    Ud, Vd, Wd = t1_list[pi]
                    for a, val in Ud.items():
                        U[a][r] += val
                    for b, val in Vd.items():
                        V[b][r] += val
                    for c, val in Wd.items():
                        W[c][r] += val
                    r += 1
                
                for pj in range(7):
                    if pj == j:
                        continue
                    Ud, Vd, Wd = t2_list[pj]
                    for a, val in Ud.items():
                        U[a][r] += val
                    for b, val in Vd.items():
                        V[b][r] += val
                    for c, val in Wd.items():
                        W[c][r] += val
                    r += 1
                
                # P3 and P4 (unmodified)
                for terms in [P3_terms, P4_terms]:
                    for Ud, Vd, Wd in terms:
                        for a, val in Ud.items():
                            U[a][r] += val
                        for b, val in Vd.items():
                            V[b][r] += val
                        for c, val in Wd.items():
                            W[c][r] += val
                        r += 1
                
                # Add shared term
                for a, val in SU.items():
                    U[a][r] += val
                for b, val in SV.items():
                    V[b][r] += val
                for c, val in SW.items():
                    W[c][r] += val
                r += 1
                
                assert r == 27, f"Expected 27, got {r}"
                
                # Verify
                if verify(U, V, W, 27):
                    best_result = (U, V, W)
                    break
            if best_result is not None:
                break
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to rank 28
    return {"U": U28, "V": V28, "W": W28}
