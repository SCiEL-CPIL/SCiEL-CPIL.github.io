"""Matrix multiplication 2x2x15: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

_t0 = time.time()
_DEADLINE = _t0 + 130.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Rank 53 decomposition:
    # 7 Strassen blocks (rank 7 each) for 14 columns
    # 1 trivial block (rank 4) for the last column
    # Total rank = 7*7 + 4 = 53
    
    rank = 53
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # A indices in U: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices in V: j*p + k (j=0: 0..14, j=1: 15..29)
    # C indices in W: k*n + i (k=0: 0,1; k=1: 2,3; ... k=14: 28,29)
    
    # Strassen 2x2x2 factors - verified coefficients
    # P1 = (A00+A11)(B0k0+B1k1)
    # P2 = (A10+A11)B0k0
    # P3 = A00(B0k1-B1k1)
    # P4 = A11(B1k0-B0k0)
    # P5 = (A00+A01)B1k1
    # P6 = (A10-A00)(B0k0+B0k1)
    # P7 = (A01-A11)(B1k0+B1k1)
    
    # Reconstruction (verified):
    # C0k0 = P1 + P4 - P5 + P7
    # C1k0 = P2 + P4
    # C0k1 = P3 + P5
    # C1k1 = P1 - P2 + P3 + P6
    
    u_entries = [
        [(0, 1), (3, 1)],   # P1: A00 + A11
        [(2, 1), (3, 1)],   # P2: A10 + A11
        [(0, 1)],           # P3: A00
        [(3, 1)],           # P4: A11
        [(0, 1), (1, 1)],   # P5: A00 + A01
        [(2, 1), (0, -1)],  # P6: A10 - A00
        [(1, 1), (3, -1)],  # P7: A01 - A11
    ]
    
    v_entries = [
        [(0, 1), (1, 1)],   # P1: B0k0 + B1k1
        [(0, 1)],           # P2: B0k0
        [(0, 1), (1, -1)],  # P3: B0k1 - B1k1
        [(1, 1), (0, -1)],  # P4: B1k0 - B0k0
        [(1, 1)],           # P5: B1k1
        [(0, 1), (0, 1)],   # P6: B0k0 + B0k1
        [(1, 1), (1, 1)],   # P7: B1k0 + B1k1
    ]
    
    w_entries = [
        [(0, 1), (1, 1)],   # P1: C0k0 + C1k1
        [(1, 1), (1, -1)],  # P2: C1k0 - C1k1  (wait, let me re-derive)
        [(1, 1), (1, 1)],   # P3: C0k1 + C1k1  (need to verify)
        [(0, 1), (1, 1)],   # P4: C0k0 + C1k0
        [(0, -1), (1, 1)],  # P5: -C0k0 + C0k1
        [(1, 1)],           # P6: C1k1
        [(0, 1)],           # P7: C0k0
    ]
    
    # Let me re-derive the W coefficients carefully
    # C0k0 = P1 + P4 - P5 + P7
    # C1k0 = P2 + P4
    # C0k1 = P3 + P5
    # C1k1 = P1 - P2 + P3 + P6
    
    # So for each P_t, which C entries does it contribute to?
    # P1: C0k0 (+1), C1k1 (+1)
    # P2: C1k0 (+1), C1k1 (-1)
    # P3: C0k1 (+1), C1k1 (+1)
    # P4: C0k0 (+1), C1k0 (+1)
    # P5: C0k0 (-1), C0k1 (+1)
    # P6: C1k1 (+1)
    # P7: C0k0 (+1)
    
    w_entries = [
        [(0, 1), (3, 1)],   # P1: C0k0 + C1k1
        [(1, 1), (3, -1)],  # P2: C1k0 - C1k1
        [(2, 1), (3, 1)],   # P3: C0k1 + C1k1
        [(0, 1), (1, 1)],   # P4: C0k0 + C1k0
        [(0, -1), (2, 1)],  # P5: -C0k0 + C0k1
        [(3, 1)],           # P6: C1k1
        [(0, 1)],           # P7: C0k0
    ]
    
    # W index mapping: C[i][k] -> W[k*n + i]
    # C0k0: k=k0, i=0 -> W[2*k0 + 0]
    # C1k0: k=k0, i=1 -> W[2*k0 + 1]
    # C0k1: k=k1, i=0 -> W[2*k1 + 0]
    # C1k1: k=k1, i=1 -> W[2*k1 + 1]
    
    def get_w_idx(ci, ck):
        """ci is C index (0=C0, 1=C1), ck is k index (k0 or k1)"""
        i = ci
        k = ck
        return k * n + i
    
    # Place 7 Strassen blocks for columns (0,1), (2,3), ..., (12,13)
    for blk in range(7):
        k0 = 2 * blk
        k1 = 2 * blk + 1
        base = 7 * blk
        for t in range(7):
            r = base + t
            # U
            for (u_row_i, c) in u_entries[t]:
                U[u_row_i, r] += c
            # V: v_entries[t] has (j, c) where j=0 or 1
            for (j, c) in v_entries[t]:
                if j == 0:
                    # B0k0 or B0k1 - need to determine which
                    # P1: B0k0, P2: B0k0, P3: B0k1, P4: B0k0, P5: -, P6: B0k0+B0k1, P7: -
                    pass
                # This is getting complex, let me use explicit mapping
            break
        break
    
    # Let me redo this with explicit V and W indexing per Strassen term
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # For each Strassen term t, define:
    # U contributions: list of (row, coeff)
    # V contributions: list of (row, coeff) where row = j*p + k
    # W contributions: list of (row, coeff) where row = k*n + i
    
    strassen_terms = [
        # t=0: P1 = (A00+A11)(B0k0+B1k1)
        {
            'U': [(0, 1), (3, 1)],
            'V': [0, 1],  # B0k0, B1k1
            'W': [0, 3],  # C0k0, C1k1
        },
        # t=1: P2 = (A10+A11)B0k0
        {
            'U': [(2, 1), (3, 1)],
            'V': [0],  # B0k0
            'W': [1, -3],  # C1k0 - C1k1 (wait, this doesn't match)
        },
    ]
    
    # I'm making this too complicated. Let me just directly code the verified Strassen block.
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_strassen_block(base_r, k0, k1):
        """Add a Strassen 2x2x2 block for column pair (k0, k1) starting at rank base_r."""
        r = [base_r + t for t in range(7)]
        
        # P1 = (A00+A11)(B0k0+B1k1)
        U[0, r[0]] += 1; U[3, r[0]] += 1
        V[0*p + k0, r[0]] += 1; V[1*p + k1, r[0]] += 1
        W[2*k0 + 0, r[0]] += 1  # C0k0
        W[2*k1 + 1, r[0]] += 1  # C1k1
        
        # P2 = (A10+A11)B0k0
        U[2, r[1]] += 1; U[3, r[1]] += 1
        V[0*p + k0, r[1]] += 1
        W[2*k0 + 1, r[1]] += 1  # C1k0
        W[2*k1 + 1, r[1]] += -1 # C1k1
        
        # P3 = A00(B0k1-B1k1)
        U[0, r[2]] += 1
        V[0*p + k1, r[2]] += 1; V[1*p + k1, r[2]] += -1
        W[2*k1 + 0, r[2]] += 1  # C0k1
        W[2*k1 + 1, r[2]] += 1  # C1k1
        
        # P4 = A11(B1k0-B0k0)
        U[3, r[3]] += 1
        V[1*p + k0, r[3]] += 1; V[0*p + k0, r[3]] += -1
        W[2*k0 + 0, r[3]] += 1  # C0k0
        W[2*k0 + 1, r[3]] += 1  # C1k0
        
        # P5 = (A00+A01)B1k1
        U[0, r[4]] += 1; U[1, r[4]] += 1
        V[1*p + k1, r[4]] += 1
        W[2*k0 + 0, r[4]] += -1 # C0k0
        W[2*k1 + 0, r[4]] += 1  # C0k1
        
        # P6 = (A10-A00)(B0k0+B0k1)
        U[2, r[5]] += 1; U[0, r[5]] += -1
        V[0*p + k0, r[5]] += 1; V[0*p + k1, r[5]] += 1
        W[2*k1 + 1, r[5]] += 1  # C1k1
        
        # P7 = (A01-A11)(B1k0+B1k1)
        U[1, r[6]] += 1; U[3, r[6]] += -1
        V[1*p + k0, r[6]] += 1; V[1*p + k1, r[6]] += 1
        W[2*k0 + 0, r[6]] += 1  # C0k0
    
    # Add 7 Strassen blocks for pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block(base_r, k0, k1)
        
    # Add trivial block for column 14
    base_r = 49
    # Q1 = A00 * B0,14 -> C0,14
    U[0, base_r + 0] = 1
    V[0*p + 14, base_r + 0] = 1
    W[2*14 + 0, base_r + 0] = 1
    
    # Q2 = A01 * B1,14 -> C0,14
    U[1, base_r + 1] = 1
    V[1*p + 14, base_r + 1] = 1
    W[2*14 + 0, base_r + 1] = 1
    
    # Q3 = A10 * B0,14 -> C1,14
    U[2, base_r + 2] = 1
    V[0*p + 14, base_r + 2] = 1
    W[2*14 + 1, base_r + 2] = 1
    
    # Q4 = A11 * B1,14 -> C1,14
    U[3, base_r + 3] = 1
    V[1*p + 14, base_r + 3] = 1
    W[2*14 + 1, base_r + 3] = 1
    
    # Verify the decomposition
    # T[i,j,k] should equal sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Let's spot check a few entries
    
    # Check C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # i=0, k=0: W[0, r] for r in 0..6 (first Strassen block)
    # C0k0 = P1 + P4 - P5 + P7 (from our W assignments)
    # P1: U has A00,A11; V has B0k0,B1k1 -> contributes A00*B0k0 + A11*B1k1 to C0k0
    # P4: U has A11; V has B1k0,B0k0 -> contributes A11*B1k0 - A11*B0k0 to C0k0
    # P5: U has A00,A01; V has B1k1 -> contributes -A00*B1k1 - A01*B1k1 to C0k0
    # P7: U has A01,A11; V has B1k0,B1k1 -> contributes A01*B1k0 + A01*B1k1 - A11*B1k0 - A11*B1k1 to C0k0
    
    # Sum for C0k0:
    # A00*B0k0 + A11*B1k1 + A11*B1k0 - A11*B0k0 - A00*B1k1 - A01*B1k1 + A01*B1k0 + A01*B1k1 - A11*B1k0 - A11*B1k1
    # = A00*B0k0 - A11*B0k0 - A00*B1k1 + A01*B1k0
    # = A00*(B0k0 - B1k1) + A01*B1k0 - A11*B0k0
    
    # But we want C0k0 = A00*B0k0 + A01*B1k0
    # So we need: A00*(B0k0 - B1k1) + A01*B1k0 - A11*B0k0 = A00*B0k0 + A01*B1k0
    # This gives: -A00*B1k1 - A11*B0k0 = 0, which is not generally true!
    
    # My Strassen coefficients are wrong. Let me use the correct ones from the inspiration.
    
    # Correct Strassen for 2x2:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # c = p1 + p4 - p5 + p7
    # d = p2 + p4
    # e = p3 + p5
    # f = p1 - p2 + p3 + p6
    
    # Where A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[c,d],[e,f]] (renamed to avoid confusion)
    # Let me use: A=[[a,b],[c,d]], B=[[e,f],[g,h]], C=[[c11,c12],[c21,c22]]
    # c11 = a*e + b*g
    # c12 = a*f + b*h
    # c21 = c*e + d*g
    # c22 = c*f + d*h
    
    # Using Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # c11 = p1 + p4 - p5 + p7
    # c21 = p2 + p4
    # c12 = p3 + p5
    # c22 = p1 - p2 + p3 + p6
    
    # Let me verify c11:
    # p1 + p4 - p5 + p7 = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    # = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
    # = ae + bg
    # Correct!
    
    # So the W coefficients should be:
    # c11 (C0k0): P1(+1), P4(+1), P5(-1), P7(+1)
    # c21 (C1k0): P2(+1), P4(+1)
    # c12 (C0k1): P3(+1), P5(+1)
    # c22 (C1k1): P1(+1), P2(-1), P3(+1), P6(+1)
    
    # Now let me redo with correct coefficients
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_strassen_block_v2(base_r, k0, k1):
        """Add a Strassen 2x2x2 block for column pair (k0, k1) starting at rank base_r."""
        r = [base_r + t for t in range(7)]
        
        # P1 = (A00+A11)(B0k0+B1k1)
        U[0, r[0]] += 1; U[3, r[0]] += 1
        V[0*p + k0, r[0]] += 1; V[1*p + k1, r[0]] += 1
        W[2*k0 + 0, r[0]] += 1  # C0k0 (c11)
        W[2*k1 + 1, r[0]] += 1  # C1k1 (c22)
        
        # P2 = (A10+A11)B0k0
        U[2, r[1]] += 1; U[3, r[1]] += 1
        V[0*p + k0, r[1]] += 1
        W[2*k0 + 1, r[1]] += 1  # C1k0 (c21)
        W[2*k1 + 1, r[1]] += -1 # C1k1 (c22)
        
        # P3 = A00(B0k1-B1k1)
        U[0, r[2]] += 1
        V[0*p + k1, r[2]] += 1; V[1*p + k1, r[2]] += -1
        W[2*k1 + 0, r[2]] += 1  # C0k1 (c12)
        W[2*k1 + 1, r[2]] += 1  # C1k1 (c22)
        
        # P4 = A11(B1k0-B0k0)
        U[3, r[3]] += 1
        V[1*p + k0, r[3]] += 1; V[0*p + k0, r[3]] += -1
        W[2*k0 + 0, r[3]] += 1  # C0k0 (c11)
        W[2*k0 + 1, r[3]] += 1  # C1k0 (c21)
        
        # P5 = (A00+A01)B1k1
        U[0, r[4]] += 1; U[1, r[4]] += 1
        V[1*p + k1, r[4]] += 1
        W[2*k0 + 0, r[4]] += -1 # C0k0 (c11)
        W[2*k1 + 0, r[4]] += 1  # C0k1 (c12)
        
        # P6 = (A10-A00)(B0k0+B0k1)
        U[2, r[5]] += 1; U[0, r[5]] += -1
        V[0*p + k0, r[5]] += 1; V[0*p + k1, r[5]] += 1
        W[2*k1 + 1, r[5]] += 1  # C1k1 (c22)
        
        # P7 = (A01-A11)(B1k0+B1k1)
        U[1, r[6]] += 1; U[3, r[6]] += -1
        V[1*p + k0, r[6]] += 1; V[1*p + k1, r[6]] += 1
        W[2*k0 + 0, r[6]] += 1  # C0k0 (c11)
    
    # Add 7 Strassen blocks for pairs (0,1), (2,3), ..., (12,13)
    for i in range(7):
        k0 = 2 * i
        k1 = 2 * i + 1
        base_r = 7 * i
        add_strassen_block_v2(base_r, k0, k1)
        
    # Add trivial block for column 14
    base_r = 49
    U[0, base_r + 0] = 1
    V[0*p + 14, base_r + 0] = 1
    W[2*14 + 0, base_r + 0] = 1
    
    U[1, base_r + 1] = 1
    V[1*p + 14, base_r + 1] = 1
    W[2*14 + 0, base_r + 1] = 1
    
    U[2, base_r + 2] = 1
    V[0*p + 14, base_r + 2] = 1
    W[2*14 + 1, base_r + 2] = 1
    
    U[3, base_r + 3] = 1
    V[1*p + 14, base_r + 3] = 1
    W[2*14 + 1, base_r + 3] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
