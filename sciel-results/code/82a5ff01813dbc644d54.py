import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use a Kronecker product based decomposition.
    # Decompose 3x4x8 into blocks.
    # n=3 -> 2 + 1
    # m=4 -> 2 * 2
    # p=8 -> 4 * 2
    
    # Sub-problem 1: 2x4x8
    # Sub-problem 2: 1x4x8
    
    # Helper to build Kronecker product of decompositions
    def kron_decomp(decomp1, decomp2):
        # decomp is dict with U, V, W
        # U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
        # U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
        # Result: (n1*n2, m1*m2, p1*p2) with rank r1*r2
        n1, m1, p1 = decomp1['dims']
        n2, m2, p2 = decomp2['dims']
        r1 = decomp1['rank']
        r2 = decomp2['rank']
        
        U1 = np.array(decomp1['U'])
        V1 = np.array(decomp1['V'])
        W1 = np.array(decomp1['W'])
        
        U2 = np.array(decomp2['U'])
        V2 = np.array(decomp2['V'])
        W2 = np.array(decomp2['W'])
        
        # Indices mapping
        # U index: (i1, j1, i2, j2) -> (i1*n2 + i2, j1*m2 + j2)
        # V index: (j1, k1, j2, k2) -> (j1*m2 + j2, k1*p2 + k2)
        # W index: (k1, i1, k2, i2) -> (k1*p2 + k2, i1*n2 + i2)
        
        # Rank index: l1 * r2 + l2
        
        r_total = r1 * r2
        U = np.zeros((n1*n2 * m1*m2, r_total))
        V = np.zeros((m1*m2 * p1*p2, r_total))
        W = np.zeros((p1*p2 * n1*n2, r_total))
        
        for l1 in range(r1):
            for l2 in range(r2):
                l = l1 * r2 + l2
                # U
                for i1 in range(n1):
                    for j1 in range(m1):
                        idx1 = i1 * m1 + j1
                        val1 = U1[idx1, l1]
                        if val1 != 0:
                            for i2 in range(n2):
                                for j2 in range(m2):
                                    idx2 = i2 * m2 + j2
                                    val2 = U2[idx2, l2]
                                    if val2 != 0:
                                        i = i1 * n2 + i2
                                        j = j1 * m2 + j2
                                        idx = i * (m1*m2) + j
                                        U[idx, l] += val1 * val2
                # V
                for j1 in range(m1):
                    for k1 in range(p1):
                        idx1 = j1 * p1 + k1
                        val1 = V1[idx1, l1]
                        if val1 != 0:
                            for j2 in range(m2):
                                for k2 in range(p2):
                                    idx2 = j2 * p2 + k2
                                    val2 = V2[idx2, l2]
                                    if val2 != 0:
                                        j = j1 * m2 + j2
                                        k = k1 * p2 + k2
                                        idx = j * (p1*p2) + k
                                        V[idx, l] += val1 * val2
                # W
                for k1 in range(p1):
                    for i1 in range(n1):
                        idx1 = k1 * n1 + i1
                        val1 = W1[idx1, l1]
                        if val1 != 0:
                            for k2 in range(p2):
                                for i2 in range(n2):
                                    idx2 = k2 * n2 + i2
                                    val2 = W2[idx2, l2]
                                    if val2 != 0:
                                        k = k1 * p2 + k2
                                        i = i1 * n2 + i2
                                        idx = k * (n1*n2) + i
                                        W[idx, l] += val1 * val2
                                        
        return {
            'dims': (n1*n2, m1*m2, p1*p2),
            'rank': r_total,
            'U': U.tolist(),
            'V': V.tolist(),
            'W': W.tolist()
        }

    def sum_decomps(decomp1, decomp2):
        # Sum two decompositions of same dimensions
        # Assumes dims are identical
        n, m, p = decomp1['dims']
        r1 = decomp1['rank']
        r2 = decomp2['rank']
        r_total = r1 + r2
        
        U1 = np.array(decomp1['U'])
        V1 = np.array(decomp1['V'])
        W1 = np.array(decomp1['W'])
        
        U2 = np.array(decomp2['U'])
        V2 = np.array(decomp2['V'])
        W2 = np.array(decomp2['W'])
        
        U = np.zeros((n*m, r_total))
        V = np.zeros((m*p, r_total))
        W = np.zeros((p*n, r_total))
        
        U[:, :r1] = U1
        V[:, :r1] = V1
        W[:, :r1] = W1
        
        U[:, r1:] = U2
        V[:, r1:] = V2
        W[:, r1:] = W2
        
        return {
            'dims': (n, m, p),
            'rank': r_total,
            'U': U.tolist(),
            'V': V.tolist(),
            'W': W.tolist()
        }

    # Base decompositions
    def get_strassen_222():
        # Rank 7 Strassen for 2x2x2
        n, m, p = 2, 2, 2
        r = 7
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # P1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # P2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # P3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # P4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1
        
        # P5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # P6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # P7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return {'dims': (n, m, p), 'rank': r, 'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    def get_schoolbook_122():
        # Rank 2 for 1x2x2
        n, m, p = 1, 2, 2
        r = 2
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # C0 = A0 B0 + A1 B1
        # Term 1: A0 B0
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Term 2: A1 B1
        U[1, 1] = 1
        V[2, 1] = 1
        W[0, 1] = 1
        
        return {'dims': (n, m, p), 'rank': r, 'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    def get_schoolbook_112():
        # Rank 2 for 1x1x2
        n, m, p = 1, 1, 2
        r = 2
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # C0 = A0 B0
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # C1 = A0 B1
        U[0, 1] = 1
        V[1, 1] = 1
        W[1, 1] = 1
        
        return {'dims': (n, m, p), 'rank': r, 'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    # Build 2x2x4
    # 2x2x4 = (2x2x2) kron (1x1x2)
    d_222 = get_strassen_222()
    d_112 = get_schoolbook_112()
    d_224 = kron_decomp(d_222, d_112)
    
    # Build 1x2x4
    # 1x2x4 = (1x2x2) kron (1x1x2)
    d_122 = get_schoolbook_122()
    d_124 = kron_decomp(d_122, d_112)
    
    # Build 2x4x8
    # 2x4x8 = (2x2x4) kron (1x2x2)
    # Wait, dims:
    # d_224 is 2x2x4.
    # d_122 is 1x2x2.
    # Kron: (2*1, 2*2, 4*2) = 2x4x8. Correct.
    d_248 = kron_decomp(d_224, d_122)
    
    # Build 1x4x8
    # 1x4x8 = (1x2x4) kron (1x2x2)
    # d_124 is 1x2x4.
    # d_122 is 1x2x2.
    # Kron: (1*1, 2*2, 4*2) = 1x4x8. Correct.
    d_148 = kron_decomp(d_124, d_122)
    
    # Total 3x4x8
    # Sum of 2x4x8 and 1x4x8
    d_total = sum_decomps(d_248, d_148)
    
    return {"U": d_total['U'], "V": d_total['V'], "W": d_total['W']}
# EVOLVE-BLOCK-END
