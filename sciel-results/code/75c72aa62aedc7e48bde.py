"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 120s kill

def _elapsed():
    return time.time() - _START_TIME

def _schoolbook_decomposition(n=2, m=3, p=6):
    """Standard rank-nmp decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, r] = 1.0
                V[j*p + k, r] = 1.0
                W[k*n + i, r] = 1.0
                r += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=2, m=3, p=6):
    """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _strassen_222_block():
    """
    Return 7 Strassen terms for J_{2,2,2} embedded in J_{2,3,6}.
    Each term is (u_dict, v_dict, w_dict) with keys (i,j), (j,k), (k,i).
    i in {0,1}, j in {0,1}, k in {0,1}.
    """
    terms = []
    # p1 = (a+d)(e+h) -> C[0,0]+=p1, C[1,1]+=p1
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    # p2 = (c+d)e -> C[0,1]+=p2, C[1,0]+=p2  ... wait let me re-derive
    # Standard Strassen for C = A*B where A,B are 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C[0,0] = ae+bg, C[0,1] = af+bh, C[1,0] = ce+dg, C[1,1] = cf+dh
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    
    # But our convention is C[k,i] = sum_j A[i,j]*B[j,k], i.e., C = B^T A
    # So C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  (i=0: sum over j of A[0,j]*B[j,0])
    # C[0,1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]  (i=1: sum over j of A[1,j]*B[j,0])
    # C[1,0] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  (i=0: sum over j of A[0,j]*B[j,1])
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]  (i=1: sum over j of A[1,j]*B[j,1])
    
    # This is C[k,i] = (B^T A)[k,i] = sum_j B[k,j] A[j,i] ... no wait.
    # C[k,i] = sum_j A[i,j] B[j,k]
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,0] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    
    # Let me map to standard 2x2 multiplication by redefining.
    # Let A' = A^T (so A'[j,i] = A[i,j]) and B' = B (B'[j,k] = B[j,k])
    # Then (A'B')[i,k] = sum_j A'[i,j] B'[j,k] = sum_j A[j,i] B[j,k]
    # But we want C[k,i] = sum_j A[i,j] B[j,k]
    # This doesn't directly map to standard.
    
    # Alternative: just directly enumerate what each Strassen-like combination gives.
    # Actually, for the tensor decomposition, we don't need to match standard Strassen exactly.
    # We just need 7 rank-1 terms whose sum equals the tensor.
    # Let me use a different approach: solve for a rank-7 decomposition numerically.
    
    # For now, let me just use the schoolbook approach for correctness.
    # The key insight is that we can use the direct sum structure.
    # J_{2,3,6} = J_{2,3,3} ⊕ J_{2,3,3}
    # J_{2,3,3} = J_{2,2,3} ⊕ J_{2,1,3}
    # J_{2,2,3} = J_{2,2,2} ⊕ J_{2,2,1}
    # J_{2,1,3} is schoolbook rank 6
    # J_{2,2,1} is schoolbook rank 4
    # J_{2,2,2} can be done in rank 7 (Strassen) or rank 8 (schoolbook)
    
    # If we use rank 7 for J_{2,2,2}, total = 2*(7+4+6) = 34
    # If we use rank 8, total = 2*(8+4+6) = 36
    
    # Let me just use schoolbook for everything to guarantee correctness.
    return None

def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition
    for 2x3 by 3x6 matrix multiplication.
    """
    n, m, p = 2, 3, 6
    nm, mp, pn = n*m, m*p, p*n
    
    # Deadline guard
    if _elapsed() > TIME_LIMIT:
        return _schoolbook_decomposition(n, m, p)
    
    # Try rank-34 construction using Strassen for the 2x2x2 blocks
    # Structure: J_{2,3,6} = (J_{2,2,2} ⊕ J_{2,2,1} ⊕ J_{2,1,3}) ⊕ (J_{2,2,2} ⊕ J_{2,2,1} ⊕ J_{2,1,3})
    # where the second copy has k offset by 3
    
    def build_block_terms(k_offset=0):
        """Build 17 terms for J_{2,3,3} with k values in {k_offset, k_offset+1, k_offset+2}."""
        terms = []
        
        # J_{2,2,2}: i in {0,1}, j in {0,1}, k in {k_offset, k_offset+1}
        # Use Strassen's 7 terms for 2x2x2
        # For the sub-problem, A[i,j] for i in {0,1}, j in {0,1}
        # B[j,k] for j in {0,1}, k in {k_offset, k_offset+1}
        # C[k,i] for k in {k_offset, k_offset+1}, i in {0,1}
        
        # Let me define local indices: i_loc = i, j_loc = j, k_loc = k - k_offset
        # A[i_loc, j_loc], B[j_loc, k_loc], C[k_loc, i_loc]
        
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[0,1] = A[1,0]B[0,0] + A[1,1]B[1,0]
        # C[1,0] = A[0,0]B[0,1] + A[0,1]B[1,1]
        # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
        
        # Strassen terms (adapted):
        # p1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # p2 = (A[1,0]+A[1,1])B[0,0]
        # p3 = A[0,0](B[0,1]-B[1,1])
        # p4 = A[1,1](B[1,0]-B[0,0])
        # p5 = (A[0,0]+A[0,1])B[1,1]
        # p6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        # p7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        
        # Verify:
        # C[0,0] = p1 + p4 - p5 + p7?
        # p1 contributes: A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
        # p4 contributes: A[1,1]B[1,0] - A[1,1]B[0,0]
        # -p5 contributes: -A[0,0]B[1,1] - A[0,1]B[1,1]
        # p7 contributes: A[0,1]B[1,0] + A[0,1]B[1,1] - A[1,1]B[1,0] - A[1,1]B[1,1]
        # Sum: A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
        #       + A[1,1]B[1,0] - A[1,1]B[0,0]
        #       - A[0,0]B[1,1] - A[0,1]B[1,1]
        #       + A[0,1]B[1,0] + A[0,1]B[1,1] - A[1,1]B[1,0] - A[1,1]B[1,1]
        # = A[0,0]B[0,0] + A[0,1]B[1,0] + (A[0,0]B[1,1]-A[0,0]B[1,1]) + (A[1,1]B[0,0]-A[1,1]B[0,0])
        #   + (A[1,1]B[1,1]-A[1,1]B[1,1]) + (A[1,1]B[1,0]-A[1,1]B[1,0]) + (-A[0,1]B[1,1]+A[0,1]B[1,1])
        # = A[0,0]B[0,0] + A[0,1]B[1,0] ✓
        
        # C[0,1] = p3 + p5?
        # p3: A[0,0]B[0,1] - A[0,0]B[1,1]
        # p5: A[0,0]B[1,1] + A[0,1]B[1,1]
        # Sum: A[0,0]B[0,1] + A[0,1]B[1,1] ✓
        
        # C[1,0] = p2 + p4?
        # p2: A[1,0]B[0,0] + A[1,1]B[0,0]
        # p4: A[1,1]B[1,0] - A[1,1]B[0,0]
        # Sum: A[1,0]B[0,0] + A[1,1]B[1,0] ✓
        
        # C[1,1] = p1 - p2 + p3 + p6?
        # p1: A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
        # -p2: -A[1,0]B[0,0] - A[1,1]B[0,0]
        # p3: A[0,0]B[0,1] - A[0,0]B[1,1]
        # p6: A[1,0]B[0,0] + A[1,0]B[0,1] - A[0,0]B[0,0] - A[0,0]B[0,1]
        # Sum: (A[0,0]B[0,0]-A[0,0]B[0,0]) + (A[1,1]B[0,0]-A[1,1]B[0,0]) + (-A[1,0]B[0,0]+A[1,0]B[0,0])
        #      + (A[0,0]B[1,1]-A[0,0]B[1,1]) + A[1,1]B[1,1] + (A[0,0]B[0,1]-A[0,0]B[0,1]) + A[1,0]B[0,1]
        # = A[1,1]B[1,1] + A[1,0]B[0,1] ✓
        
        # So the Strassen terms work. Now map to global indices.
        # Term 1: (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        terms.append(({(0,0):1, (1,1):1}, {(0,k_offset):1, (1,k_offset+1):1}, {(k_offset,0):1, (k_offset+1,1):1}))
        # Term 2: (A[1,0]+A[1,1])B[0,0]
        terms.append(({(1,0):1, (1,1):1}, {(0,k_offset):1}, {(k_offset,1):1}))
        # Term 3: A[0,0](B[0,1]-B[1,1])
        terms.append(({(0,0):1}, {(0,k_offset+1):1, (1,k_offset+1):-1}, {(k_offset+1,0):1}))
        # Term 4: A[1,1](B[1,0]-B[0,0])
        terms.append(({(1,1):1}, {(1,k_offset):1, (0,k_offset):-1}, {(k_offset,1):1}))
        # Term 5: (A[0,0]+A[0,1])B[1,1]
        terms.append(({(0,0):1, (0,1):1}, {(1,k_offset+1):1}, {(k_offset+1,0):1}))
        # Term 6: (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        terms.append(({(1,0):1, (0,0):-1}, {(0,k_offset):1, (0,k_offset+1):1}, {(k_offset+1,1):1}))
        # Term 7: (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        terms.append(({(0,1):1, (1,1):-1}, {(1,k_offset):1, (1,k_offset+1):1}, {(k_offset,0):1}))
        
        # J_{2,2,1}: i in {0,1}, j in {0,1}, k = k_offset+2
        # C[k_offset+2, i] = sum_j A[i,j] B[j, k_offset+2]
        for i in range(2):
            for j in range(2):
                terms.append(({(i,j):1}, {(j, k_offset+2):1}, {(k_offset+2, i):1}))
        
        # J_{2,1,3}: i in {0,1}, j=2, k in {k_offset, k_offset+1, k_offset+2}
        # C[k, i] = A[i,2] B[2,k]
        for i in range(2):
            for kk in range(3):
                terms.append(({(i,2):1}, {(2, k_offset+kk):1}, {(k_offset+kk, i):1}))
        
        return terms
    
    if _elapsed() > TIME_LIMIT:
        return _schoolbook_decomposition(n, m, p)
    
    # Build all 34 terms
    all_terms = []
    all_terms.extend(build_block_terms(0))
    all_terms.extend(build_block_terms(3))
    
    rank = len(all_terms)
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for (i, j), coeff in u_dict.items():
            U[i*m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j*p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k*n + i, r] = coeff
    
    # Verify
    if _elapsed() > TIME_LIMIT - 5:
        # Not enough time to verify carefully, return schoolbook to be safe
        return _schoolbook_decomposition(n, m, p)
    
    if _verify(U, V, W, n, m, p):
        # Success! Rank 34
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook
        return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
