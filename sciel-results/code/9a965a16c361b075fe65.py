"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]
import time

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=2, m=2, p=8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[r]) != rank or len(V[r]) != rank or len(W[r]) != rank:
            return False
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # And for all other (i', j', k') where not (i'=i and j'=j and k'=k) or (j' != j), the sum should be 0
    # Actually the tensor T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k' else 0
    # So we need: for all i,j,k,i',j',k': sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r] = delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    # But that's O(n^2 * m^2 * p^2 * rank) which is 4*4*64*rank = 1024*rank checks
    # For rank ~28, that's ~28672 operations, fine.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[(i,j),(j,k),(i,k)] should be 1
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
                # Check that T[(i,j),(j',k),(i',k')] = 0 for j' != j or i' != i or k' != k
                # Actually we need to check all combinations, but let's just verify the full tensor
    # Full verification: check all (i,j,j',k,i',k') tuples
    for i in range(n):
        for j in range(m):
            for jp in range(m):  # j'
                for k in range(p):
                    for ip in range(n):  # i'
                        for kp in range(p):  # k'
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            s = 0
                            for r in range(rank):
                                s += U[i * m + j][r] * V[jp * p + k][r] * W[kp * n + ip][r]
                            if abs(s - expected) > 1e-9:
                                return False
    return True


def _strassen_2x2_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2.
    Returns lists of (U_row_coeffs, V_row_coeffs, W_row_coeffs) for each of the 7 terms.
    U is indexed by (i,j) -> i*2+j for A[i,j]
    V is indexed by (j,k) -> j*2+k for B[j,k]
    W is indexed by (k,i) -> k*2+i for C[k,i] (note: k*n+i convention)
    """
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M7
    # C10 = M6 + M2
    # C11 = M1 - M5 + M4 + M3
    #
    # In tensor form, each term c contributes:
    #   U_c[i,j] * V_c[j,k] * W_c[k,i] summed over j
    #   But actually the decomposition is:
    #   sum_c (linear_form_A_c) * (linear_form_B_c) * W_c[k,i]
    #   where W_c is a 2x2 matrix (or rather, indexed by (k,i))
    #
    # Let me define each term by its A-coefficients (4-vector: A00,A01,A10,A11),
    # B-coefficients (4-vector: B00,B01,B10,B11), and C-coefficients (4-vector: C00,C01,C10,C11)
    # where C is indexed as C[k,i] -> but wait, the output is C[i,k] with W indexed by (k*n+i).
    #
    # For the 2x2x2 case: W is indexed by (k, i) where k is column of C, i is row of C.
    # W[k*2+i, c] is the coefficient.
    #
    # Term 0 (M1): A: [1,0,0,1], B: [1,0,0,1], C: C00+=1, C11+=1 -> W[0*2+0]=1, W[1*2+1]=1
    #   W vector (indexed by k*2+i): k=0,i=0 -> 1; k=0,i=1 -> 0; k=1,i=0 -> 0; k=1,i=1 -> 1
    #   So W = [1, 0, 0, 1]
    #
    # Term 1 (M2): A: [0,0,1,1], B: [1,0,0,0], C: C00-=1, C10+=1 -> W[0*2+0]=-1, W[0*2+1]=1
    #   Wait, C10 means i=1, k=0. W[k*2+i] = W[0*2+1] = W[1]. And C00 means i=0,k=0, W[0*2+0]=W[0].
    #   So W = [-1, 1, 0, 0]
    #
    # Term 2 (M3): A: [1,0,0,0], B: [0,1,0,-1], C: C00+=1, C11+=1 -> W[0]=1, W[3]=1
    #   W = [1, 0, 0, 1]
    #
    # Term 3 (M4): A: [0,0,0,1], B: [-1,0,1,0], C: C00+=1, C11+=1 -> W[0]=1, W[3]=1
    #   W = [1, 0, 0, 1]
    #
    # Term 4 (M5): A: [1,1,0,0], B: [0,0,0,1], C: C01+=1, C11-=1 -> W[1*2+0]=W[2]=1, W[1*2+1]=W[3]=-1
    #   Wait, C01 means i=0, k=1. W[k*2+i] = W[1*2+0] = W[2]. C11 means i=1, k=1. W[1*2+1]=W[3].
    #   So W = [0, 0, 1, -1]
    #
    # Term 5 (M6): A: [-1,0,1,0], B: [1,1,0,0], C: C10+=1 -> W[0*2+1]=W[1]=1
    #   W = [0, 1, 0, 0]
    #
    # Term 6 (M7): A: [0,1,0,-1], B: [0,0,1,1], C: C01+=1 -> W[1*2+0]=W[2]=1
    #   W = [0, 0, 1, 0]
    
    terms = [
        # (A_coeffs[4], B_coeffs[4], W_coeffs[4])
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [-1, 1, 0, 0]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [1, 0, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 0, 1]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, -1]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, 1, 0]),   # M7
    ]
    return terms


def _build_strassen_2x2_2x8():
    """Build rank-28 decomposition: 4 groups of 2 columns, each using Strassen rank-7."""
    n, m, p = 2, 2, 8
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # (4, 28)
    V = [[0] * rank for _ in range(m * p)]   # (16, 28)
    W = [[0] * rank for _ in range(p * n)]   # (16, 28)
    
    strassen = _strassen_2x2_2x2()
    
    for group in range(4):
        base = group * 7
        for c in range(7):
            a_coeffs, b_coeffs, w_coeffs = strassen[c]
            # A coefficients: a_coeffs[i*2+j] for i,j in {0,1}
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][base + c] = a_coeffs[i * 2 + j]
            
            # B coefficients: for this group, columns 2*group and 2*group+1
            # b_coeffs[j*2+k'] where k' is 0 or 1 (local column index)
            for j in range(2):
                for kp in range(2):
                    k = 2 * group + kp
                    V[j * p + k][base + c] = b_coeffs[j * 2 + kp]
            
            # W coefficients: w_coeffs[k'*2+i] where k' is local (0 or 1), i in {0,1}
            for kp in range(2):
                k = 2 * group + kp
                for i in range(2):
                    W[k * n + i][base + c] = w_coeffs[kp * 2 + i]
    
    return U, V, W


def _build_schoolbook_2x2_2x8():
    """Build rank-32 schoolbook decomposition."""
    n, m, p = 2, 2, 8
    rank = n * m * p  # 32
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
    return U, V, W


def construct():
    """Construct the rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Try Strassen-based rank-28 first
    U, V, W = _build_strassen_2x2_2x8()
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank-32
    U, V, W = _build_schoolbook_2x2_2x8()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
