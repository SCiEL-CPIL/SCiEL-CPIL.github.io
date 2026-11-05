"""Matrix multiplication 4x5x6 — recursive Strassen-style decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

# Precompute the full decomposition at import time to keep construct() fast.
def _build_decomposition(n, m, p):
    """Build a decomposition of the (n,m,p) matrix multiplication tensor.
    
    Returns (U, V, W) as lists of lists with the correct index conventions:
      U: shape (n*m, rank), U[i*m + j, c]
      V: shape (m*p, rank), V[j*p + k, c]
      W: shape (p*n, rank), W[k*n + i, c]
    
    Uses recursive block decomposition with Strassen for 2x2x2.
    """
    # Base case: 1x1x1
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    # Base case: 2x2x2 via Strassen
    if n == 2 and m == 2 and p == 2:
        # Strassen's 7 products for 2x2 * 2x2:
        # A = [[a, b],[c, d]], B = [[e, f],[g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # U maps (i,j) -> contributions to the 7 products
        # V maps (j,k) -> contributions to the 7 products
        # W maps (k,i) -> contributions to the 7 products
        
        # Product c corresponds to: sum over (i,j,k) of U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] * A[i,j]*B[j,k]
        # We need: for each (i,j,k) in C, the product U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = 1 if (i,j,k) is in that product's support, 0 otherwise.
        
        # Let's define the 7 products with their (i,j,k) supports:
        # M1: (a+d)(e+h) = a*e + a*h + d*e + d*h -> (0,0,0),(0,1,1),(1,0,0),(1,1,1)
        # M2: (c+d)e = c*e + d*e -> (1,0,0),(1,1,0)
        # M3: a(f-h) = a*f - a*h -> (0,1,0),(0,1,1) with signs
        # M4: d(g-e) = d*g - d*e -> (1,1,0),(1,0,0) with signs
        # M5: (a+b)h = a*h + b*h -> (0,1,1),(1,1,1)
        # M6: (c-a)(e+f) = c*e + c*f - a*e - a*f -> (1,0,0),(1,1,0),(1,0,1),(0,0,1) with signs
        # M7: (b-d)(g+h) = b*g + b*h - d*g - d*h -> (1,1,0),(1,1,1),(1,0,0),(1,0,1) with signs
        
        # Actually, let me use a cleaner formulation. For Strassen, the tensor decomposition has rank 7.
        # The rank-1 terms are:
        # t1 = (e0+e1) ⊗ (f0+f1) ⊗ (g0+g1)  where ei are row indices, fj are col-of-A indices... 
        # Wait, let me use the standard notation.
        
        # A is 2x2: A[i,j] for i,j in {0,1}
        # B is 2x2: B[j,k] for j,k in {0,1}
        # C[i,k] = sum_j A[i,j]*B[j,k]
        
        # The tensor T[i,j,k] = 1 (it's the matrix multiplication tensor, every entry is 1).
        # We want to decompose T as sum_c u_c[i] * v_c[j] * w_c[k] where:
        #   u_c is a vector of length 2 (indexed by i)
        #   v_c is a vector of length 2 (indexed by j)  
        #   w_c is a vector of length 2 (indexed by k)
        # But wait, in our convention:
        #   U has shape (n*m, rank) = (4, rank), indexed by (i,j)
        #   V has shape (m*p, rank) = (4, rank), indexed by (j,k)
        #   W has shape (p*n, rank) = (4, rank), indexed by (k,i)
        
        # So U[i*m+j, c] is the coefficient for A[i,j] in product c
        # V[j*p+k, c] is the coefficient for B[j,k] in product c
        # W[k*n+i, c] is the coefficient for C[i,k] in product c
        
        # For the decomposition to work: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
        
        # Strassen decomposition (standard):
        # Let me define 7 rank-1 terms. Each term is a product of three "linear forms".
        # Term c: U_c[i,j] * V_c[j,k] * W_c[k,i]
        
        # I'll use the following well-known Strassen tensors:
        # (using i,j,k in {0,1})
        
        # Let me just hardcode the 7 terms as 3D arrays.
        terms = []
        # t1: (1,1) ⊗ (1,1) ⊗ (1,1)  => U[i,j]=1, V[j,k]=1, W[k,i]=1 ... no that's rank 1 for the whole thing
        # 
        # Actually for Strassen, the 7 multiplications are:
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        
        # For each M_c, I need to express it as sum over (i,j,k) of U[i,j]*V[j,k]*W[k,i] * A[i,j]*B[j,k]
        # This means: U[i,j]*V[j,k]*W[k,i] should equal the coefficient of A[i,j]*B[j,k] in M_c.
        
        # M1: coefficients: A[0,0]*B[0,0]=1, A[0,0]*B[1,1]=1, A[1,1]*B[0,0]=1, A[1,1]*B[1,1]=1
        #   So we need U*V*W to be 1 at (0,0,0),(0,1,1),(1,0,0),(1,1,1) and 0 elsewhere.
        #   One way: U = [[1,0],[0,1]] (U[0,0]=1,U[1,1]=1), V = [[1,1],[1,1]] (all 1s), W = [[1,0],[0,1]]
        #   Check: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
        #          U[0,0]*V[0,1]*W[1,0] = 1*1*0 = 0 ✗ (should be 0, good)
        #          Wait, W[k,i]: W[0,0] means k=0,i=0. W[1,0] means k=1,i=0.
        #          U[0,0]*V[0,1]*W[1,0]: this is for (i=0,j=0,k=1): A[0,0]*B[0,1]. Coefficient should be 0.
        #          With U[0,0]=1, V[0,1]=1, W[1,0]=0: product = 0. ✓
        #          (i=0,j=1,k=0): U[0,1]*V[1,0]*W[0,0] = 0*1*1 = 0. Should be 0. ✓
        #          (i=0,j=1,k=1): U[0,1]*V[1,1]*W[1,0] = 0*1*0 = 0. Should be 1. ✗
        #   Hmm, that doesn't work. Let me think differently.
        
        # The issue is that for M1, the pattern is diagonal in (i,k) with all j's involved.
        # Let me use: U[i,j] = δ_{i,0}+δ_{i,1} for j=0,1? No...
        
        # Actually, I think the cleanest way is to represent each Strassen term as a 3-tensor
        # and verify. Let me use a different approach: just enumerate all possible simple decompositions
        # for 2x2x2 and verify.
        
        # Alternative: use the known rank-7 decomposition from the literature.
        # Let me define the 7 terms explicitly as 4-element vectors for U (indexed by i*2+j),
        # 4-element vectors for V (indexed by j*2+k), and 4-element vectors for W (indexed by k*2+i).
        
        # Term 1: U=[1,0,0,1], V=[1,1,1,1], W=[1,0,0,1]
        #   (i,j): (0,0)->1, (0,1)->0, (1,0)->0, (1,1)->1
        #   (j,k): all 1
        #   (k,i): (0,0)->1, (0,1)->0, (1,0)->0, (1,1)->1
        #   Product at (i,j,k): U[i*2+j]*V[j*2+k]*W[k*2+i]
        #   (0,0,0): 1*1*1=1
        #   (0,0,1): 1*1*0=0
        #   (0,1,0): 0*1*1=0
        #   (0,1,1): 0*1*0=0
        #   (1,0,0): 0*1*1=0
        #   (1,0,1): 0*1*0=0
        #   (1,1,0): 1*1*0=0
        #   (1,1,1): 1*1*1=1
        #   Sum over c of products should give 1 for all 8 (i,j,k). So far we have 1 at (0,0,0) and (1,1,1).
        
        # This is getting complicated. Let me use a brute-force verification approach instead.
        # I'll define candidate decompositions and verify them numerically.
        
        # Let me just use a known correct Strassen decomposition.
        # From standard references, the 7 products and their contributions:
        
        def verify_2x2x2(U, V, W):
            """Verify that sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] == 1 for all i,j,k in {0,1}."""
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        s = 0
                        for c in range(len(U[0])):
                            s += U[i*2+j][c] * V[j*2+k][c] * W[k*2+i][c]
                        if abs(s - 1) > 1e-10:
                            return False
            return True
        
        # Try the standard Strassen decomposition
        # U columns (for i*2+j): each column is [U(0,0), U(0,1), U(1,0), U(1,1)]
        # V columns (for j*2+k): each column is [V(0,0), V(0,1), V(1,0), V(1,1)]
        # W columns (for k*2+i): each column is [W(0,0), W(0,1), W(1,0), W(1,1)]
        
        # Standard Strassen (one valid decomposition):
        U_strassen = [
            [1,  0, 1, 0, 1, -1, 0],  # (0,0)
            [0,  0, 0, 0, 1,  0, 1],  # (0,1)
            [0,  1, 0, 0, 0,  1, 0],  # (1,0)
            [1,  1, 0, 1, 0,  0, -1], # (1,1)
        ]
        V_strassen = [
            [1,  1, 1, 0, 1,  1, 0],  # (0,0)
            [1,  0, -1, 0, 1,  1, 1], # (0,1)
            [1,  0, 0, -1, 0,  0, 0], # (1,0)
            [1,  0, 0, 1, 1,  0, 1],  # (1,1)
        ]
        W_strassen = [
            [1,  0, 0, 1, 0,  0, 1],  # (0,0)
            [0,  0, 0, 0, 1,  0, 0],  # (0,1)
            [1,  1, 1, 0, 0,  1, 0],  # (1,0)
            [1,  0, 0, -1, 0,  0, -1],# (1,1)
        ]
        
        if not verify_2x2x2(U_strassen, V_strassen, W_strassen):
            # Try to find a valid decomposition by solving
            # Actually, let me try another known decomposition
            U_strassen = [
                [1,  1, 1, 1, 1,  1, 1],
                [0,  0, 1, 0, 1,  0, 1],
                [0,  1, 0, 1, 0,  1, 0],
                [1,  1, 0, 1, 0,  0, 1],
            ]
            V_strassen = [
                [1,  1, 1, 1, 1,  1, 1],
                [1,  0, -1, 0, 1,  1, 1],
                [1,  0, 0, -1, 0,  0, 0],
                [1,  0, 0, 1, 1,  0, 1],
            ]
            W_strassen = [
                [1,  0, 0, 1, 0,  0, 1],
                [0,  0, 0, 0, 1,  0, 0],
                [1,  1, 1, 0, 0,  1, 0],
                [1,  0, 0, -1, 0,  0, -1],
            ]
            if not verify_2x2x2(U_strassen, V_strassen, W_strassen):
                # Fallback: schoolbook for 2x2x2 (rank 8)
                U_strassen = [[int(i*2+j == r*4+s*2+t) for r in range(2) for s in range(2) for t in range(2) for c in range(8)] for i in range(2) for j in range(2)]
                # This is getting too complex. Let me just use rank-8 schoolbook as fallback.
                rank_fb = 8
                U_fb = [[0]*rank_fb for _ in range(4)]
                V_fb = [[0]*rank_fb for _ in range(4)]
                W_fb = [[0]*rank_fb for _ in range(4)]
                col = 0
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            U_fb[i*2+j][col] = 1
                            V_fb[j*2+k][col] = 1
                            W_fb[k*2+i][col] = 1
                            col += 1
                return U_fb, V_fb, W_fb
        
        return U_strassen, V_strassen, W_strassen
    
    # For larger matrices, use block decomposition
    # Split n = n1 + n2, m = m1 + m2, p = p1 + p2
    # Choose splits as evenly as possible
    n1, n2 = n // 2, n - n // 2
    m1, m2 = m // 2, m - m // 2
    p1, p2 = p // 2, p - p // 2
    
    # Recursively build sub-decompositions
    # We need 7 block products (Strassen) or 8 (schoolbook)
    # For simplicity, use schoolbook (8 products) first, then try Strassen
    
    # Actually, for non-square blocks, Strassen doesn't directly apply.
    # Let's use a general block decomposition approach.
    
    # Simple approach: decompose into sub-problems and combine
    # For 4x5x6, we can split as:
    #   A is 4x5, B is 5x6
    #   Split rows of A into 2+2, cols of A / rows of B into 2+3, cols of B into 3+3
    #   This gives 8 sub-products
    
    # Let me use a recursive approach where we always split into two parts
    # and use 8 sub-products (schoolbook at the block level)
    
    # Sub-problems:
    # (n1,m1,p1), (n1,m1,p2), (n1,m2,p1), (n1,m2,p2)
    # (n2,m1,p1), (n2,m1,p2), (n2,m2,p1), (n2,m2,p2)
    
    sub_decomps = []
    for ni, mi, pi in [(n1,m1,p1),(n1,m1,p2),(n1,m2,p1),(n1,m2,p2),
                        (n2,m1,p1),(n2,m1,p2),(n2,m2,p1),(n2,m2,p2)]:
        if ni == 0 or mi == 0 or pi == 0:
            sub_decomps.append(([], [], []))  # empty
        else:
            sub_decomps.append(_build_decomposition(ni, mi, pi))
    
    # Now combine
    # Total rank = sum of ranks of sub-decomps
    total_rank = sum(len(sd[0][0]) if sd[0] else 0 for sd in sub_decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    row_offsets = [0, n1]  # row offsets for A blocks
    col_offsets_a = [0, m1]  # col offsets for A (= row offsets for B)
    col_offsets_b = [0, p1]  # col offsets for B
    
    for idx, (U_sub, V_sub, W_sub) in enumerate(sub_decomps):
        if not U_sub:
            continue
        sub_rank = len(U_sub[0])
        ni = n1 if idx < 4 else n2
        mi = m1 if idx % 4 < 2 else m2
        pi = p1 if idx % 2 == 0 else p2
        r_off = row_offsets[idx // 4]
        c_off_a = col_offsets_a[(idx % 4) // 2]
        c_off_b = col_offsets_b[idx % 2]
        
        for i in range(ni):
            for j in range(mi):
                for c in range(sub_rank):
                    U[(r_off+i)*m + (c_off_a+j)][col_offset+c] = U_sub[i*mi+j][c]
        
        for j in range(mi):
            for k in range(pi):
                for c in range(sub_rank):
                    V[(c_off_a+j)*p + (c_off_b+k)][col_offset+c] = V_sub[j*pi+k][c]
        
        for k in range(pi):
            for i in range(ni):
                for c in range(sub_rank):
                    W[(c_off_b+k)*n + (r_off+i)][col_offset+c] = W_sub[k*ni+i][c]
        
        col_offset += sub_rank
    
    return U, V, W

# Build the decomposition
_U, _V, _W = _build_decomposition(4, 5, 6)
_RANK = len(_U[0]) if _U else 120

def construct():
    """Return the tensor decomposition as {'U': U, 'V': V, 'W': W}."""
    return {"U": _U, "V": _V, "W": _W}
# EVOLVE-BLOCK-END
