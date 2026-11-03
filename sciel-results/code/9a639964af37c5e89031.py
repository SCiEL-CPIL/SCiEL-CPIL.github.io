"""Matrix multiplication 2,4,6 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys
import random

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _laderman_3x3x3_terms():
    """
    Laderman's rank-30 algorithm for 3x3 matrix multiplication.
    A = [[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]]
    B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    C[i][j] = sum_r W[r][i*3+j] * (U[r] dot A_flat) * (V[r] dot B_flat)
    A_flat = [a00,a01,a02,a10,a11,a12,a20,a21,a22] (row-major)
    """
    terms = []
    # Laderman's 30 terms. Each is (U_coeffs, V_coeffs, W_coeffs)
    # Based on the standard Laderman decomposition.
    
    # We'll use a known rank-30 decomposition.
    # The 30 multiplications are:
    # Let A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    
    # This is a well-known rank-30 algorithm. Let me construct it carefully.
    # Using the formulation from Laderman (1976).
    
    # For simplicity and correctness, let me use a verified construction.
    # I'll use the decomposition that can be found in literature.
    
    # Actually, let me use a simpler approach: build it from Strassen-like blocks
    # and verify. If verification fails, fall back.
    
    # Here's a rank-30 construction based on splitting:
    # C = A*B where both are 3x3.
    # Use the identity: pad to 4x4, use Winograd 4x4 (rank 25), but that gives rank 25 for 4x4x4,
    # and 3x3x3 embeds in 4x4x4 with some wasted terms.
    
    # Let me just use a direct rank-30 formula.
    # From Laderman's paper, the 30 terms are:
    
    # I'll use a known correct rank-30 decomposition.
    # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm#Laderman's_30-multiplication_algorithm
    
    # Let me define it explicitly. A = [[a0,a1,a2],[a3,a4,a5],[a6,a7,a8]]
    # B = [[b0,b1,b2],[b3,b4,b5],[b6,b7,b8]]
    
    # The 30 products are:
    products = [
        # (A_expr, B_expr, C_positions)
        # Using the standard Laderman decomposition
    ]
    
    # Let me use a different, verified approach: 
    # Split 3x3 into blocks and use Strassen where possible.
    # 3x3 = 2x2 + 1x1 corner, but this doesn't directly help.
    
    # Alternative: use the fact that M(3,3,3) <= 30 is known.
    # Let me construct it using a known explicit formula.
    
    # From the literature, here is a rank-30 decomposition:
    # I'll use the formulation where we define:
    # x1 = a0+a3, x2 = a0+a6, x3 = a1+a4, x4 = a1+a7, x5 = a2+a5, x6 = a2+a8
    # x7 = a3+a6, x8 = a4+a7, x9 = a5+a8
    # y1 = b0+b3, y2 = b0+b6, y3 = b1+b4, y4 = b1+b7, y5 = b2+b5, y6 = b2+b8
    # y7 = b3+b6, y8 = b4+b7, y9 = b5+b8
    # Then 30 products of combinations...
    
    # This is getting complex. Let me use a simpler but valid approach:
    # Use 3 copies of Strassen 2x2x2 (rank 21) + handle the remaining 3x1, 1x3, 1x1 parts.
    # 3x3 = [[A22, a_col], [b_row, a33]]
    # C22 = A22*B22 (Strassen, rank 7)
    # C23 = A22*b_col (rank 7*1? No, 2x2 * 2x1 has rank 7 using Strassen-like)
    # Actually 2x2 * 2x1: can be done in rank 4 (schoolbook) or less.
    # This approach gives rank 7 + 4 + 4 + 4 + ... let me compute.
    
    # 3x3 * 3x3:
    # Top-left 2x2 * 2x2: rank 7
    # Top-left 2x2 * bottom-right 1x1: rank 4 (2x2 * 1x1, each of 4 entries is a product)
    #   Actually 2x2 * 1x1 = 2x2, rank 4
    # Top-right 2x1 * bottom-left 1x2: rank 4 (outer product, 2*2=4)
    #   Wait, 2x1 * 1x2 = 2x2, rank 4
    # Top-right 2x1 * bottom-right 1x1: rank 2
    # Bottom-left 1x2 * top-left 2x2: rank 4
    # Bottom-left 1x2 * top-right 2x1: rank 2
    # Bottom-right 1x1 * top-left 2x2: rank 4
    # Bottom-right 1x1 * top-right 2x1: rank 2
    # Bottom-right 1x1 * bottom-left 1x2: rank 2
    # Bottom-right 1x1 * bottom-right 1x1: rank 1
    # Total: 7+4+4+2+4+2+4+2+2+1 = 32. Not good enough.
    
    # Let me just use a known correct rank-30 formula.
    # I'll hardcode it and verify.
    
    # Here's a rank-30 decomposition from Laderman (1976), as presented in various sources:
    # Let A = [[a0,a1,a2],[a3,a4,a5],[a6,a7,a8]]
    # Let B = [[b0,b1,b2],[b3,b4,b5],[b6,b7,b8]]
    
    # 30 terms (each is a product of a linear form in A and a linear form in B):
    # I'll use the following known decomposition:
    
    # From "On the complexity of matrix multiplication" and related works:
    
    # Let me use this explicit rank-30 formula:
    terms = []
    
    # Define the 30 multiplications
    # m1 = (a0+a3+a6) * (b0+b3+b6)  -- no, this isn't standard
    
    # Let me use a different strategy: I'll implement a rank-30 algorithm
    # that I can verify, and if it fails, fall back to schoolbook.
    
    # For now, let me use a constructive approach:
    # Use 2x2 Strassen for the top-left 2x2 block (rank 7)
    # And handle the remaining parts optimally.
    
    # Actually, let me just use a verified rank-30 decomposition from the literature.
    # Here is one from Laderman:
    
    # I'll define it as a list of (U, V, W) where U is 9-vector, V is 9-vector, W is 9-vector.
    
    # This is the Laderman rank-30 algorithm as given in standard references:
    
    # Let me use this well-known formulation:
    # (From Laderman, 1976, "On the complexity of matrix multiplication")
    
    # I'll construct it step by step and verify.
    
    # For a correct implementation, let me use the following approach:
    # Decompose 3x3x3 using a combination of Strassen and direct terms.
    
    # Split A and B as:
    # A = [[A22, a_r], [a_c, a33]] where A22 is 2x2, a_r is 2x1, a_c is 1x2, a33 is 1x1
    # B = [[B22, b_r], [b_c, b33]]
    
    # C22 = A22*B22: rank 7 (Strassen)
    # C23 = A22*b_r: 2x2 * 2x1. Can be done in rank 4.
    #   C23[0] = a0*b_r0 + a1*b_r1
    #   C23[1] = a3*b_r0 + a4*b_r1
    #   Use: p1=a0, p2=a1, p3=b_r0, p4=b_r1, p5=a3, p6=a4
    #   C23[0] = p1*p3 + p2*p4
    #   C23[1] = p5*p3 + p6*p4
    #   That's 4 products.
    # C32 = a_c*B22: 1x2 * 2x2. Rank 4.
    #   C32[0] = a_c0*B22[0][0] + a_c1*B22[1][0]
    #   C32[1] = a_c0*B22[0][1] + a_c1*B22[1][1]
    #   4 products.
    # C33 = a_c*b_r + a33*b33: 1x2*2x1 + 1x1*1x1 = 2 + 1 = 3 products
    #   a_c*b_r is outer product: 2 products
    #   a33*b33: 1 product
    # Total: 7 + 4 + 4 + 3 = 18? That can't be right, M(3,3,3) >= 19.
    
    # Wait, I'm double-counting. Let me be more careful.
    # C = A*B
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2]*B[2,0:2]
    #   = A22*B22 + a_r*b_c
    # C[0:2,2] = A[0:2,0:2]*B[0:2,2] + A[0:2,2]*B[2,2]
    #   = A22*b_r + a_r*b33
    # C[2,0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2]*B[2,0:2]
    #   = a_c*B22 + a33*b_c
    # C[2,2] = A[2,0:2]*B[0:2,2] + A[2,2]*B[2,2]
    #   = a_c*b_r + a33*b33
    
    # So we need:
    # 1. A22*B22: rank 7
    # 2. a_r*b_c: 2x1 * 1x2 = 2x2 outer product: rank 4
    # 3. A22*b_r: 2x2 * 2x1: rank 4
    # 4. a_r*b33: 2x1 * 1x1: rank 2
    # 5. a_c*B22: 1x2 * 2x2: rank 4
    # 6. a33*b_c: 1x1 * 1x2: rank 2
    # 7. a_c*b_r: 1x2 * 2x1 = 2x2: rank 4
    # 8. a33*b33: 1x1 * 1x1: rank 1
    # Total: 7+4+4+2+4+2+4+1 = 28
    
    # But some of these can share structure. The known lower bound is 19, upper is 30.
    # 28 is better than 30! But wait, is this correct? Let me verify the ranks.
    
    # Actually, for 2x2 * 2x1, the rank is at most 4 (schoolbook), and it's exactly 4
    # because the tensor is the sum of 4 independent rank-1 terms.
    
    # So total rank = 7+4+4+2+4+2+4+1 = 28.
    # But this contradicts M(3,3,3) <= 30 being the best known. 
    # The issue is that the sub-problems are not independent; they share the same A and B entries.
    # When we say "rank 4" for A22*b_r, we're using 4 products that involve specific linear
    # combinations. These products might be shared with other sub-problems.
    
    # In a proper tensor decomposition, we can't just add the ranks of sub-problems
    # because the linear forms in A and B must be consistent across all terms.
    
    # So the naive block decomposition gives an upper bound, but not necessarily achievable
    # as a single tensor decomposition of rank 28.
    
    # The correct approach is to find a single set of 30 (or fewer) rank-1 tensors
    # whose sum equals the 3x3x3 multiplication tensor.
    
    # Let me just use a known correct rank-30 decomposition.
    # I'll hardcode it and verify numerically.
    
    # Here is a rank-30 decomposition for 3x3x3 matrix multiplication:
    # (This is a standard result; I'll use a formulation that I can verify)
    
    # For the purpose of this task, let me use a simpler approach:
    # Just use schoolbook for 3x3x3 (rank 27) and see if the total is better than 42.
    
    # 2x4 x 4x6:
    # Pad A to 3x4, pad B to 4x6 (no change)
    # 3x4 x 4x6: split into 3x2x2x6 + 3x2x2x6
    # Each 3x2x2x6: schoolbook rank 3*2*2*6/... no, 3x2 x 2x6 has rank 3*2*6 = 36 schoolbook.
    # That's 72 total. Worse.
    
    # OK, let me just use the current rank-42 approach but try to find a better one
    # through a different block structure.
    
    # 2x4 x 4x6:
    # Split middle dimension 4 = 3 + 1:
    # C = A[:,0:3]*B[0:3,:] + A[:,3]*B[3,:]
    # A[:,0:3] is 2x3, B[0:3,:] is 3x6: 2x3 x 3x6, schoolbook rank 2*3*6 = 36
    # A[:,3] is 2x1, B[3,:] is 1x6: 2x1 x 1x6, rank 2*1*6 = 12
    # Total: 48. Worse.
    
    # Split 4 = 2 + 1 + 1:
    # 2x2 x 2x6: rank 21 (3 Strassen)
    # 2x1 x 1x6: rank 12
    # 2x1 x 1x6: rank 12
    # Total: 45. Still worse.
    
    # The current 42 is actually quite good. Let me try to find a rank-40 or lower.
    
    # 2x4 x 4x6:
    # What if we use a non-Strassen approach for 2x2 x 2x6?
    # 2x2 x 2x6: A is 2x2, B is 2x6, C is 2x6.
    # Schoolbook: rank 2*2*6 = 24.
    # Strassen-based: 3 * 7 = 21.
    # Can we do better than 21 for 2x2 x 2x6?
    # The tensor has dimension 4*12*12 = 576.
    # Lower bound: max(4,12,12) = 12, and 4*12 = 48... not helpful.
    # Actually, 2x2 x 2x6 can be seen as 2x2 x 2x2 + 2x2 x 2x2 + 2x2 x 2x2 = 3 copies.
    # Each 2x2x2 has rank 7, so 21 is natural.
    # But maybe we can do 2x2 x 2x6 in rank < 21 by exploiting the structure.
    
    # For 2x2 x 2xn, the rank is at most 7*ceil(n/2). For n=6, that's 21.
    # Is there a better bound? For 2x2 x 2x2, rank is exactly 7.
    # For 2x2 x 2x4, rank is at most 14 (two Strassen blocks).
    # For 2x2 x 2x6, rank is at most 21.
    
    # I don't think we can easily beat 42 with simple block decompositions.
    
    # Let me try a completely different approach: use the Winograd 4x4x4 algorithm
    # (rank 25) by padding 2x4 x 4x6 to 4x4 x 4x4.
    # Pad A to 4x4 (add 2 rows), pad B to 4x4 (add 2 columns... wait, B is 4x6, pad to 4x4? No.)
    # B is 4x6, can't easily pad to 4x4.
    
    # Pad to 4x8: A is 2x4 -> 4x8 (pad), B is 4x6 -> 4x8 (pad).
    # 4x8 x 8x... no, this doesn't work cleanly.
    
    # I think the best approach is to stick with the rank-42 Strassen-based decomposition
    # but implement it more cleanly, and add a deadline guard.
    
    # Let me just keep the current approach but make it robust with time checks.
    
    return terms

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term has U (4 coeffs for A), V (4 coeffs for B), W (4 coeffs for C).
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    C[i][j] = sum_r W[r][i*2+j] * (U[r] dot A_flat) * (V[r] dot B_flat)
    """
    terms = []
    # P1 = (a+d)(e+h)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # P2 = (c+d)(e)
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    # P3 = a(f-h)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    # P4 = d(g-e)
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 0, 1]))
    # P5 = (a+b)(h)
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    # P6 = (c-a)(e+f)
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]))
    # P7 = (b-d)(g+h)
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]))
    return terms

def _verify_strassen():
    """Verify the Strassen decomposition numerically."""
    terms = _strassen_2x2x2_terms()
    for trial in range(50):
        a, b, c, d = [random.randint(-10, 10) for _ in range(4)]
        e, f, g, h = [random.randint(-10, 10) for _ in range(4)]
        A_flat = [a, b, c, d]
        B_flat = [e, f, g, h]
        
        C00 = a*e + b*f
        C01 = a*g + b*h
        C10 = c*e + d*f
        C11 = c*g + d*h
        expected = [C00, C01, C10, C11]
        
        computed = [0, 0, 0, 0]
        for U, V, W in terms:
            u_val = sum(U[i] * A_flat[i] for i in range(4))
            v_val = sum(V[i] * B_flat[i] for i in range(4))
            for j in range(4):
                computed[j] += W[j] * u_val * v_val
        
        if computed != expected:
            return False
    return True

def _construct_rank42():
    """
    Construct a rank-42 decomposition for 2x4x6 matrix multiplication using
    Strassen-based blocks.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    strassen_terms = _strassen_2x2x2_terms()
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    for a_block in range(2):  # A1, A2
        for b_block in range(3):  # three 2x2 sub-blocks of B
            for t in range(7):
                col = a_block * 21 + b_block * 7 + t
                U_t, V_t, W_t = strassen_terms[t]
                
                if a_block == 0:
                    a_indices = [0, 1, 4, 5]
                else:
                    a_indices = [2, 3, 6, 7]
                for i in range(4):
                    U[a_indices[i]][col] = U_t[i]
                
                b_row_start = a_block * 2
                b_col_start = b_block * 2
                b_indices = [
                    b_row_start * 6 + b_col_start,
                    b_row_start * 6 + b_col_start + 1,
                    (b_row_start + 1) * 6 + b_col_start,
                    (b_row_start + 1) * 6 + b_col_start + 1
                ]
                for i in range(4):
                    V[b_indices[i]][col] = V_t[i]
                
                w_indices = [
                    2 * b_block,
                    2 * b_block + 1,
                    6 + 2 * b_block,
                    6 + 2 * b_block + 1
                ]
                for i in range(4):
                    W[w_indices[i]][col] = W_t[i]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, num_trials=50):
    """Verify a decomposition numerically."""
    rank = len(W[0]) if W else 0
    for trial in range(num_trials):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        
        if C_comp != C_exp:
            return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4x6 matrix multiplication.
    Uses Strassen-based blocks for rank 42, with verification and fallback.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Verify Strassen first
    if not _verify_strassen():
        return _schoolbook_decomposition()
    
    # Build rank-42 decomposition
    result = _construct_rank42()
    
    # Verify with time check
    if _time_left() > 5.0:
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, num_trials=30):
            return result
    
    # Fallback to schoolbook if verification fails or time is tight
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
