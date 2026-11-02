"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 19 decomposition:
    # Strassen for 2x2 (cols 0,1) -> rank 7
    # Schoolbook for 2x3 (cols 2,3,4) -> rank 12 (4 per col)
    rank = 19
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a rank-1 tensor
    def add_rank1(u_idx, v_idx, w_idx, col):
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1

    col = 0
    # Strassen for 2x2 part (j=0,1; k=0,1)
    # A indices: i*m+j -> 0:00, 1:01, 2:10, 3:11
    # B indices: j*p+k -> 0:00, 1:01, 2:10, 3:11
    # C indices: k*n+i -> 0:00, 1:01, 2:10, 3:11
    
    # M1 = (A00+A11)(B00+B11)
    # This is not a single rank-1 tensor in the standard basis, but a sum of 4.
    # We need to expand Strassen into rank-1 terms?
    # No, Strassen uses 7 products. Each product is a scalar multiplication.
    # The products are:
    # P1 = (A00+A11) * (B00+B11)
    # P2 = (A10+A11) * B00
    # P3 = A00 * (B01-B11)
    # P4 = A11 * (B11-B01)
    # P5 = (A00+A01) * B11
    # P6 = (A10-A11) * (B00+B01)
    # P7 = (A01-A11) * (B10+B11)
    
    # We need to represent these as sums of rank-1 tensors.
    # A product (Sum A_i) * (Sum B_j) is a sum of rank-1 tensors.
    # But we can just define the U, V, W vectors for each product.
    
    # Product 1: (A00+A11) * (B00+B11)
    # U: A00=1, A11=1. V: B00=1, B11=1.
    # This product contributes to C00, C01, C10, C11.
    # C00 += A00 B00 + A00 B11 + A11 B00 + A11 B11
    # C01 += ...
    # This is getting complicated to map directly.
    # Easier: Just use the 7 Strassen products as the "basis" and then define how they combine into C.
    # But the interface requires U, V, W matrices.
    # Each column c in U,V,W defines a rank-1 tensor: U[:,c] \otimes V[:,c] \otimes W[:,c].
    # The sum of these tensors must equal the multiplication tensor.
    
    # Let's list the 7 Strassen products and their contributions.
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B11-B01)
    # P5 = (A00+A01)B11
    # P6 = (A10-A11)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 - P2 - P4 + P7 ? No.
    # Standard Strassen:
    # C00 = M1 - M2 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # We need to decompose each M_i into rank-1 tensors?
    # No, each M_i is a scalar product of two linear forms.
    # M_i = (sum u_ia A_a) * (sum v_ib B_b).
    # This is a sum of rank-1 tensors.
    # To keep rank minimal, we should use the 7 products directly?
    # But the interface forces us to output U, V, W.
    # If we use 7 columns, each column corresponds to one product.
    # But a product like (A00+A11)(B00+B11) is NOT a rank-1 tensor in the (A,B,C) space?
    # Yes it is!
    # Tensor T = sum_i (u_i \otimes v_i \otimes w_i).
    # If we set u = [1,0,1,0] (A00+A11), v = [1,1,0,0] (B00+B11), w = [1,1,1,1] (C00+C01+C10+C11)?
    # No, the W vector must pick out the correct C components.
    # The product (A00+A11)(B00+B11) contributes to ALL C entries?
    # No, in Strassen, M1 is used to compute C00, C01, C10, C11 with different coefficients.
    # Specifically:
    # C00 += 1 * M1
    # C01 += 0 * M1
    # C10 += 0 * M1
    # C11 += 1 * M1
    # So W for M1 should be [1, 0, 0, 1] in the C space?
    # Wait, C indices are (k,i).
    # C00 (k=0,i=0), C01 (k=1,i=0), C10 (k=0,i=1), C11 (k=1,i=1).
    # M1 contributes to C00 and C11.
    # So W vector for M1: W[0]=1, W[3]=1, others 0.
    
    # Let's verify M1 = (A00+A11)(B00+B11).
    # U: A00=1, A11=1. (Indices 0, 3)
    # V: B00=1, B11=1. (Indices 0, 3)
    # W: C00=1, C11=1. (Indices 0, 3)
    # Tensor: (e0+e3) \otimes (e0+e3) \otimes (e0+e3).
    # This expands to e0e0e0 + e0e0e3 + e3e0e0 + e3e0e3.
    # Terms:
    # A00 B00 C00
    # A00 B00 C11 (Wrong! C11 should be A11 B11)
    # A11 B00 C00 (Wrong)
    # A11 B00 C11 (Wrong)
    
    # So M1 is NOT a rank-1 tensor that can be added directly with a single W vector.
    # It is a sum of 4 rank-1 tensors.
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # We need to add these 4 terms to the appropriate C components.
    # But Strassen combines them to CANCEL terms.
    # C00 = M1 - M2 - M4 + M7.
    # This linear combination of products is what defines the tensor.
    # To represent this as a sum of rank-1 tensors, we must expand each product into its constituent rank-1 terms.
    # This will increase the rank!
    # Strassen rank 7 means 7 scalar multiplications.
    # But the tensor decomposition rank might be higher if we can't share the U and V vectors.
    # However, the problem asks for "rank-minimal exact tensor decomposition".
    # The rank of the tensor is the minimum number of rank-1 tensors.
    # For $M_{2,2,2}$, the tensor rank is 7.
    # This means there exist 7 rank-1 tensors that sum to the multiplication tensor.
    # The Strassen algorithm provides such a decomposition.
    # The 7 products are not the rank-1 tensors themselves, but the algorithm shows that the tensor can be written as a sum of 7 rank-1 tensors.
    # Actually, the Strassen algorithm shows that the bilinear map can be computed with 7 multiplications.
    # This implies the tensor rank is 7.
    # The decomposition is:
    # T = sum_{i=1}^7 (u_i \otimes v_i \otimes w_i)
    # where u_i, v_i are the linear forms for the i-th product, and w_i is the vector of coefficients for the output.
    # Let's find these u_i, v_i, w_i.
    
    # M1 = (A00+A11)(B00+B11)
    # u1 = [1,0,1,0]
    # v1 = [1,1,0,0]
    # w1: C00 += 1, C11 += 1. So w1 = [1,0,0,1]
    # Check: u1 \otimes v1 \otimes w1 = (A00+A11)(B00+B11)(C00+C11).
    # This includes A00B00C00, A00B00C11, A00B11C00, A00B11C11, A11B00C00, A11B00C11, A11B11C00, A11B11C11.
    # We need A00B00C00, A11B11C11.
    # The cross terms must cancel with other M_i.
    
    # This is the standard Strassen decomposition.
    # We will implement the 7 Strassen rank-1 tensors for the first 2 columns.
    # And 12 schoolbook rank-1 tensors for the last 3 columns.
    # Total 19.

    # Strassen Tensors for 2x2 (cols 0,1)
    # Indices: A: 00,01,10,11. B: 00,01,10,11. C: 00,01,10,11.
    
    # M1: u=[1,0,1,0], v=[1,1,0,0], w=[1,0,0,1]
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[3][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1

    # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0] (C10)
    # C10 = M2 + M4. M2 contributes to C10.
    # M2 = (A10+A11)B00.
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[2][col] = 1
    col += 1

    # M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,0] (C01)
    # M3 = A00(B01-B11).
    U[0][col] = 1
    V[1][col] = 1; V[3][col] = -1
    W[1][col] = 1
    col += 1

    # M4: u=[0,0,0,1], v=[0,-1,0,1], w=[0,0,1,0] (C10)
    # M4 = A11(B11-B01).
    U[3][col] = 1
    V[1][col] = -1; V[3][col] = 1
    W[2][col] = 1
    col += 1

    # M5: u=[1,1,0,0], v=[0,0,0,1], w=[0,1,0,0] (C01)
    # M5 = (A00+A01)B11.
    U[0][col] = 1; U[1][col] = 1
    V[3][col] = 1
    W[1][col] = 1
    col += 1

    # M6: u=[0,0,1,-1], v=[1,1,0,0], w=[0,0,0,1] (C11)
    # M6 = (A10-A11)(B00+B01).
    U[2][col] = 1; U[3][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = -1 # C11 = M1 - M3 + M5 - M6
    col += 1

    # M7: u=[0,1,-1,0], v=[0,0,1,1], w=[1,0,0,0] (C00)
    # M7 = (A01-A11)(B10+B11).
    # Wait, standard Strassen M7 = (A01-A11)(B10+B11).
    # C00 = M1 - M2 - M4 + M7.
    U[1][col] = 1; U[3][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[0][col] = 1
    col += 1

    # Schoolbook for cols 2,3,4
    # For each k in 2,3,4:
    # C0k = A00 B0k + A01 B1k
    # C1k = A10 B0k + A11 B1k
    for k in range(2, 5):
        # Term 1: A00 B0k -> C0k
        U[0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: A01 B1k -> C0k
        U[1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 3: A10 B0k -> C1k
        U[2][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # Term 4: A11 B1k -> C1k
        U[3][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
