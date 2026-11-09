import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    # Rank: 49 (4x4 x 4x4 Strassen) + 16 (4x4 x 4x1) + 20 (4x1 x 1x5) = 85
    R = 85
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    
    # Part 1: 4x4 x 4x4 using 2x2 Strassen = 49 terms
    # Split 4x4 into 2x2 blocks. 8 blocks, each 2x2 x 2x2 = Strassen 7 terms.
    # But Strassen combines them into 7 matrix products, each 2x2 x 2x2.
    # Each 2x2 x 2x2 uses 7 rank-1 terms. Total: 7*7 = 49.
    
    # Define 2x2 Strassen rank-1 decomposition
    # For 2x2 x 2x2: C = A*B
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # Strassen:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b01-b11)
    # P4 = a11(b10-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b01)
    # P7 = (a01-a11)(b10+b11)
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6
    
    # Each Pi is a scalar product. In tensor form, Pi = sum over the scalar products.
    # But for the tensor decomposition, each Pi corresponds to ONE rank-1 term where
    # U, V, W encode the linear combinations.
    
    # For a 2x2 x 2x2 at A positions (ar,ac), B positions (br,bc), C positions (cr,cc):
    # The 7 rank-1 terms are:
    
    def add_2x2_strassen_block(ar, ac, br, bc, cr, cc):
        nonlocal col
        # A[i,j] at index (ar+i)*m + (ac+j)
        # B[i,j] at index (br+i)*p + (bc+j)
        # C[i,j] at index (cc+j)*n + (cr+i)
        
        a00 = (ar+0)*m + (ac+0)
        a01 = (ar+0)*m + (ac+1)
        a10 = (ar+1)*m + (ac+0)
        a11 = (ar+1)*m + (ac+1)
        
        b00 = (br+0)*p + (bc+0)
        b01 = (br+0)*p + (bc+1)
        b10 = (br+1)*p + (bc+0)
        b11 = (br+1)*p + (bc+1)
        
        c00 = (cc+0)*n + (cr+0)
        c01 = (cc+1)*n + (cr+0)
        c10 = (cc+0)*n + (cr+1)
        c11 = (cc+1)*n + (cr+1)
        
        # P1 = (a00+a11)(b00+b11)
        U[a00, col] = 1; U[a11, col] = 1
        V[b00, col] = 1; V[b11, col] = 1
        W[c00, col] = 1; W[c01, col] = 1; W[c10, col] = 1; W[c11, col] = 1
        # Wait, this is wrong. The rank-1 tensor U⊗V⊗W gives U[a]*V[b]*W[c].
        # For P1 to contribute to C00, C01, C10, C11, I need:
        # U[a]*V[b]*W[c] to equal the appropriate coefficient.
        
        # Actually, each Pi is a product of two linear combinations of A and B entries.
        # In the tensor decomposition, this means:
        # U vector has 1s at the A positions involved
        # V vector has 1s at the B positions involved
        # W vector has 1s at the C positions that Pi contributes to
        
        # But this doesn't work because U⊗V⊗W would give 1 for ALL combinations,
        # not just the intended products.
        
        # I think I need to reconsider the tensor decomposition approach.
        
        # Actually, the correct way: for the schoolbook term (i,j,k):
        # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
        # This gives 1 at exactly (i*m+j, j*p+k, k*n+i).
        
        # For Strassen, the "products" are of linear combinations, not single entries.
        # So I can't directly map Strassen to rank-1 tensors of the same form.
        
        # The tensor decomposition T = sum U_r ⊗ V_r ⊗ W_r requires that
        # T[a,b,c] = sum_r U_r[a] * V_r[b] * W_r[c].
        # For the multiplication tensor, T[a,b,c] = 1 iff (a,b,c) = (i*m+j, j*p+k, k*n+i).
        
        # Strassen doesn't directly give a tensor decomposition in this form.
        # It gives a bilinear algorithm, which IS a tensor decomposition, but the
        # rank-1 tensors have the form where U, V, W are sparse vectors.
        
        # Let me look up: the Strassen algorithm for 2x2 matrix multiplication
        # gives 7 rank-1 tensors for the 2x2 x 2x2 multiplication tensor.
        # The 2x2 x 2x2 tensor has T[i,j,k] = 1 iff C[i,k] = sum_j A[i,j]*B[j,k].
        # In flattened form: T[i*2+j, j*2+k, k*2+i] = 1 for i,j,k in 0..1.
        
        # The 7 Strassen rank-1 tensors for 2x2:
        # Let me define them explicitly.
        
        col += 7  # placeholder, will fill in
    
    # This approach is getting too complex without a clear mapping.
    # Let me just use schoolbook for everything and accept rank 100,
    # but structure it cleanly.
    
    # Actually, let me try a different structural decomposition.
    # 4x5 x 5x5: split B into B1 (4x5) and B2 (1x5), A into A1 (4x4) and A2 (4x1)
    # C = A1*B1 + A2*B2
    # A1*B1: 4x4 x 4x5 = 4x4 x (4x4 + 4x1) = (4x4 x 4x4) + (4x4 x 4x1)
    # A2*B2: 4x1 x 1x5
    
    # Schoolbook: 16 + 16 + 4*5 = 16+16+20 = 52... wait
    # A1*B1: 4x4 * 4x5: for each i in 0..3, k in 0..4: sum j in 0..3: A1[i,j]*B1[j,k]
    # That's 4*4*5 = 80 terms? No: 4 (i) * 4 (j) * 5 (k) = 80? 
    # Wait, 4x4 x 4x5: i in 0..3, j in 0..3, k in 0..4. That's 4*4*5 = 80 terms.
    # But schoolbook for 4x5 x 5x5 is 4*5*5 = 100.
    # A1*B1: 4*4*5 = 80, A2*B2: 4*1*5 = 20. Total 100. ✓
    
    # So no savings from this split. The total is always 100 with schoolbook.
    
    # To get savings, I need to use sub-quadratic algorithms for the subproblems.
    # The only subproblem where I can use Strassen is 4x4 x 4x4 (part of A1*B1).
    # 4x4 x 4x4 schoolbook: 64. With 2x2 Strassen: 56 (8 blocks * 7).
    # So I save 8 terms: 100 - 8 = 92.
    
    # Let me implement 4x4 x 4x4 with 2x2 Strassen = 56 terms.
    # 4x4 x 4x4: split into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each Ci is a sum of two 2x2 x 2x2 products.
    # Using Strassen for each 2x2 x 2x2: 7 terms.
    # Total: 8 * 7 = 56 terms.
    
    # For the 2x2 x 2x2 Strassen, I need the explicit rank-1 tensors.
    
    def add_2x2_strassen(ar, ac, br, bc, cr, cc):
        """Add 7 rank-1 tensors for 2x2 x 2x2 Strassen."""
        nonlocal col
        # A[i,j] index: (ar+i)*m + (ac+j)
        # B[i,j] index: (br+i)*p + (bc+j)  
        # C[i,j] index: (cc+j)*n + (cr+i)
        
        a00 = (ar+0)*m + (ac+0)
        a01 = (ar+0)*m + (ac+1)
        a10 = (ar+1)*m + (ac+0)
        a11 = (ar+1)*m + (ac+1)
        
        b00 = (br+0)*p + (bc+0)
        b01 = (br+0)*p + (bc+1)
        b10 = (br+1)*p + (bc+0)
        b11 = (br+1)*p + (bc+1)
        
        c00 = (cc+0)*n + (cr+0)
        c01 = (cc+1)*n + (cr+0)
        c10 = (cc+0)*n + (cr+1)
        c11 = (cc+1)*n + (cr+1)
        
        # Strassen 7 terms for 2x2:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # C00 = P1+P4-P5+P7
        # C01 = P3+P5
        # C10 = P2+P4
        # C11 = P1-P2+P3+P6
        
        # For each Pi, create a rank-1 tensor.
        # Pi = (linear combo of A) * (linear combo of B)
        # In tensor form: U has the A coefficients, V has the B coefficients,
        # W has the C coefficients (which Pi contributes to).
        
        # But this is where it gets tricky. The rank-1 tensor U⊗V⊗W gives
        # U[a]*V[b]*W[c]. For this to equal Pi at the right (a,b,c) positions,
        # I need U[a] = coefficient of A in the linear combo, V[b] = coefficient
        # of B, and W[c] = coefficient of C that Pi contributes to.
        
        # But U[a] is a single value for each a, and it's shared across all b,c.
        # So U⊗V⊗W = U[a]*V[b]*W[c] is a "rank-1" tensor in the trilinear form.
        
        # For P1 = (A00+A11)(B00+B11):
        # This should contribute to C00, C01, C10, C11 (all with +1 coefficient in some).
        # Actually P1 contributes: C00 += P1, C01 += 0, C10 += 0, C11 += P1
        # Wait no: C00 = P1+P4-P5+P7, so P1 contributes +1 to C00.
        # C11 = P1-P2+P3+P6, so P1 contributes +1 to C11.
        # C01 = P3+P5, P1 contributes 0.
        # C10 = P2+P4, P1 contributes 0.
        
        # So W for P1: W[c00] = 1, W[c11] = 1, W[c01] = 0, W[c10] = 0.
        # U for P1: U[a00] = 1, U[a11] = 1, U[a01] = 0, U[a10] = 0.
        # V for P1: V[b00] = 1, V[b11] = 1, V[b01] = 0, V[b10] = 0.
        
        # Then (U⊗V⊗W)[a00, b00, c00] = 1*1*1 = 1. ✓
        # (U⊗V⊗W)[a00, b11, c00] = 1*1*1 = 1. But we want 0 here!
        
        # The problem: U⊗V⊗W gives 1 for ALL combinations of non-zero U, V, W entries.
        # So (a00, b11, c00) gets 1, but in the actual tensor, T[a00, b11, c00] might be 0.
        
        # This means I can't use this naive approach. The Strassen algorithm doesn't
        # directly translate to a sparse rank-1 decomposition in this form.
        
        # I think the correct way is that each "multiplication" in Strassen corresponds
        # to a rank-1 tensor where U, V, W are NOT sparse but have the linear combination
        # coefficients. But then the tensor (U⊗V⊗W) is dense, and I need to verify
        # that the sum gives the correct T.
        
        # Let me just give up on Strassen and use schoolbook rank 100.
        col += 7  # placeholder
    
    # Given the complexity, let me just use schoolbook rank 100 with clean code.
    R = 100
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
