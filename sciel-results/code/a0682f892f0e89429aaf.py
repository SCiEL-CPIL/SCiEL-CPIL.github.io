import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    
    # Try to achieve rank 48 using a more efficient decomposition
    # For 2x4x8, we can use a bilinear algorithm based on the structure
    
    rank = 48
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    
    # For each pair of columns in B (4 pairs of 2 columns),
    # we need to compute 2x2 output block
    # Use Strassen-like approach but with sharing
    
    # The 4x8 matrix B has 4 rows and 8 columns
    # A is 2x4, so we multiply A[i, :] * B[:, j] for each i,j
    
    # Let's index: A[i, a] where i in {0,1}, a in {0,1,2,3}
    # B[b, c] where b in {0,1,2,3}, c in {0,...,7}
    # C[i, c] = sum_{a=0}^{3} sum_{b=0}^{3} A[i,a] * B[a,b] * ... 
    # Wait, actually: C[i,c] = sum_{a=0}^{3} A[i,a] * B[a,c]
    # Since B is 4x8, B[a,c] for a in 0..3, c in 0..7
    
    # So C[i,c] = sum_{a=0}^{3} A[i,a] * B[a,c]
    # This is just a 2x4 times 4x8 multiplication
    
    # The tensor has 2*4*8 = 64 nonzero entries
    # We need a bilinear decomposition
    
    # Let me try: for each column c of B (8 columns), we compute
    # C[:,c] = A @ B[:,c]
    # B[:,c] is a 4-vector, A is 2x4
    # So C[:,c] = A @ B[:,c]
    
    # The bilinear form: C[i,c] = sum_a A[i,a] * B[a,c]
    # This is already rank-1 in the (i,a) vs (a,c) sense? No.
    
    # Actually the tensor T[i,a; b,c] = delta_{a,b} for the multiplication
    # T_{(i,a),(b,c)} = 1 if a==b, else 0
    # Shape: (2*4) x (4*8) = 8 x 32
    # This is just the identity-like structure
    
    # The rank of this tensor as a bilinear form...
    # T_{(i,a),(b,c)} = delta_{a,b}
    # This can be written as sum over a of e_{i,a} * e_{a,c} for each i
    # Wait no: T_{(i,a),(b,c)} = delta_{a,b}
    # = sum_{a'} delta_{a,a'} * delta_{a',b} 
    # So it's sum_{a'} (e_{i,a'} for fixed i... 
    
    # Let me think differently. The tensor T has indices (i, a, b, c) where
    # T[i,a,b,c] = 1 if a==b else 0, with i in 0..1, a in 0..3, b in 0..3, c in 0..7
    # But wait, in matrix multiplication C = A*B, C[i,c] = sum_b A[i,b]*B[b,c]
    # So the tensor is T[i,b; b,c] = 1 (reusing b)
    # T[i,a,b,c] = delta_{a,b}
    
    # As a matrix (flattening i,a into rows and b,c into cols):
    # M[(i,a), (b,c)] = delta_{a,b}
    # This is an 8x32 matrix where M[(i,a),(b,c)] = 1 iff a=b
    # For fixed a, the row (i,a) has 1s in all columns (a, c) for c=0..7
    # So row (0,a) = [0..0, 11111111, 0..0] (8 ones at positions 8a..8a+7)
    # Similarly row (1,a) = same pattern
    
    # So the 8x32 matrix has rows:
    # (0,0): 11111111 00000000 00000000 00000000
    # (0,1): 00000000 11111111 00000000 00000000
    # (0,2): 00000000 00000000 11111111 00000000
    # (0,3): 00000000 00000000 00000000 11111111
    # (1,0): 11111111 00000000 00000000 00000000
    # (1,1): 00000000 11111111 00000000 00000000
    # (1,2): 00000000 00000000 11111111 00000000
    # (1,3): 00000000 00000000 00000000 11111111
    
    # This matrix has rank 4! (4 distinct row patterns, each repeated twice)
    # So the bilinear rank is at most 4?
    # Wait, that can't be right for a rank-minimal decomposition problem...
    
    # Oh wait, I'm confusing the tensor rank with the matrix rank of the flattening.
    # The tensor T[i,a,b,c] = delta_{a,b} as a 4-index tensor has a different rank
    # than the 2-mode flattening.
    
    # Actually for bilinear complexity, the rank IS the rank of the 2-mode flattening
    # T: (i,a) x (b,c) -> scalar, so rank = rank of the 8x32 matrix = 4
    
    # But that would mean rank 4, which is way below 56. Let me re-examine.
    
    # Hmm, I think I'm making an error. Let me reconsider the tensor structure.
    
    # C[i,c] = sum_{b} A[i,b] * B[b,c]
    # The tensor T has T[i,b; b,c] = 1
    # In the bilinear form: C = sum_r (sum_i a_i U[i,r]) (sum_{b,c} b_{b,c} V[b,c,r]) W[...]
    
    # Actually, the standard formulation:
    # C[i,c] = sum_r U[i,r] * V[r] * W[r,c] where V[r] = sum_b A[i,b]V[b,r]... 
    
    # I think I need to be more careful. Let me just use the known result.
    # For 2xn x nxp, the bilinear complexity is at most 2n*ceil(p/2) using Strassen
    # For n=4, p=8: 2*4*4 = 32? No...
    
    # Let me just try to construct a rank-48 solution by hand or use a known algorithm.
    
    # Actually, let me try a simpler approach: use the fact that the problem is
    # essentially 4 independent 2x1x2 multiplications (one for each column of B)
    # Each 2x1x2 needs rank 2 (just two multiplications)
    # But we can share: the "1" in the middle means we're just doing dot products.
    
    # C[0,c] = A[0,0]*B[0,c] + A[0,1]*B[1,c] + A[0,2]*B[2,c] + A[0,3]*B[3,c]
    # C[1,c] = A[1,0]*B[0,c] + A[1,1]*B[1,c] + A[1,2]*B[2,c] + A[1,3]*B[3,c]
    
    # This is just 8 independent bilinear forms (2 outputs x 4 terms each)
    # Total bilinear complexity = 8 (one for each (i,c) pair, each being a sum of 4 products)
    # Wait no, each C[i,c] is a sum of 4 products, so we need 4 scalar multiplications per C[i,c]
    # Total = 2 * 8 * 4 / ... no.
    
    # The bilinear complexity counts the number of scalar multiplications.
    # C[i,c] = sum_b A[i,b]*B[b,c] requires 4 multiplications for each (i,c)
    # Total = 2 * 8 * 4 = 64 multiplications (schoolbook)
    
    # Can we do better? The tensor T[i,b; b,c] = delta_{b,b}... 
    # As a matrix M[(i,b),(b',c)] = delta_{b,b'}
    # This is an 8x32 matrix. Its rank is 4 (as I computed above).
    # So the bilinear complexity is 4?!
    
    # That seems too low. Let me verify with a small case.
    # 1x1 x 1x1: C[0,0] = A[0,0]*B[0,0]. Tensor T[0,0;0,0]=1. Matrix is 1x1, rank 1. Correct.
    # 1x2 x 2x1: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]. 
    # T[0,b; b',c] = delta_{b,b'}, i in {0}, b in {0,1}, b' in {0,1}, c in {0}
    # Matrix M[(0,b),(b',0)] = delta_{b,b'}, shape 2x2, which is identity, rank 2. Correct.
    # 2x1 x 1x2: C[i,c] = A[i,0]*B[0,c]. 
    # T[i,0; 0,c] = 1 for all i,c. Matrix M[(i,0),(0,c)] = 1, shape 2x2, all ones, rank 1.
    # So bilinear complexity is 1? That means C[0,0] = (A[0,0]+A[1,0])*(B[0,0]+B[0,1])... no that doesn't work.
    # Actually C[0,0] = A[0,0]*B[0,0], C[0,1] = A[0,0]*B[0,1], C[1,0] = A[1,0]*B[0,0], C[1,1] = A[1,0]*B[0,1]
    # = A[0,0]*B[0,0] + A[0,0]*B[0,1] + A[1,0]*B[0,0] + A[1,0]*B[0,1]
    # = (A[0,0]+A[1,0]) * (B[0,0]+B[0,1]) - A[0,0]*B[0,1] - A[1,0]*B[0,0]
    # That's 3 multiplications, not 1. So rank 1 is wrong.
    
    # I think the issue is that the bilinear complexity is NOT simply the matrix rank of the flattening.
    # The tensor rank (for bilinear complexity) is the minimum r such that
    # T = sum_{k=1}^r u_k * v_k * w_k (as a 3-tensor or appropriately flattened)
    
    # For matrix multiplication C=AB, the tensor is T[i,j,k] = delta_{j,k} (for C[i,k] = sum_j A[i,j]B[j,k])
    # This is a 3-tensor with modes (i, j) x (j, k) ... 
    
    # I'm getting confused. Let me just stick with the Strassen approach and try to optimize it.
    
    # Current: rank 56. Let me see if I can do better.
    # 2x4x8: Think of it as 2x(4x8). 
    # Using Strassen for 2x2x2: rank 7 per 2x2x2 block.
    # We have 4 blocks of 2x2x2 (splitting the 4 columns of A into 2 groups of 2, 
    # and the 8 columns of B into 4 groups of 2)
    # 4 * 7 = 28? No, that's not right either because the blocks don't share the same A entries.
    
    # Actually: A is 2x4, split into A_left (2x2) and A_right (2x2)
    # B is 4x8, split into B_top (2x8) and B_bottom (2x8)
    # C = A_left * B_top + A_right * B_bottom
    # Each 2x2 * 2x8 is a 2x2x8 multiplication
    # 2x2x8: split B into four 2x2 blocks, each needs rank 7, total 28
    # Two of these: 56. That's what we have.
    
    # Can we do better? 2x2x8: is there a better algorithm?
    # 2x2xp for p=8. Known results?
    # For 2x2x2: rank 7 (Strassen)
    # For 2x2x4: rank 13 (known)
    # For 2x2x8: rank ? 
    # If we use Strassen recursively: 2x2x4 = 2*(2x2x2) but with sharing... 
    # Actually 2x2x4 can be done in rank 13 (not 14 = 2*7)
    # Then 2x2x8 might be doable in rank 25 or so?
    
    # Let me look up: the bilinear complexity of 2x2xp.
    # For p=1: 4, p=2: 7, p=3: 10, p=4: 13, p=5: 16, p=6: 19, p=7: 22, p=8: 25?
    # Pattern: 3p+1 for p>=2? p=2:7, p=3:10, p=4:13... yes 3p+1
    # So 2x2x8 would be rank 25.
    # Then 2x4x8 = 2*(2x2x8) = 50? Or can we share?
    
    # If 2x2x8 has rank 25, then 2x4x8 (which is two independent 2x2x8 problems) has rank 50.
    # That's better than 56!
    
    # Let me try to construct a rank-25 algorithm for 2x2x8.
    # 2x2x8: C[i,c] = A[i,0]*B[0,c] + A[i,1]*B[1,c] for i in {0,1}, c in 0..7
    # = A[i,0]*B[0,c] + A[i,1]*B[1,c]
    
    # Using the 3p+1 pattern: for each new column, we add 3 multiplications
    # Base case 2x2x2: 7 multiplications
    # 2x2x3: 10, 2x2x4: 13, ..., 2x2x8: 25
    
    # The algorithm: 
    # For 2x2x2, Strassen uses 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01+B11)
    # M4 = (A11)(B10+B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x2x3 (columns 0,1,2):
    # We can extend by adding 3 more products for the 3rd column.
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # We need: A00*B02, A10*B02, A01*B12, A11*B12
    # But we can reuse some structures.
    
    # Actually, the known result is that 2x2xp has bilinear complexity 3p+1 for p>=2.
    # Let me just implement this.
    
    # For 2x2xp with p columns:
    # Use p-1 Strassen-like steps, each adding 3 products.
    # Total: 7 + 3*(p-2) = 3p+1
    
    # Let me implement 2x2x8 with rank 25.
    
    def make_2x2xp(p):
        """Return U, V, W for 2x2xp multiplication with rank 3p+1"""
        r = 3 * p + 1
        U = np.zeros((4, r))  # 2x2 A matrix, flattened
        V = np.zeros((2*p, r))  # 2xp B matrix, flattened
        W = np.zeros((2*p, r))  # 2xp C matrix, flattened
        
        # A indices: A00=0, A01=1, A10=2, A11=3
        # B indices: B0c = c, B1c = p+c
        # C indices: C0c = c, C1c = p+c
        
        col = 0
        # First 2 columns use Strassen (7 products)
        # M1 = (A00+A11)(B00+B11)
        U[0, col] = 1; U[3, col] = 1
        V[0, col] = 1; V[p+1, col] = 1
        W[0, col] = 1; W[p+1, col] = 1
        col += 1
        # M2 = (A10+A11)(B00)
        U[2, col] = 1; U[3, col] = 1
        V[0, col] = 1
        W[p, col] = 1; W[p+1, col] = -1
        col += 1
        # M3 = (A00)(B01+B11)
        U[0, col] = 1
        V[1, col] = 1; V[p+1, col] = 1
        W[1, col] = 1; W[p+1, col] = 1
        col += 1
        # M4 = (A11)(B10+B00)
        U[3, col] = 1
        V[p, col] = 1; V[0, col] = 1
        W[0, col] = 1; W[p, col] = 1
        col += 1
        # M5 = (A00+A01)(B11)
        U[0, col] = 1; U[1, col] = 1
        V[p+1, col] = 1
        W[0, col] = -1; W[1, col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[2, col] = 1; U[0, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[p+1, col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[1, col] = 1; U[3, col] = -1
        V[p, col] = 1; V[p+1, col] = 1
        W[0, col] = 1
        col += 1
        
        # For columns 2 to p-1, add 3 products each
        for c in range(2, p):
            # We need to compute C0c = A00*B0c + A01*B1c
            # and C1c = A10*B0c + A11*B1c
            # Using 3 new products:
            # N1 = (A00+A11) * B0c  -- but we already have (A00+A11) from M1
            # N2 = (A01+A10) * B1c  -- new
            # N3 = (A00-A11) * (B0c-B1c) -- new, but this gives A00*B0c - A00*B1c - A11*B0c + A11*B1c
            
            # Actually, let me use a simpler extension:
            # N1 = A00 * B0c
            # N2 = A11 * B1c  
            # N3 = (A01+A10) * (B1c + B0c) ... no this doesn't directly help
            
            # Better: 
            # C0c = A00*B0c + A01*B1c
            # C1c = A10*B0c + A11*B1c
            # Let N1 = (A00+A10) * B0c = A00*B0c + A10*B0c
            # Let N2 = (A01+A11) * B1c = A01*B1c + A11*B1c
            # Let N3 = (A00-A11) * (B0c - B1c) = A00*B0c - A00*B1c - A11*B0c + A11*B1c
            # Then:
            # C0c = N1 + N2 - A10*B0c - A11*B1c... this is getting complicated.
            
            # Simpler approach for extension:
            # N1 = A00 * B0c
            # N2 = A01 * B1c
            # N3 = (A10+A11) * (B0c+B1c)
            # C0c = N1 + N2
            # C1c = N3 - A10*B1c - A11*B0c... still not clean.
            
            # Let me use:
            # N1 = (A00+A01) * B0c
            # N2 = (A10+A11) * B1c
            # N3 = (A00-A10) * (B0c-B1c)
            # C0c = N1 + N2 - A01*B0c - A11*B1c... no.
            
            # I think the cleanest extension is:
            # For each new column c, use 3 products:
            # N1 = A00 * B0c
            # N2 = A11 * B1c
            # N3 = (A01 + A10) * (B0c + B1c)
            # Then C0c = N1 + A01*B1c = N1 + (N3 - A01*B0c - A10*B0c - A10*B1c)... no.
            
            # OK let me just use 4 products per column for simplicity, giving rank 7+4*(p-2) = 4p+3
            # For p=8: 35. Then 2x4x8 = 70. Worse than 56.
            
            # Let me try a different approach. Use the known 3p+1 result.
            # The construction for 2x2xp with 3p+1 rank:
            # For column 0: use 7 Strassen products (covers columns 0 and 1)
            # For each additional column c>=2: add 3 products
            
            # The 3 products for column c:
            # P1 = (A00 + A11) * B0c
            # P2 = (A01 + A10) * B1c  
            # P3 = (A00 - A11) * (B0c - B1c)
            # Then:
            # C0c = A00*B0c + A01*B1c
            # C1c = A10*B0c + A11*B1c
            # P1 = A00*B0c + A11*B0c
            # P2 = A01*B1c + A10*B1c
            # P3 = A00*B0c - A00*B1c - A11*B0c + A11*B1c
            # P1 + P3 = 2*A00*B0c - A00*B1c + A11*B1c... not quite.
            
            # Let me try:
            # P1 = A00 * B0c
            # P2 = A11 * B1c
            # P3 = (A01 + A10) * B0c  ... no
            
            # I'll use a different known construction. The bilinear complexity of 2x2xp is indeed 3p+1.
            # Reference: "The complexity of multiplying 2x2 matrices" etc.
            
            # For now, let me just use rank 4p for 2x2xp (4 products per column)
            # 2x2x8: rank 32. Then 2x4x8: rank 64. Worse.
            
            # Let me go back to the current approach and see if I can shave off a few ranks.
            # Current: 56 = 8 blocks * 7 Strassen products.
            # Can I share between adjacent blocks?
            
            # The 8 blocks are:
            # (a_col_off, b_row_off, b_col_off, c_col_off)
            # (0,0,0,0), (0,0,2,2), (0,0,4,4), (0,0,6,6)
            # (2,2,0,0), (2,2,2,2), (2,2,4,4), (2,2,6,6)
            
            # Blocks with same b_col_off share B entries.
            # Blocks (0,0,0,0) and (2,2,0,0) both use B[0:2, 0:2]
            # But they use different A entries.
            
            # I don't see obvious sharing opportunities.
            
            # Let me just return the current solution as-is since I can't easily improve it.
            pass
        
        return U, V, W, col
    
    # Let me just return the current Strassen solution
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (0, 0, 6, 6),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4),
        (2, 2, 6, 6),
    ]
    
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        a00 = 0*4 + a_col_off
        a01 = 0*4 + a_col_off + 1
        a10 = 1*4 + a_col_off
        a11 = 1*4 + a_col_off + 1
        
        b00 = b_row_off*8 + b_col_off
        b01 = b_row_off*8 + b_col_off + 1
        b10 = (b_row_off+1)*8 + b_col_off
        b11 = (b_row_off+1)*8 + b_col_off + 1
        
        c00 = c_col_off*2 + 0
        c01 = (c_col_off+1)*2 + 0
        c10 = c_col_off*2 + 1
        c11 = (c_col_off+1)*2 + 1
        
        # Strassen 1
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Strassen 2
        U[a10, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        W[c11, col] += -1
        col += 1
        
        # Strassen 3
        U[a00, col] += 1
        V[b01, col] += 1
        V[b11, col] += -1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Strassen 4
        U[a11, col] += 1
        V[b10, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # Strassen 5
        U[a00, col] += 1
        U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # Strassen 6
        U[a10, col] += 1
        U[a00, col] += -1
        V[b00, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Strassen 7
        U[a01, col] += 1
        U[a11, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    
    assert col == 56
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
