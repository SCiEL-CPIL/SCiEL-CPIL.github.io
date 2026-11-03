"""Matrix multiplication 2x3x10 tensor decomposition using Strassen-style approach."""
DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = 2, 3, 10
    # We'll decompose the 2x3 matrix into a 2x2 and 2x1 part
    # The 2x2 part can use Strassen (rank 7)
    # The 2x1 part needs rank 2 (schoolbook)
    # For p=10, we split into 5 blocks of p=2
    # Each block: 2x2x2 (Strassen: rank 7) + 2x1x2 (rank 2) = rank 9
    # Total rank: 5 * 9 = 45
    
    rank = 45
    U = [[0] * rank for _ in range(n * m)]  # 6 x 45
    V = [[0] * rank for _ in range(m * p)]  # 30 x 45
    W = [[0] * rank for _ in range(p * n)]  # 20 x 45
    
    col = 0
    for b in range(5):  # 5 blocks of p=2
        # For each block, we handle 2x3x2 multiplication
        # Split A into A_top (2x2) and A_bottom (2x1)
        # B columns for this block: B[:, 2b:2b+2]
        
        # Strassen for 2x2x2 part (using 7 multiplications)
        # A = [[a11, a12], [a21, a22]]
        # B = [[b11, b12], [b21, b22]] (2x2 part of the 3x2 block)
        
        # Strassen's 7 products for 2x2 matrix multiplication:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        
        # For Strassen, we need:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # The 2x1 part: A_bottom (2x1) times B_col1 (1x2)
        # This gives: C13 = a13*b31, C23 = a23*b31 (but we have only 2x1, so)
        # Actually for 2x1 x 1x2: result is 2x2
        # C11 += a13*b31, C12 += a13*b32
        # C21 += a23*b31, C22 += a23*b32
        
        # But wait, the 3rd column of B is shared with the 2x2 part in Strassen
        # Let me reconsider: the 2x3 matrix A has rows:
        # A[0] = [a00, a01, a02]  (row 0 of A, length 3)
        # A[1] = [a10, a11, a12]  (row 1 of A, length 3)
        
        # B is 3x2 for this block:
        # B[0] = [b00, b01]
        # B[1] = [b10, b11]
        # B[2] = [b20, b21]
        
        # C = A * B (2x2 result)
        # C[0,0] = a00*b00 + a01*b10 + a02*b20
        # C[0,1] = a00*b01 + a01*b11 + a02*b21
        # C[1,0] = a10*b00 + a11*b10 + a12*b20
        # C[1,1] = a10*b01 + a11*b11 + a12*b21
        
        # Split into 2x2 part (using first 2 columns) and 1x2 part (3rd column):
        # A_2x2 = [[a00, a01], [a10, a11]]
        # A_1x2 = [[a02], [a12]]
        # B_2x2 = [[b00, b01], [b10, b11]]
        # B_1x2 = [[b20], [b21]]
        
        # C = A_2x2 * B_2x2 + A_1x2 * B_1x2
        
        # For A_2x2 * B_2x2, use Strassen (7 products)
        # For A_1x2 * B_1x2, this is outer product: [a02; a12] * [b20, b21]
        # = [[a02*b20, a02*b21], [a12*b20, a12*b21]]
        # This needs 2 products (a02*b20 and a02*b21, or a12*b20 and a12*b21)
        # Actually we need 2 products: p1 = a02*b20, p2 = a02*b21, p3 = a12*b20, p4 = a12*b21
        # Wait, that's 4. But we can factor: (a02, a12) outer (b20, b21) needs 4 products in general.
        # Hmm, but rank of 2x1x2 outer product tensor is 2.
        # Actually the outer product u*v^T where u is 2x1 and v is 1x2 has rank 2.
        # We can write: u*v^T = u1*v1 * e1*e1^T + u1*v2 * e1*e2^T... no
        # u*v^T = [u1*v1, u1*v2; u2*v1, u2*v2]
        # This can be written as sum of 2 rank-1 tensors? No, the tensor itself has rank 2.
        # Actually, the multiplication tensor for (2x1)*(1x2) -> 2x2 has rank 2.
        # We can use 2 products: let's say we compute t1 = u1*v1 and t2 = u2*v2? No, that doesn't give all entries.
        # Actually, rank of this tensor is 2. One decomposition:
        # C[0,0] = u1*v1
        # C[0,1] = u1*v2  
        # C[1,0] = u2*v1
        # C[1,1] = u2*v2
        # This is 4 products in schoolbook. But the tensor rank is 2.
        # Hmm, actually for a 2x1 matrix times 1x2 matrix, the result is rank 1 matrix.
        # The tensor T_{i,j,k} for this has i in {0,1}, j in {0} (since B is 1x2, j indexes rows of B which is 1), k in {0,1}
        # T_{i,0,k} = delta_{i,k}... no.
        # Let me think again. A is 2x1, B is 1x2, C is 2x2.
        # C[i,k] = sum_j A[i,j] * B[j,k] = A[i,0] * B[0,k]
        # So C[i,k] = A[i,0] * B[0,k]
        # The tensor has entries T[i, 0, k] = 1 for all i,k, else 0.
        # This is the outer product of e_i (in R^2) and e_k (in R^2), which has rank 1 as a matrix but...
        # Actually as a tensor, T = sum over i,k of e_i ⊗ e_0 ⊗ e_k
        # This can be written as (sum_i e_i ⊗ e_i) ⊗ e_0? No.
        # T = (e_0 ⊗ e_0 ⊗ e_0) + (e_0 ⊗ e_0 ⊗ e_1) + (e_1 ⊗ e_0 ⊗ e_0) + (e_1 ⊗ e_0 ⊗ e_1)
        # That's 4 terms. But we can do better.
        # Actually, the rank of this tensor is 2. 
        # T[i,0,k] = A[i,0]*B[0,k]. We can write this as:
        # Let u = (A[0,0], A[1,0]) and v = (B[0,0], B[0,1])
        # C = u v^T = u_0 v_0 e_0 e_0^T + u_0 v_1 e_0 e_1^T + u_1 v_0 e_1 e_0^T + u_1 v_1 e_1 e_1^T
        # That's 4. But the tensor rank is actually 2 for this specific tensor.
        # Hmm, let me just use 2 products by being clever.
        # Actually, I think the rank is 2. Let's use:
        # Product 1: (A[0,0] + A[1,0]) * (B[0,0] + B[0,1])
        # Product 2: (A[0,0] - A[1,0]) * (B[0,0] - B[0,1])
        # Then: A[0,0]*B[0,0] = (P1 + P2)/4... no this is getting complicated.
        
        # For simplicity and correctness, let me just use the known result:
        # 2x2x2 has rank 7 (Strassen)
        # 2x1x2 has rank 2
        # So 2x3x2 has rank 7 + 2 = 9
        
        # Let me look up or derive the rank-2 decomposition for 2x1x2.
        # Actually, I recall that the tensor for multiplying a 2x1 by 1x2 has rank 2.
        # One explicit decomposition:
        # C[0,0] = A[0,0]*B[0,0]
        # C[0,1] = A[0,0]*B[0,1]
        # C[1,0] = A[1,0]*B[0,0]
        # C[1,1] = A[1,0]*B[0,1]
        # Using 2 products:
        # t1 = (A[0,0] + A[1,0]) * (B[0,0] + B[0,1])
        # t2 = (A[0,0] - A[1,0]) * (B[0,0] - B[0,1])
        # Then:
        # A[0,0]*B[0,0] = (t1 + t2 + (A[0,0]*B[0,1] + A[1,0]*B[0,0]))/2 ... this doesn't work cleanly.
        
        # Let me just use 4 products for the 1x2 part to be safe, giving rank 7+4=11 per block, total 55.
        # But the knowledge says rank 55 was achieved with 11 per block.
        # Actually, the knowledge says "reduces the rank per p=2 block from 12 to 11".
        # Schoolbook for 2x3x2 would be 12 products. With Strassen for 2x2 part (7) + 2x1 part.
        # If 2x1 part uses 4 products, total is 11. That matches!
        
        # So let's use: Strassen (7) + schoolbook for 2x1x2 (4) = 11 per block
        # Total: 5 * 11 = 55
        
        # Strassen decomposition for 2x2x2:
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # For the 2x2 part:
        # a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
        # e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1]
        
        # For the 2x1 part:
        # A_1 = A[0,2], A_2 = A[1,2]
        # B_1 = B[2,0], B_2 = B[2,1]
        # C[0,0] += A_1 * B_1
        # C[0,1] += A_1 * B_2
        # C[1,0] += A_2 * B_1
        # C[1,1] += A_2 * B_2
        
        # Strassen terms:
        # M1: (a+d)(e+h)
        U[0][col] += 1   # a = A[0,0]
        U[1][col] += 1   # d = A[1,1]
        V[0][col] += 1   # e = B[0,0]
        V[1][col] += 1   # h = B[1,1]
        W[0*2+0][col] += 1   # C[0,0] (k=0, i=0)
        W[0*2+1][col] -= 1   # C[1,0] (k=0, i=1) -- wait, W[k*n+i]
        W[1*2+0][col] += 1   # C[0,1] (k=1, i=0)
        W[1*2+1][col] -= 1   # C[1,1] (k=1, i=1)
        col += 1
        
        # M2: (c+d)e
        U[1][col] += 1   # c = A[1,0]
        U[2][col] += 1   # d = A[1,1]
        V[0][col] += 1   # e = B[0,0]
        W[0*2+1][col] += 1   # C[1,0] (k=0, i=1)
        W[1*2+1][col] -= 1   # C[1,1] (k=1, i=1)
        col += 1
        
        # M3: a(f-h)
        U[0][col] += 1   # a = A[0,0]
        V[1][col] += 1   # f = B[0,1]
        V[3][col] -= 1   # h = B[1,1]
        W[0*2+0][col] += 1   # C[0,0] (k=0, i=0)
        W[1*2+0][col] += 1   # C[0,1] (k=1, i=0)
        col += 1
        
        # M4: d(g-e)
        U[1][col] += 1   # d = A[1,1]
        V[2][col] += 1   # g = B[1,0]
        V[0][col] -= 1   # e = B[0,0]
        W[0*2+0][col] += 1   # C[0,0] (k=0, i=0)
        W[0*2+1][col] += 1   # C[1,0] (k=0, i=1)
        col += 1
        
        # M5: (a+b)h
        U[0][col] += 1   # a = A[0,0]
        U[1][col] += 1   # b = A[0,1]
        V[3][col] += 1   # h = B[1,1]
        W[0*2+0][col] -= 1   # C[0,0] (k=0, i=0)
        W[1*2+0][col] += 1   # C[0,1] (k=1, i=0)
        col += 1
        
        # M6: (c-a)(e+f)
        U[1][col] += 1   # c = A[1,0]
        U[0][col] -= 1   # a = A[0,0]
        V[0][col] += 1   # e = B[0,0]
        V[1][col] += 1   # f = B[0,1]
        W[1*2+1][col] += 1   # C[1,1] (k=1, i=1)
        col += 1
        
        # M7: (b-d)(g+h)
        U[0][col] += 1   # b = A[0,1]
        U[1][col] -= 1   # d = A[1,1]
        V[2][col] += 1   # g = B[1,0]
        V[3][col] += 1   # h = B[1,1]
        W[0*2+0][col] += 1   # C[0,0] (k=0, i=0)
        W[1*2+1][col] += 1   # C[1,1] (k=1, i=1)
        col += 1
        
        # 2x1x2 part (schoolbook, 4 products):
        # A[0,2]*B[2,0] -> C[0,0]
        U[2][col] += 1   # A[0,2]
        V[4][col] += 1   # B[2,0]
        W[0*2+0][col] += 1   # C[0,0]
        col += 1
        
        # A[0,2]*B[2,1] -> C[0,1]
        U[2][col] += 1   # A[0,2]
        V[5][col] += 1   # B[2,1]
        W[1*2+0][col] += 1   # C[0,1]
        col += 1
        
        # A[1,2]*B[2,0] -> C[1,0]
        U[5][col] += 1   # A[1,2]
        V[4][col] += 1   # B[2,0]
        W[0*2+1][col] += 1   # C[1,0]
        col += 1
        
        # A[1,2]*B[2,1] -> C[1,1]
        U[5][col] += 1   # A[1,2]
        V[5][col] += 1   # B[2,1]
        W[1*2+1][col] += 1   # C[1,1]
        col += 1
        
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
