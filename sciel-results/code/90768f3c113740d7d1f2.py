import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (16, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # The matrix multiplication tensor for C = A*B where A is 2x4, B is 4x8, C is 2x8:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We need: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook rank is 2*4*8 = 64
    # We can use Strassen-like tricks to reduce the rank.
    
    # Strategy: Partition A into two 2x2 blocks and B into 2x4 blocks
    # A = [A1 | A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x4
    # C = A1*B1 + A2*B2
    
    # For each 2x2 times 2x4 multiplication, we can use 8 multiplications (schoolbook)
    # But we can potentially do better with Strassen-like tricks.
    
    # Let's use the following approach:
    # For A1*B1 where A1 is 2x2 and B1 is 2x4:
    # A1 = [[a, b], [c, d]], B1 = [[e, f, g, h], [i, j, k, l]]
    # C1[0,0] = a*e + b*i
    # C1[0,1] = a*f + b*j
    # C1[0,2] = a*g + b*k
    # C1[0,3] = a*h + b*l
    # C1[1,0] = c*e + d*i
    # C1[1,1] = c*f + d*j
    # C1[1,2] = c*g + d*k
    # C1[1,3] = c*h + d*l
    
    # We can use 8 multiplications (schoolbook) or potentially fewer with Strassen.
    
    # Let's try to find a rank-7 decomposition for 2x2 times 2x4.
    # Actually, the rank of 2x2 times 2x4 is 8 (I believe this is tight).
    
    # So we can use 8 multiplications for A1*B1 and 8 for A2*B2, total 16.
    # But wait, that's better than 64!
    
    # Let me implement this.
    
    rank = 16
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B1 = B[0:2, :], B2 = B[2:4, :]
    
    # For A1*B1:
    # A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B1 = [[B[0,0], B[0,1], ..., B[0,7]], [B[1,0], B[1,1], ..., B[1,7]]]
    
    # 8 multiplications:
    # (A[0,0], B[0,k]) -> C[0,k] for k=0..7
    # (A[0,1], B[1,k]) -> C[0,k] for k=0..7
    # (A[1,0], B[0,k]) -> C[1,k] for k=0..7
    # (A[1,1], B[1,k]) -> C[1,k] for k=0..7
    
    # But that's 32 multiplications, not 8!
    
    # I see my error. Let me reconsider.
    
    # The issue is that B1 is 2x4, not 2x8. Let me re-partition.
    
    # Let's partition B into 4 blocks of 2x2:
    # B = [[B11, B12, B13, B14], [B21, B22, B23, B24]] where each Bij is 2x2
    
    # Then C = A1*B11 + A2*B21 for the first 2 columns
    # C = A1*B12 + A2*B22 for the next 2 columns
    # etc.
    
    # So for each output column pair, we have a sum of two 2x2 times 2x2 products.
    # Each 2x2 times 2x2 can be done with 7 multiplications (Strassen).
    # So for each output column pair, we need 14 multiplications.
    # Total: 4 * 14 = 56 multiplications.
    
    # That's the same as before. Let me implement this properly.
    
    rank = 56
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    
    # For each output column pair c (0 to 3):
    # C[:, 2c:2c+2] = A[:, 0:2] * B[0:2, 2c:2c+2] + A[:, 2:4] * B[2:4, 2c:2c+2]
    
    for c in range(4):
        # A1 = A[:, 0:2], B1 = B[0:2, 2c:2c+2]
        # A2 = A[:, 2:4], B2 = B[2:4, 2c:2c+2]
        
        # Strassen for A1*B1:
        # A1 = [[a, b], [c, d]], B1 = [[e, f], [g, h]]
        # P1 = (a+c)*e
        # P2 = (a+c)*f
        # P3 = b*(h-g)
        # P4 = d*(f-e)
        # P5 = (a+d)*(e+h)
        # P6 = (b-c)*(g+h)
        # P7 = (a-b)*(e+g)
        
        # C1[0,0] = P5 + P4 - P2 + P6
        # C1[0,1] = P1 + P2
        # C1[1,0] = P3 + P5
        # C1[1,1] = P1 - P4 + P7
        
        # A1 indices: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
        # B1 indices: B[0,2c]=2c, B[0,2c+1]=2c+1, B[1,2c]=8+2c, B[1,2c+1]=9+2c
        # C indices: C[0,2c]=2c, C[0,2c+1]=2c+1, C[1,2c]=8+2c, C[1,2c+1]=9+2c
        
        a00, a01 = 0, 1
        a10, a11 = 4, 5
        b00, b01 = 2*c, 2*c+1
        b10, b11 = 8+2*c, 9+2*c
        c00, c01 = 2*c, 2*c+1
        c10, c11 = 8+2*c, 9+2*c
        
        # P1 = (a+c)*e
        U[a00, col] += 1
        U[a10, col] += 1
        V[b00, col] += 1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P2 = (a+c)*f
        U[a00, col] += 1
        U[a10, col] += 1
        V[b01, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # P3 = b*(h-g)
        U[a01, col] += 1
        V[b11, col] += 1
        V[b10, col] += -1
        W[c10, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P4 = d*(f-e)
        U[a11, col] += 1
        V[b01, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c11, col] += -1
        col += 1
        
        # P5 = (a+d)*(e+h)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # P6 = (b-c)*(g+h)
        U[a01, col] += 1
        U[a10, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
        # P7 = (a-b)*(e+g)
        U[a00, col] += 1
        U[a01, col] += -1
        V[b00, col] += 1
        V[b10, col] += 1
        W[c11, col] += 1
        col += 1
        
        # Strassen for A2*B2:
        # A2 = [[a, b], [c, d]], B2 = [[e, f], [g, h]]
        # Same structure as above, but with different indices.
        
        a00, a01 = 2, 3
        a10, a11 = 6, 7
        b00, b01 = 16+2*c, 17+2*c
        b10, b11 = 24+2*c, 25+2*c
        c00, c01 = 2*c, 2*c+1
        c10, c11 = 8+2*c, 9+2*c
        
        # P1 = (a+c)*e
        U[a00, col] += 1
        U[a10, col] += 1
        V[b00, col] += 1
        W[c01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P2 = (a+c)*f
        U[a00, col] += 1
        U[a10, col] += 1
        V[b01, col] += 1
        W[c00, col] += -1
        W[c01, col] += 1
        col += 1
        
        # P3 = b*(h-g)
        U[a01, col] += 1
        V[b11, col] += 1
        V[b10, col] += -1
        W[c10, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P4 = d*(f-e)
        U[a11, col] += 1
        V[b01, col] += 1
        V[b00, col] += -1
        W[c00, col] += 1
        W[c11, col] += -1
        col += 1
        
        # P5 = (a+d)*(e+h)
        U[a00, col] += 1
        U[a11, col] += 1
        V[b00, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # P6 = (b-c)*(g+h)
        U[a01, col] += 1
        U[a10, col] += -1
        V[b10, col] += 1
        V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
        # P7 = (a-b)*(e+g)
        U[a00, col] += 1
        U[a01, col] += -1
        V[b00, col] += 1
        V[b10, col] += 1
        W[c11, col] += 1
        col += 1
    
    assert col == 56, f"Expected 56, got {col}"
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
