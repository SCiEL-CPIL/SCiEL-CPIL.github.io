"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for M_{n,m,p}."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _build_2x4x7_decomposition():
    """Build a decomposition for 2x4x7 using block methods with Strassen on 2x2x2 blocks."""
    n, m, p = 2, 4, 7
    
    # Strategy: Split the 4 middle dimension into 2+2, giving two M_{2,2,7} subproblems.
    # For each M_{2,2,7}, split the 7 columns into 2+2+2+1.
    # Use Strassen (rank 7) for each 2x2x2 block, and schoolbook (rank 4) for 2x2x1.
    # Total per M_{2,2,7}: 3*7 + 4 = 25
    # Total for M_{2,4,7}: 2*25 = 50
    
    # But we can do better. Let's try a different split.
    # Split 7 into 2+2+2+1: 3 Strassen blocks + 1 schoolbook block
    # Each Strassen: rank 7, each schoolbook 2x2x1: rank 4
    # Per M_{2,2,7}: 3*7 + 4 = 25
    # Total: 50
    
    # Alternative: Use a direct search approach.
    # Let's implement a random local search starting from schoolbook.
    
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = len(U_sb[0])
    
    # Try to reduce rank by finding linear dependencies among the rank-1 tensors
    # This is equivalent to finding a lower-rank decomposition.
    
    # Approach: Start with schoolbook and try random rank-reducing operations.
    # A rank-reducing operation: pick two terms r1, r2 and try to replace them
    # with a single term that approximates their combined effect.
    
    # Actually, let's try a more systematic approach.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's try to construct a rank-35 decomposition using the following idea:
    # Group the 7 columns into {0,1}, {2,3}, {4,5}, {6}
    # For each pair of columns, use a 2x2x2 Strassen-like decomposition
    # For the single column, use schoolbook
    
    # For M_{2,4,2} (multiplying 2x4 by 4x2):
    # Split j into 2+2: C = A1@B1 + A2@B2 where each is M_{2,2,2}
    # Each M_{2,2,2} has rank 7 (Strassen)
    # Total for M_{2,4,2}: 14
    
    # For M_{2,4,1} (multiplying 2x4 by 4x1):
    # This is just 8 scalar multiplications, rank 8
    
    # Total: 3*14 + 8 = 50
    
    # Hmm, that's still 50. Let me think differently.
    
    # Actually, let me try to use a smarter decomposition for M_{2,2,p}.
    # For M_{2,2,p}, we can use the following rank-3p decomposition:
    # M_r for r=1..3p where:
    # - For r=1..p: M_r = A[0,0] * B[0,r-1]  (but this is just schoolbook)
    
    # Let me try a different approach: use the fact that we can share
    # linear combinations across different k values.
    
    # For M_{2,2,p}:
    # C[0,k] = A00*B0k + A01*B1k
    # C[1,k] = A10*B0k + A11*B1k
    #
    # Define:
    # X1 = A00 + A11, X2 = A00 - A11, X3 = A01, X4 = A10
    # Y1 = B0k + B1k, Y2 = B0k - B1k, Y3 = B1k, Y4 = B0k
    #
    # Then:
    # C[0,k] = (X1+X2)/2 * (Y1+Y4)/2 + X3*Y3  -- this is getting complicated
    
    # Let me just implement a random search with a time limit.
    
    start = time.time()
    deadline = start + 130.0
    
    # Random search: try to find a lower-rank decomposition
    # by randomly perturbing and then trying to reduce rank.
    
    # Actually, let's try a structured approach:
    # Split the 4 middle index into 2+2.
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
    # Each term is M_{2,2,7}.
    
    # For M_{2,2,7}, let's try to construct a low-rank decomposition.
    # Use the following 19-term decomposition (if it works):
    # Based on the idea that rank(M_{2,2,p}) <= 3p-2 for p>=2
    
    def _m22p_decomp(p):
        """Try to build a low-rank decomposition for M_{2,2,p}."""
        n = m = 2
        # Schoolbook as fallback
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        
        # Try a structured decomposition
        # For M_{2,2,p}, use the following:
        # Split p columns into pairs and a possible remainder
        num_pairs = p // 2
        remainder = p % 2
        
        # For each pair of columns, use Strassen (rank 7)
        # For remainder column, use schoolbook (rank 4)
        total_rank = num_pairs * 7 + remainder * 4
        
        if total_rank >= n * m * p:
            return U_sb, V_sb, W_sb
        
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        offset = 0
        col = 0
        for _ in range(num_pairs):
            k1 = col
            k2 = col + 1
            col += 2
            # Strassen for columns k1, k2
            # M1 = (A00+A11)(B0k1+B0k2)  -- wait, this isn't quite right for rectangular
            
            # Actually for M_{2,2,2} with columns k1,k2:
            # C[0,k1] = A00*B0k1 + A01*B1k1
            # C[0,k2] = A00*B0k2 + A01*B1k2
            # C[1,k1] = A10*B0k1 + A11*B1k1
            # C[1,k2] = A10*B0k2 + A11*B1k2
            #
            # Strassen:
            # P1 = (A00+A11)*(B0k1+B0k2)  -- no, this mixes k1 and k2
            #
            # Standard Strassen for 2x2x2:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            #
            # Here B00=B0k1, B01=B0k2, B10=B1k1, B11=B1k2
            # C[0,k1] = M1+M4-M5+M7
            # C[0,k2] = M3+M5
            # C[1,k1] = M2+M4
            # C[1,k2] = M1-M2+M3+M6
            
            # M1 = (A00+A11)(B0k1+B1k2)
            U[0][offset+0] = 1.0; U[3][offset+0] = 1.0
            V[0][offset+0] = 1.0; V[3][offset+0] = 1.0
            W[k1*2+0][offset+0] = 1.0; W[k2*2+1][offset+0] = 1.0
            
            # M2 = (A10+A11)B0k1
            U[2][offset+1] = 1.0; U[3][offset+1] = 1.0
            V[0][offset+1] = 1.0
            W[k1*2+1][offset+1] = 1.0
            
            # M3 = A00(B0k2-B1k2)
            U[0][offset+2] = 1.0
            V[1][offset+2] = 1.0; V[3][offset+2] = -1.0
            W[k2*2+0][offset+2] = 1.0
            
            # M4 = A11(B1k1-B0k1)
            U[3][offset+3] = 1.0
            V[2][offset+3] = 1.0; V[0][offset+3] = -1.0
            W[k1*2+0][offset+3] = 1.0; W[k1*2+1][offset+3] = 1.0
            
            # M5 = (A00+A01)B1k2
            U[0][offset+4] = 1.0; U[1][offset+4] = 1.0
            V[3][offset+4] = 1.0
            W[k1*2+0][offset+4] = -1.0; W[k2*2+0][offset+4] = 1.0
            
            # M6 = (A10-A00)(B0k1+B0k2)
            U[2][offset+5] = 1.0; U[0][offset+5] = -1.0
            V[0][offset+5] = 1.0; V[1][offset+5] = 1.0
            W[k2*2+1][offset+5] = 1.0
            
            # M7 = (A01-A11)(B1k1+B1k2)
            U[1][offset+6] = 1.0; U[3][offset+6] = -1.0
            V[2][offset+6] = 1.0; V[3][offset+6] = 1.0
            W[k1*2+0][offset+6] = 1.0; W[k2*2+1][offset+6] = 1.0
            
            offset += 7
        
        # Handle remainder column
        if remainder == 1:
            k = col
            # Schoolbook for 2x2x1: 4 terms
            U[0][offset+0] = 1.0; V[0][offset+0] = 1.0; W[k*2+0][offset+0] = 1.0
            U[1][offset+1] = 1.0; V[1][offset+1] = 1.0; W[k*2+0][offset+1] = 1.0
            U[2][offset+2] = 1.0; V[0][offset+2] = 1.0; W[k*2+1][offset+2] = 1.0
            U[3][offset+3] = 1.0; V[1][offset+3] = 1.0; W[k*2+1][offset+3] = 1.0
            offset += 4
        
        return U, V, W
    
    # Build M_{2,4,7} by splitting j into 2+2
    U1, V1, W1 = _m22p_decomp(p)  # for A[:,0:2] @ B[0:2,:]
    U2, V2, W2 = _m22p_decomp(p)  # for A[:,2:4] @ B[2:4,:]
    
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # First half: A[:,0:2] @ B[0:2,:]
    for i in range(n):
        for j in range(2):
            for r in range(rank1):
                U[i*m+j][r] = U1[i*2+j][r]
    for j in range(2):
        for k in range(p):
            for r in range(rank1):
                V[j*p+k][r] = V1[j*2+k][r]
    for k in range(p):
        for i in range(n):
            for r in range(rank1):
                W[k*n+i][r] = W1[k*2+i][r]
    
    # Second half: A[:,2:4] @ B[2:4,:]
    for i in range(n):
        for j in range(2):
            for r in range(rank2):
                U[i*m+(j+2)][rank1+r] = U2[i*2+j][r]
    for j in range(2):
        for k in range(p):
            for r in range(rank2):
                V[(j+2)*p+k][rank1+r] = V2[j*2+k][r]
    for k in range(p):
        for i in range(n):
            for r in range(rank2):
                W[k*n+i][rank1+r] = W2[k*2+i][r]
    
    # Verify
    if _verify_decomposition(n, m, p, U, V, W):
        return U, V, W
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    U, V, W = _build_2x4x7_decomposition()
    
    rank = len(U[0])
    
    # Verify the decomposition
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
