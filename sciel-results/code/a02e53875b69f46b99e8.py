"""Matrix multiplication 4x4x8 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, n, m, p):
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_2x2x2():
        """Return a verified rank-7 decomposition for 2x2x2."""
        # Verified decomposition from Strassen's algorithm
        U = np.array([
            [1, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, -1, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1],
        ], dtype=float)
        V = np.array([
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, -1, 1, 0, 1],
        ], dtype=float)
        W = np.array([
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0],
        ], dtype=float)
        
        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] for c in range(7))
                    if abs(s - 1) > 1e-9:
                        return None
        return U, V, W
    
    def kron_decomp(U1, V1, W1, U2, V2, W2):
        """
        Combine two tensor decompositions using Kronecker product.
        If T1[i1,j1,k1] = sum_c1 U1[i1*a1+j1,c1]*V1[j1*b1+k1,c1]*W1[k1*a1+i1,c1] = 1
        and T2[i2,j2,k2] = sum_c2 U2[i2*a2+j2,c2]*V2[j2*b2+k2,c2]*W2[k2*a2+i2,c2] = 1
        then T1⊗T2[i1*a2+i2, j1*b2+j2, k1*c2+k2] = 1 can be decomposed with
        rank = rank1 * rank2.
        
        For our case: T1 is 2x2x2 (a1=2,b1=2,c1=2) and T2 is 2x2x2 (a2=2,b2=2,c2=2)
        Result is 4x4x4 with rank = 7*7 = 49.
        """
        a1, b1, c1 = U1.shape[0]//2, 2, 2  # For 2x2x2
        a2, b2, c2 = U2.shape[0]//2, 2, 2
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 * r2
        
        U = np.zeros((a1*a2*b1*b2, r))  # (4, 49) for 4x4x4
        V = np.zeros((b1*b2*c1*c2, r))  # (4, 49)
        W = np.zeros((c1*c2*a1*a2, r))  # (4, 49)
        
        for c1_idx in range(r1):
            for c2_idx in range(r2):
                c = c1_idx * r2 + c2_idx
                for i1 in range(a1):
                    for j1 in range(b1):
                        for i2 in range(a2):
                            for j2 in range(b2):
                                U[(i1*a2+i2)*b1*b2 + j1*b2+j2, c] = U1[i1*b1+j1, c1_idx] * U2[i2*b2+j2, c2_idx]
                for j1 in range(b1):
                    for k1 in range(c1):
                        for j2 in range(b2):
                            for k2 in range(c2):
                                V[(j1*b2+j2)*c1*c2 + k1*c2+k2, c] = V1[j1*c1+k1, c1_idx] * V2[j2*c2+k2, c2_idx]
                for k1 in range(c1):
                    for i1 in range(a1):
                        for k2 in range(c2):
                            for i2 in range(a2):
                                W[(k1*c2+k2)*a1*a2 + i1*a2+i2, c] = W1[k1*a1+i1, c1_idx] * W2[k2*a2+i2, c2_idx]
        
        return U, V, W
    
    def extend_4x4x4_to_4x4x8(U444, V444, W444):
        """
        Extend a 4x4x4 decomposition to 4x4x8 by duplicating the last dimension.
        T[i,j,k] = 1 for k in {0,...,7}.
        We can use the same decomposition for k=0..3 and k=4..7.
        This doubles the rank: 49 -> 98.
        """
        r = U444.shape[1]
        U = np.zeros((4*4, 2*r))
        V = np.zeros((4*8, 2*r))
        W = np.zeros((8*4, 2*r))
        
        # For k in 0..3, use the original decomposition
        for c in range(r):
            for ij in range(16):
                U[ij, c] = U444[ij, c]
            for jk in range(16):
                V[jk, c] = V444[jk, c]
            for ki in range(16):
                W[ki, c] = W444[ki, c]
        
        # For k in 4..7, use the same decomposition shifted
        for c in range(r):
            for ij in range(16):
                U[ij, r+c] = U444[ij, c]
            for j in range(4):
                for k in range(4):
                    V[j*8+(k+4), r+c] = V444[j*4+k, c]
            for k in range(4):
                for i in range(4):
                    W[(k+4)*4+i, r+c] = W444[k*4+i, c]
        
        return U, V, W
    
    # Try to build a rank-98 decomposition using Kronecker product of two Strassen decomps
    s2 = strassen_2x2x2()
    
    best_rank = n * m * p  # 128
    best_decomp = None
    
    # Schoolbook as baseline
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_decomp = (U_sb.tolist(), V_sb.tolist(), W_sb.tolist())
    
    # Try Kronecker product approach: 2x2x2 ⊗ 2x2x2 = 4x4x4 with rank 49
    if s2 is not None:
        U444, V444, W444 = kron_decomp(s2[0], s2[1], s2[2], s2[0], s2[1], s2[2])
        
        # Verify 4x4x4 decomposition
        valid_444 = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = sum(U444[i*4+j, c] * V444[j*4+k, c] * W444[k*4+i, c] for c in range(49))
                    if abs(s - 1) > 1e-9:
                        valid_444 = False
                        break
                if not valid_444:
                    break
            if not valid_444:
                break
        
        if valid_444:
            # Extend to 4x4x8
            U448, V448, W448 = extend_4x4x4_to_4x4x8(U444, V444, W444)
            
            # Verify 4x4x8 decomposition
            valid_448 = True
            for i in range(4):
                for j in range(4):
                    for k in range(8):
                        s = sum(U448[i*4+j, c] * V448[j*8+k, c] * W448[k*4+i, c] for c in range(98))
                        if abs(s - 1) > 1e-9:
                            valid_448 = False
                            break
                    if not valid_448:
                        break
                if not valid_448:
                    break
            
            if valid_448:
                best_rank = 98
                best_decomp = (U448.tolist(), V448.tolist(), W448.tolist())
    
    U, V, W = best_decomp
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    r = len(result["U"][0])
    print(f"Rank: {r}")
    
    # Verify
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    valid = False
                    print(f"Failed at ({i},{j},{k}): sum={s}")
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
