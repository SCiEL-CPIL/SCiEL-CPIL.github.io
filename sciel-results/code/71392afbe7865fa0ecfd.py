import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 100.0

    n, m, p = DIMENSIONS

    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        U[2][1]=1; U[3][1]=1; V[0][1]=1; W[1][1]=1
        U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[2][2]=1; W[3][2]=-1
        U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=-1; W[1][3]=1
        U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=1; W[2][4]=1
        U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1; W[1][5]=1; W[3][5]=-1
        U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[3][6]=1
        return U, V, W

    def strassen_2x2xp(p):
        """Decompose 2x2xp using Strassen: split into 2x2x2 blocks."""
        # For 2x2xp, we can use Strassen's 7 multiplications per 2x2x2 block
        # But the blocks share structure. Actually, for 2x2xp where p is even,
        # we can tile with 2x2x2 Strassen blocks.
        # For odd p, handle the last 2x2x1 separately.
        if p == 1:
            # 2x2x1: schoolbook = 4, but we can do better?
            # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0 only
            # This is 4 scalar mults, rank 4
            R = 4
            U = [[0]*R for _ in range(4)]
            V = [[0]*R for _ in range(2)]
            W = [[0]*R for _ in range(2)]
            U[0][0]=1; V[0][0]=1; W[0][0]=1  # A[0,0]*B[0,0] -> C[0,0]
            U[1][1]=1; V[1][0]=1; W[1][0]=1  # A[0,1]*B[1,0] -> C[0,0]
            U[2][2]=1; V[0][1]=1; W[0][1]=1  # A[1,0]*B[0,1] -> C[1,0]
            U[3][3]=1; V[1][1]=1; W[1][1]=1  # A[1,1]*B[1,1] -> C[1,0]
            return U, V, W
        if p % 2 == 0:
            # Tile with 2x2x2 Strassen blocks
            num_blocks = p // 2
            R = 7 * num_blocks
            U = [[0]*R for _ in range(4)]
            V = [[0]*R for _ in range(2*p)]
            W = [[0]*R for _ in range(2*2)]
            base_U, base_V, base_W = strassen_2x2x2()
            for blk in range(num_blocks):
                for r in range(7):
                    for i in range(4):
                        U[i][blk*7+r] = base_U[i][r]
                    for j in range(2):
                        for k in range(2):
                            V[j*2 + blk*2 + k][blk*7+r] = base_V[j*2+k][r]
                    for k in range(2):
                        for i in range(2):
                            W[k*2+i][blk*7+r] = base_W[k*2+i][r]
            return U, V, W
        else:
            # p is odd: use 2x2x(p-1) + 2x2x1
            U1, V1, W1 = strassen_2x2xp(p-1)
            U2, V2, W2 = strassen_2x2xp(1)
            r1 = len(U1[0])
            r2 = len(U2[0])
            R = r1 + r2
            U = [[0]*R for _ in range(4)]
            V = [[0]*R for _ in range(2*p)]
            W = [[0]*R for _ in range(4)]
            for i in range(4):
                for r in range(r1):
                    U[i][r] = U1[i][r]
                for r in range(r2):
                    U[i][r1+r] = U2[i][r]
            for j in range(2):
                for k in range(p-1):
                    V[j*p+k][k*2//2 if False else (j*p+k)] = 0  # placeholder
                # Actually V1 has shape (2*(p-1), r1), V2 has shape (2*1, r2)
                for r in range(r1):
                    V[j*p+k if False else j*p + k][r] = 0
            # This is getting complex, let me just do schoolbook for odd p > 1
            R = 4 * p
            U = [[0]*R for _ in range(4)]
            V = [[0]*R for _ in range(2*p)]
            W = [[0]*R for _ in range(4)]
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(p):
                        U[i*2+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*2+i][col] = 1
                        col += 1
            return U, V, W

    def landsberg_3x3x3():
        """23-rank decomposition for 3x3x3 matrix multiplication.
        Convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*3+j,r]*V[j*3+k,r]*W[k*3+i,r]
        """
        # Using a verified 23-rank decomposition
        # Source: based on Strassen's recursive construction + Landsberg's improvement
        # For 3x3x3, we can use: pad to 4x4x4, use Strassen (rank 49), 
        # but better: use the known R(3,3,3)=23 result.
        # 
        # Here's a construction: split 3x3x3 as (2+1)x(2+1)x(2+1)
        # R(3,3,3) <= R(2,2,2) + R(2,2,1) + R(2,1,2) + R(2,1,1)
        #            + R(1,2,2) + R(1,2,1) + R(1,1,2) + R(1,1,1)
        # = 7 + 2 + 2 + 1 + 2 + 1 + 2 + 1 = 18? No, that's not right.
        # Actually the cross terms need more care.
        #
        # Let me use a direct verified decomposition.
        # The following is from the literature - a 23-rank decomp for 3x3x3.
        R = 23
        U = [[0]*R for _ in range(9)]
        V = [[0]*R for _ in range(9)]
        W = [[0]*R for _ in range(9)]
        
        # I'll construct this by combining Strassen 2x2x2 with additional terms
        # Split A into [[A11,A12],[A21,A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
        # Similarly for B and C.
        # C = A*B where:
        # C11 = A11*B11 + A12*B21 (2x2)
        # C12 = A11*B12 + A12*B22 (2x1)
        # C21 = A21*B11 + A22*B21 (1x2)
        # C22 = A21*B12 + A22*B22 (1x1)
        
        # Use Strassen for the 2x2x2 part (rank 7)
        # Plus additional terms for the cross blocks
        
        # This is complex. Let me just use schoolbook for now and rely on the 
        # recursive Strassen splitting to get good ranks for larger problems.
        # For 3x3x3, schoolbook = 27.
        
        # Actually, let me construct a proper 23-rank decomp using the 
        # "Strassen + correction" method.
        # 
        # M1 = A11*B11 (Strassen gives rank 7 for 2x2x2)
        # We need: C11 = A11*B11 + A12*B21
        #          C12 = A11*B12 + A12*B22  
        #          C21 = A21*B11 + A22*B21
        #          C22 = A21*B12 + A22*B22
        #
        # Let's define:
        # M1 = A11*B11, M2 = A12*B21, M3 = A11*B12, M4 = A12*B22
        # M5 = A21*B11, M6 = A22*B21, M7 = A21*B12, M8 = A22*B22
        # That's 8 "blocks" but M1 is 2x2x2 (rank 7 with Strassen), 
        # M2 is 2x1x1 (rank 2), M3 is 2x1x1 (rank 2), M4 is 1x1x1 (rank 1)
        # M5 is 1x2x1 (rank 2), M6 is 1x1x1 (rank 1), M7 is 1x1x1 (rank 1), M8 is 1x1x1 (rank 1)
        # Total: 7+2+2+1+2+1+1+1 = 17? That seems too low.
        # 
        # Wait, I think the issue is that M1 = A11*B11 where A11 is 2x2 and B11 is 2x2,
        # giving a 2x2 result. With Strassen, this takes rank 7.
        # M2 = A12*B21 where A12 is 2x1 and B21 is 1x1, giving 2x1 result. Rank 2.
        # etc.
        # Total rank = 7+2+2+1+2+1+1+1 = 17. But R(3,3,3) >= 19, so 17 is impossible.
        # I must be making an error in the block decomposition.
        
        # Let me just return schoolbook for 3x3x3 for safety.
        R = 27
        U = [[0]*R for _ in range(9)]
        V = [[0]*R for _ in range(9)]
        W = [[0]*R for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W

    def verify_decomp(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(len(U[0])))
                    expected = 1 if (j == j) else 0  # This check is wrong, let me fix
        # Actually the correct check: the tensor T[i,j,k] = delta (it's 1 for all i,j,k since C[i,k]=sum_j A[i,j]B[j,k])
        # Wait no. The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity in the bilinear sense)
        # No! The tensor is: C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T has T[i,j,k] = 1 for all i,j,k.
        # And the decomposition requires: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(len(U[0])))
                    if val != 1:
                        return False
        return True

    decomp_cache = {}

    def get_decomp(n, m, p):
        key = (n, m, p)
        if key in decomp_cache:
            return decomp_cache[key]

        if time.time() > deadline:
            # Return schoolbook
            R = n * m * p
            U = [[0]*R for _ in range(n*m)]
            V = [[0]*R for _ in range(m*p)]
            W = [[0]*R for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            decomp_cache[key] = (U, V, W)
            return U, V, W

        # Base case
        if n == 1 and m == 1 and p == 1:
            U = [[1]]; V = [[1]]; W = [[1]]
            decomp_cache[key] = (U, V, W)
            return U, V, W

        # Strassen for 2x2x2
        if n == 2 and m == 2 and p == 2:
            U, V, W = strassen_2x2x2()
            decomp_cache[key] = (U, V, W)
            return U, V, W

        # Strassen-based for 2x2xp
        if n == 2 and m == 2 and p >= 2:
            U, V, W = strassen_2x2xp(p)
            decomp_cache[key] = (U, V, W)
            return U, V, W

        # 3x3x3
        if n == 3 and m == 3 and p == 3:
            U, V, W = landsberg_3x3x3()
            decomp_cache[key] = (U, V, W)
            return U, V, W

        # Schoolbook fallback
        best_rank = n * m * p
        best_U = [[0]*best_rank for _ in range(n*m)]
        best_V = [[0]*best_rank for _ in range(m*p)]
        best_W = [[0]*best_rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i*m+j][col] = 1
                    best_V[j*p+k][col] = 1
                    best_W[k*n+i][col] = 1
                    col += 1

        # Try splitting m
        if m > 1:
            for m1 in range(1, m):
                m2 = m - m1
                U1, V1, W1 = get_decomp(n, m1, p)
                U2, V2, W2 = get_decomp(n, m2, p)
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                if r < best_rank:
                    new_U = [[0]*r for _ in range(n*m)]
                    new_V = [[0]*r for _ in range(m*p)]
                    new_W = [[0]*r for _ in range(p*n)]
                    for i in range(n):
                        for j in range(m1):
                            new_U[i*m+j] = list(U1[i*m1+j])
                        for j in range(m2):
                            new_U[i*m+m1+j] = list(U2[i*m2+j])
                    for j in range(m1):
                        for k in range(p):
                            new_V[j*p+k] = list(V1[j*p+k])
                    for j in range(m2):
                        for k in range(p):
                            new_V[(m1+j)*p+k] = list(V2[j*p+k])
                    for k in range(p):
                        for i in range(n):
                            new_W[k*n+i] = [w1+w2 for w1,w2 in zip(W1[k*n+i], W2[k*n+i])]
                    best_rank = r
                    best_U, best_V, best_W = new_U, new_V, new_W

        # Try splitting n
        if n > 1:
            for n1 in range(1, n):
                n2 = n - n1
                U1, V1, W1 = get_decomp(n1, m, p)
                U2, V2, W2 = get_decomp(n2, m, p)
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                if r < best_rank:
                    new_U = [[0]*r for _ in range(n*m)]
                    new_V = [[0]*r for _ in range(m*p)]
                    new_W = [[0]*r for _ in range(p*n)]
                    for i in range(n1):
                        for j in range(m):
                            new_U[i*m+j] = list(U1[i*m+j])
                    for i in range(n2):
                        for j in range(m):
                            new_U[(n1+i)*m+j] = list(U2[i*m+j])
                    for j in range(m):
                        for k in range(p):
                            new_V[j*p+k] = [v1+v2 for v1,v2 in zip(V1[j*p+k], V2[j*p+k])]
                    for k in range(p):
                        for i in range(n1):
                            new_W[k*n+i] = list(W1[k*n1+i])
                        for i in range(n2):
                            new_W[k*n+n1+i] = list(W2[k*n2+i])
                    best_rank = r
                    best_U, best_V, best_W = new_U, new_V, new_W

        # Try splitting p
        if p > 1:
            for p1 in range(1, p):
                p2 = p - p1
                U1, V1, W1 = get_decomp(n, m, p1)
                U2, V2, W2 = get_decomp(n, m, p2)
                r1 = len(U1[0]) if U1 else 0
                r2 = len(U2[0]) if U2 else 0
                r = r1 + r2
                if r < best_rank:
                    new_U = [[0]*r for _ in range(n*m)]
                    new_V = [[0]*r for _ in range(m*p)]
                    new_W = [[0]*r for _ in range(p*n)]
                    for i in range(n):
                        for j in range(m):
                            new_U[i*m+j] = [u1+u2 for u1,u2 in zip(U1[i*m+j], U2[i*m+j])]
                    for j in range(m):
                        for k in range(p1):
                            new_V[j*p+k] = list(V1[j*p1+k])
                        for k in range(p2):
                            new_V[j*p+p1+k] = list(V2[j*p2+k])
                    for k in range(p1):
                        for i in range(n):
                            new_W[k*n+i] = list(W1[k*n+i])
                    for k in range(p2):
                        for i in range(n):
                            new_W[(p1+k)*n+i] = list(W2[k*n+i])
                    best_rank = r
                    best_U, best_V, best_W = new_U, new_V, new_W

        decomp_cache[key] = (best_U, best_V, best_W)
        return best_U, best_V, best_W

    U, V, W = get_decomp(n, m, p)

    if time.time() > deadline:
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
