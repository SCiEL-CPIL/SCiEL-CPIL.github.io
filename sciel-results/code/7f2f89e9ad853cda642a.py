import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 100.0

    n, m, p = 3, 3, 7

    # Strassen 2x2x2: U is 4x7, V is 4x7, W is 4x7
    # Convention: U[i*m+j, r] for A (2x2), V[j*p+k, r] for B (2x2), W[k*n+i, r] for C (2x2)
    # So U index = i*2+j, V index = j*2+k, W index = k*2+i
    def strassen_2x2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1 = (A11+A22)(B11+B22)
        U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
        # M2 = A22(B21-B11)
        U[3][1]=1; V[1][1]=1; V[0][1]=-1; W[2][1]=1; W[3][1]=-1
        # M3 = (A11-A22)(B22+B12)
        U[0][2]=1; U[3][2]=-1; V[2][2]=1; V[3][2]=1; W[1][2]=1; W[2][2]=1
        # M4 = A11(B12-B22)
        U[0][3]=1; V[1][3]=1; V[2][3]=-1; W[1][3]=1; W[2][3]=-1
        # M5 = (A12+A22)(B11+A21)
        U[1][4]=1; U[3][4]=1; V[0][4]=1; V[3][4]=1; W[0][4]=1; W[1][4]=1
        # M6 = (A11+A12)(B22-B21)
        U[0][5]=1; U[1][5]=1; V[2][5]=1; V[1][5]=-1; W[2][5]=1; W[3][5]=1
        # M7 = (A21-A11)(B11+B12)
        U[2][6]=1; U[0][6]=-1; V[0][6]=1; V[1][6]=1; W[0][6]=1; W[1][6]=1
        return U, V, W

    def verify_strassen():
        U, V, W = strassen_2x2x2()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = sum(U[i*2+j][r] * V[j*2+k][r] * W[k*2+i][r] for r in range(7))
                    if val != 1:
                        print(f"Strassen FAIL at i={i},j={j},k={k}: got {val}")
                        return False
        print("Strassen 2x2x2 verified OK")
        return True

    # Verify Strassen is correct
    if not verify_strassen():
        # Fallback to schoolbook if Strassen is wrong
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

    # Now build the 3x3x7 decomposition:
    # Split as: (2x2x4) + (1x3x3)
    # 2x2x4: split p=4 into two 2x2x2 blocks, each using Strassen (rank 7)
    #   Total rank for 2x2x4: 7+7 = 14
    # 1x3x3: schoolbook, rank 9
    # Total rank: 14 + 9 = 23

    # Actually, let me reconsider. Split m=3 into m1=2, m2=1:
    # Part 1: 3x2x7 (n=3, m=2, p=7)
    # Part 2: 3x1x7 (n=3, m=1, p=7)
    
    # For 3x2x7: split n=3 into n1=2, n2=1
    #   Part 1a: 2x2x7 -> split p=7 into p1=4, p2=3
    #     2x2x4: two Strassen 2x2x2, rank 14
    #     2x2x3: split p=3 into p1=2, p2=1: Strassen(7) + schoolbook(4) = 11
    #     Total 2x2x7: 14+11 = 25
    #   Part 1b: 1x2x7: schoolbook, rank 14
    #   Total 3x2x7: 25+14 = 39
    # Part 2: 3x1x7: schoolbook, rank 21
    # Total: 39+21 = 60. Worse than 63? No, 60 < 63.
    
    # Let me try splitting p=7 into p1=4, p2=3:
    # Part 1: 3x3x4: split n=3 into 2+1
    #   2x3x4: split m=3 into 2+1
    #     2x2x4: two Strassen, rank 14
    #     2x1x4: schoolbook, rank 8
    #     Total 2x3x4: 22
    #   1x3x4: schoolbook, rank 12
    #   Total 3x3x4: 34
    # Part 2: 3x3x3: schoolbook, rank 27
    # Total: 34+27 = 61

    # Let me try a different approach. Split n=3 into 2+1:
    # Part 1: 2x3x7: split m=3 into 2+1
    #   2x2x7: split p=7 into 4+3
    #     2x2x4: two Strassen (2x2x2), rank 14
    #     2x2x3: split p=3 into 2+1: Strassen(7) + 2x2x1(4) = 11
    #     Total 2x2x7: 25
    #   2x1x7: schoolbook, rank 14
    #   Total 2x3x7: 39
    # Part 2: 1x3x7: schoolbook, rank 21
    # Total: 60

    # Better: use Strassen more aggressively.
    # For 2x2x3: can we do better than 11?
    # 2x2x3: The known bound is R(2,2,3) <= 10 (from Strassen + 1 extra).
    # Actually R(2,2,3) = 10 is known? Let me check.
    # R(2,2,2) = 7, R(2,2,3) <= 7+4 = 11 by adding a 2x2x1.
    # But there might be a 10-rank decomposition.
    
    # Let me just go with the simple approach: split 3x3x7 into 2x3x7 + 1x3x7
    # 2x3x7: split into 2x2x7 + 2x1x7
    # 2x2x7: split into 2x2x4 + 2x2x3
    # 2x2x4: two Strassen 2x2x2 = rank 14
    # 2x2x3: Strassen 2x2x2 (rank 7) + 2x2x1 (rank 4) = rank 11
    # 2x2x7 total: 25
    # 2x1x7: schoolbook = rank 14
    # 2x3x7 total: 39
    # 1x3x7: schoolbook = rank 21
    # Grand total: 60

    # Even better: for 2x2x3, I can use a direct 10-rank decomposition if one exists.
    # But let me first get 60 working, then optimize.

    # Actually, let me try: split 3x3x7 as 3x3x4 + 3x3x3
    # 3x3x4: split n=3 into 2+1
    #   2x3x4: split m=3 into 2+1
    #     2x2x4: two Strassen = 14
    #     2x1x4: schoolbook = 8
    #     Total: 22
    #   1x3x4: schoolbook = 12
    #   Total 3x3x4: 34
    # 3x3x3: schoolbook = 27
    # Grand total: 61

    # The 60 approach is better. Let me implement it.

    def schoolbook(n, m, p):
        """Return schoolbook decomposition for n x m x p."""
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
        return U, V, W

    def combine_split_n(n1, n2, m, p, U1, V1, W1, U2, V2, W2):
        """Combine two decompositions split along n."""
        n = n1 + n2
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        r = r1 + r2
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        for i in range(n1):
            for j in range(m):
                U[i*m+j] = list(U1[i*m+j])
        for i in range(n2):
            for j in range(m):
                U[(n1+i)*m+j] = list(U2[i*m+j])
        for j in range(m):
            for k in range(p):
                V[j*p+k] = [v1+v2 for v1,v2 in zip(V1[j*p+k], V2[j*p+k])]
        for k in range(p):
            for i in range(n1):
                W[k*n+i] = list(W1[k*n1+i])
            for i in range(n2):
                W[k*n+n1+i] = list(W2[k*n2+i])
        return U, V, W

    def combine_split_m(n, m1, m2, p, U1, V1, W1, U2, V2, W2):
        """Combine two decompositions split along m."""
        m = m1 + m2
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        r = r1 + r2
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        for i in range(n):
            for j in range(m1):
                U[i*m+j] = list(U1[i*m1+j])
            for j in range(m2):
                U[i*m+m1+j] = list(U2[i*m2+j])
        for j in range(m1):
            for k in range(p):
                V[j*p+k] = list(V1[j*p+k])
        for j in range(m2):
            for k in range(p):
                V[(m1+j)*p+k] = list(V2[j*p+k])
        for k in range(p):
            for i in range(n):
                W[k*n+i] = [w1+w2 for w1,w2 in zip(W1[k*n+i], W2[k*n+i])]
        return U, V, W

    def combine_split_p(n, m, p1, p2, U1, V1, W1, U2, V2, W2):
        """Combine two decompositions split along p."""
        p = p1 + p2
        r1 = len(U1[0]) if U1 else 0
        r2 = len(U2[0]) if U2 else 0
        r = r1 + r2
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                U[i*m+j] = [u1+u2 for u1,u2 in zip(U1[i*m+j], U2[i*m+j])]
        for j in range(m):
            for k in range(p1):
                V[j*p+k] = list(V1[j*p1+k])
            for k in range(p2):
                V[j*p+p1+k] = list(V2[j*p2+k])
        for k in range(p1):
            for i in range(n):
                W[k*n+i] = list(W1[k*n+i])
        for k in range(p2):
            for i in range(n):
                W[(p1+k)*n+i] = list(W2[k*n+i])
        return U, V, W

    def verify_decomp(U, V, W, n, m, p):
        """Verify decomposition correctness."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(len(U[0])))
                    if val != 1:
                        return False
        return True

    # Build 2x2x4: two Strassen 2x2x2 blocks
    def get_2x2x4():
        U1, V1, W1 = strassen_2x2x2()
        U2, V2, W2 = strassen_2x2x2()
        return combine_split_p(2, 2, 2, 2, U1, V1, W1, U2, V2, W2)

    # Build 2x2x3: Strassen 2x2x2 + schoolbook 2x2x1
    def get_2x2x3():
        U1, V1, W1 = strassen_2x2x2()
        U2, V2, W2 = schoolbook(2, 2, 1)
        return combine_split_p(2, 2, 2, 1, U1, V1, W1, U2, V2, W2)

    # Build 2x2x7: 2x2x4 + 2x2x3
    def get_2x2x7():
        U1, V1, W1 = get_2x2x4()
        U2, V2, W2 = get_2x2x3()
        return combine_split_p(2, 2, 4, 3, U1, V1, W1, U2, V2, W2)

    # Build 2x3x7: 2x2x7 + 2x1x7
    def get_2x3x7():
        U1, V1, W1 = get_2x2x7()
        U2, V2, W2 = schoolbook(2, 1, 7)
        return combine_split_m(2, 2, 1, 7, U1, V1, W1, U2, V2, W2)

    # Build 3x3x7: 2x3x7 + 1x3x7
    def get_3x3x7():
        U1, V1, W1 = get_2x3x7()
        U2, V2, W2 = schoolbook(1, 3, 7)
        return combine_split_n(2, 1, 3, 7, U1, V1, W1, U2, V2, W2)

    U, V, W = get_3x3x7()

    # Verify
    if not verify_decomp(U, V, W, n, m, p):
        print("Decomposition FAILED verification, falling back to schoolbook")
        U, V, W = schoolbook(n, m, p)

    rank = len(U[0])
    print(f"Final rank: {rank}")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
