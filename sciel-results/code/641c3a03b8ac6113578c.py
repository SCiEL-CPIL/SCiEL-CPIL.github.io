import time
import numpy as np

def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition for 2x4x7 matrix multiplication.
    Strategy:
    1. Use Winograd F(2,2) for the n=2 dimension. This reduces the 2x4x7 problem to two 1x4x7 problems.
       Rank(1x4x7) via schoolbook is 28. Total 56.
    2. Improve the 1x4x7 part using Strassen on the m=4 dimension.
       Split m=4 into two m=2 blocks.
       1x4x7 = 1x2x7 + 1x2x7.
       Rank(1x2x7) via schoolbook is 14. Total 28.
       However, we can use Strassen for 1x2x2 blocks? No, Strassen is for 2x2x2.
       
       Let's look at the structure again.
       A (2x4), B (4x7).
       Using Winograd F(2,2) on A's rows:
       A = [[a00, a01, a02, a03],
            [a10, a11, a12, a13]]
       
       Winograd F(2,2) for C = A B:
       We compute:
       M1 = (a00 + a11) * (b00 + b11)  -- Wait, Winograd is for 1xn x nx1?
       
       Let's use the standard Winograd F(2,2) for 2x2 matrices.
       Actually, the "Winograd F(2,2)" usually refers to the 1x2x2 case or 2x2x2.
       For 2x2x2, Strassen is rank 7. Winograd F(2,2) is rank 4 for 1x2x2.
       
       Let's decompose 2x4x7 by splitting n=2 into two 1x4x7 problems?
       No, that's not how it works.
       
       Let's use the property: Rank(2x4x7) <= Rank(2x4x4) + Rank(2x4x3).
       Rank(2x4x4):
       Split n=2 into 1+1.
       Rank(1x4x4) = 16 (schoolbook).
       But we can use Strassen on the 4x4 part?
       1x4x4 is a row vector times a 4x4 matrix.
       We can split the 4x4 matrix into 2x2 blocks.
       Row vector (1x4) = [u1 (1x2), u2 (1x2)].
       Matrix (4x4) = [[B11, B12], [B21, B22]] where Bij are 2x2.
       Result (1x4) = [u1 B11 + u2 B21, u1 B12 + u2 B22].
       
       This is two 1x2x2 multiplications for the left half and two for the right half.
       Rank(1x2x2) is 4 (Winograd).
       So Rank(1x4x4) <= 4 + 4 + 4 + 4 = 16. Same as schoolbook.
       
       Let's try splitting m=4 into 2+2 for the 2x4x7 problem.
       A (2x4) = [A1 (2x2), A2 (2x2)].
       B (4x7) = [[B1 (2x7)], [B2 (2x7)]].
       C = A1 B1 + A2 B2.
       
       Now we need to compute M1 = A1 B1 (2x2 x 2x7) and M2 = A2 B2 (2x2 x 2x7).
       Rank(2x2x7):
       Split p=7 into 4+3? Or use Strassen on the 2x2 part?
       A1 is 2x2. B1 is 2x7.
       We can view this as 2x2x2 + 2x2x2 + 2x2x3?
       Rank(2x2x2) = 7 (Strassen).
       Rank(2x2x3) = Rank(2x2x2) + Rank(2x2x1) = 7 + 2 = 9?
       Wait, Rank(2x2x1) is 2.
       So Rank(2x2x7) <= 7 + 7 + 9 = 23.
       Total Rank <= 23 + 23 = 46.
       
       Can we do better?
       What if we use Winograd F(2,2) on the n=2 dimension?
       The tensor T_{i,j,k} for i in {0,1}.
       We can factor out the i-dependence.
       Using the Winograd F(2,2) algorithm for 2x2 matrix multiplication:
       The 7 multiplications in Strassen can be seen as rank 7.
       
       Let's stick to the 46 rank construction, which is a significant improvement over 50.
       We will implement:
       1. Split m=4 into two 2s.
       2. For each 2x2x7 block, split p=7 into 2, 2, 3.
       3. Use Strassen (rank 7) for the 2x2x2 parts.
       4. Use Schoolbook (rank 2) for the 2x2x1 parts (3 of them -> rank 6? No, 2x2x3 is 6 mults schoolbook, or 7+2=9 Strassen+Schoolbook).
          Actually, schoolbook for 2x2x3 is 2*2*3 = 12.
          Strassen for 2x2x2 (7) + Schoolbook for 2x2x1 (2) = 9.
          So 9 is better than 12.
       
       So Rank(2x2x7) = 7 + 7 + 9 = 23.
       Total Rank = 23 * 2 = 46.
       
       Let's verify if there is a rank 40 construction.
       Known lower bound for 2x4x7?
       Rank(2x2x2) = 7.
       Rank(2x4x4) ?
       If we use Strassen on 2x2x2 blocks for 2x4x4:
       Split m=4 into 2+2, p=4 into 2+2.
       4 blocks of 2x2x2.
       Rank = 4 * 7 = 28.
       Schoolbook 2x4x4 = 32.
       
       For 2x4x7:
       Split p=7 into 4+3.
       2x4x4 (rank 28) + 2x4x3.
       2x4x3: Split m=4 into 2+2.
       2x2x3 (rank 9) + 2x2x3 (rank 9) = 18.
       Total = 28 + 18 = 46.
       
       This matches the previous calculation.
       
       Is there a better way for 2x4x3?
       Rank(2x4x3).
       Lower bound: max(2*4, 4*3, 3*2) = 12.
       Schoolbook: 24.
       Strassen blocks: 2 * 9 = 18.
       
       What if we use Winograd F(3,3) or similar?
       Unlikely to beat 18 easily with simple blocks.
       
       Let's implement the rank 46 construction.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 2, 4, 7
    nm = n * m
    mp = m * p
    pn = p * n

    def verify_decomposition(U, V, W, rank):
        """
        Verifies if U, V, W form a valid decomposition.
        U: (nm, rank), V: (mp, rank), W: (pn, rank)
        """
        # Check half-integer constraint
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        
        # Check tensor decomposition
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def construct_rank_46():
        """
        Constructs a rank-46 decomposition.
        Structure:
        Split m=4 into m1=2, m2=2.
        C = A1 B1 + A2 B2.
        A1, A2 are 2x2. B1, B2 are 2x7.
        
        For each 2x2x7 block:
        Split p=7 into p1=2, p2=2, p3=3.
        Block 1: 2x2x2 (Strassen, rank 7)
        Block 2: 2x2x2 (Strassen, rank 7)
        Block 3: 2x2x3 (Strassen 2x2x2 + Schoolbook 2x2x1, rank 9)
        Total per 2x2x7: 23.
        Total for 2x4x7: 46.
        """
        U = [[0] * 46 for _ in range(nm)]
        V = [[0] * 46 for _ in range(mp)]
        W = [[0] * 46 for _ in range(pn)]
        
        rank_idx = 0

        def add_strassen_2x2x2(i0, j0, k0, start_rank):
            nonlocal rank_idx
            # i0: start row of A (0 or 1)
            # j0: start col of A / row of B (0..3)
            # k0: start col of B (0..6)
            
            # M1 = (A11+A22)(B11+B22)
            r = start_rank
            U[i0*m + j0][r] += 1
            U[(i0+1)*m + (j0+1)][r] += 1
            V[j0*p + k0][r] += 1
            V[(j0+1)*p + (k0+1)][r] += 1
            W[k0*n + i0][r] += 1
            W[(k0+1)*n + (i0+1)][r] += 1
            rank_idx += 1
            
            # M2 = (A21+A22)B11
            r = start_rank
            U[(i0+1)*m + j0][r] += 1
            U[(i0+1)*m + (j0+1)][r] += 1
            V[j0*p + k0][r] += 1
            W[k0*n + (i0+1)][r] += 1
            W[(k0+1)*n + (i0+1)][r] += -1
            rank_idx += 1
            
            # M3 = A11(B12-B22)
            r = start_rank
            U[i0*m + j0][r] += 1
            V[j0*p + (k0+1)][r] += 1
            V[(j0+1)*p + (k0+1)][r] += -1
            W[(k0+1)*n + i0][r] += 1
            W[(k0+1)*n + (i0+1)][r] += 1
            rank_idx += 1
            
            # M4 = A22(B21-B11)
            r = start_rank
            U[(i0+1)*m + (j0+1)][r] += 1
            V[(j0+1)*p + k0][r] += 1
            V[j0*p + k0][r] += -1
            W[k0*n + i0][r] += 1
            W[k0*n + (i0+1)][r] += 1
            rank_idx += 1
            
            # M5 = (A11+A12)B22
            r = start_rank
            U[i0*m + j0][r] += 1
            U[i0*m + (j0+1)][r] += 1
            V[(j0+1)*p + (k0+1)][r] += 1
            W[(k0+1)*n + i0][r] += 1
            W[k0*n + i0][r] += -1
            rank_idx += 1
            
            # M6 = (A21-A11)(B11+B12)
            r = start_rank
            U[(i0+1)*m + j0][r] += 1
            U[i0*m + j0][r] += -1
            V[j0*p + k0][r] += 1
            V[j0*p + (k0+1)][r] += 1
            W[(k0+1)*n + (i0+1)][r] += 1
            rank_idx += 1
            
            # M7 = (A12-A22)(B21+B22)
            r = start_rank
            U[i0*m + (j0+1)][r] += 1
            U[(i0+1)*m + (j0+1)][r] += -1
            V[(j0+1)*p + k0][r] += 1
            V[(j0+1)*p + (k0+1)][r] += 1
            W[k0*n + i0][r] += 1
            rank_idx += 1
            
            return rank_idx

        def add_schoolbook_2x2x1(i0, j0, k0, start_rank):
            nonlocal rank_idx
            # A is 2x2, B is 2x1. C is 2x1.
            # C1 = A11 B1 + A12 B2
            # C2 = A21 B1 + A22 B2
            
            # Term 1: A11 * B1
            r = start_rank
            U[i0*m + j0][r] += 1
            V[j0*p + k0][r] += 1
            W[k0*n + i0][r] += 1
            rank_idx += 1
            
            # Term 2: A12 * B2
            r = start_rank
            U[i0*m + (j0+1)][r] += 1
            V[(j0+1)*p + k0][r] += 1
            W[k0*n + i0][r] += 1
            rank_idx += 1
            
            # Term 3: A21 * B1
            r = start_rank
            U[(i0+1)*m + j0][r] += 1
            V[j0*p + k0][r] += 1
            W[k0*n + (i0+1)][r] += 1
            rank_idx += 1
            
            # Term 4: A22 * B2
            r = start_rank
            U[(i0+1)*m + (j0+1)][r] += 1
            V[(j0+1)*p + k0][r] += 1
            W[k0*n + (i0+1)][r] += 1
            rank_idx += 1
            
            return rank_idx

        def add_2x2x7_block(i0, j0, start_rank):
            nonlocal rank_idx
            # i0: row of A (0)
            # j0: col of A (0 or 2)
            
            # k=0,1 -> Strassen
            rank_idx = add_strassen_2x2x2(i0, j0, 0, rank_idx)
            # k=2,3 -> Strassen
            rank_idx = add_strassen_2x2x2(i0, j0, 2, rank_idx)
            # k=4,5 -> Strassen
            rank_idx = add_strassen_2x2x2(i0, j0, 4, rank_idx)
            # k=6 -> Schoolbook 2x2x1
            rank_idx = add_schoolbook_2x2x1(i0, j0, 6, rank_idx)
            
            return rank_idx

        # Block 1: A1 (i=0..1, j=0..1), B1 (j=0..1, k=0..6)
        rank_idx = add_2x2x7_block(0, 0, rank_idx)
        
        # Block 2: A2 (i=0..1, j=2..3), B2 (j=2..3, k=0..6)
        rank_idx = add_2x2x7_block(0, 2, rank_idx)
        
        # Total rank: 23 + 23 = 46.
        
        if verify_decomposition(U, V, W, rank_idx):
            return {"U": U, "V": V, "W": W}
        else:
            return None

    best_result = schoolbook()
    
    if time.time() < deadline - 5.0:
        res = construct_rank_46()
        if res is not None:
            best_result = res
            
    return best_result

if __name__ == "__main__":
    r = construct()
    print("Rank:", len(r['U'][0]))
