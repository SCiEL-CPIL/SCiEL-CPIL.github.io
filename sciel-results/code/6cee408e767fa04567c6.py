def construct():
    n, m, p = 3, 3, 6
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                c = (i * m + j) * 3 + k # This is not a bijection for rank 9.
                # Standard schoolbook for 3x3x3 has 9 terms.
                # Term index t corresponds to (i,j,k) ? No.
                # Schoolbook: T = sum_{i,j,k} E_ij^A E_jk^B E_ki^C.
                # The terms are indexed by (j,k) ? No.
                # The decomposition is:
                # U is indexed by (i,j), V by (j,k), W by (k,i).
                # The "columns" are the products.
                # In schoolbook, each term in the sum is a rank-1 tensor.
                # Sum_{i,j,k} (e_i e_j^T) (e_j e_k^T) (e_k e_i^T).
                # The term for specific i,j,k is:
                # u = e_i \otimes e_j (vector in C^{n*m})
                # v = e_j \otimes e_k (vector in C^{m*p})
                # w = e_k \otimes e_i (vector in C^{p*n})
                # There are n*m*p such terms.
                # For 3x3x3, there are 27 terms?
                # NO.
                # Schoolbook rank is n*m*p.
                # For 3x3x3, rank is 27.
                # My previous calculation of 9 was wrong.
                # 3*3*3 = 27.
                
                # So Block 1 rank = 27.
                # Block 2 rank = 27.
                # Total rank = 54.
                
                # This is the SAME as the seed program.
                # The seed program is already using schoolbook for the whole 3x3x6.
                # Rank 54.
                
                # My idea of splitting p into 3+3 doesn't reduce rank if we use schoolbook for each part.
                # We need a decomposition of 3x3x3 with rank < 27.
                # Strassen for 3x3x3 has rank 7.
                # If I use rank 7 for each block, total rank 14.
