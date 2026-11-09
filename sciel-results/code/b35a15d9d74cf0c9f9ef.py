"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_rank = 58

    # Preallocate
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    col = 0

    # Helper: Add a single rank-1 tensor term:
    #   U[i0 + di][col] = a, V[j0 + dj][col] = b, W[k0 + dk][col] = c
    def add_term(i0, j0, k0, a, b, c):
        U[i0][col] += a
        V[j0][col] += b
        W[k0][col] += c
        return col + 1

    # Strassen 2x2x2 decomposition (local i0, j0, k0)
    # Local indices:
    #   i in {0,1} -> global i0 + i
    #   j in {0,1} -> global j0 + j
    #   k in {0,1} -> global k0 + k
    # Strassen coefficients:
    #   M1 = (A11 + A22)(B12 + B22) -> C += M1 e22 + M1 e12 - M1 e11 - M1 e22 ... wait, let me use standard.
    # Standard Strassen for C = AB:
    #   P1 = (A11 + A22)(B11 + B22)
    #   P2 = (A11 + A22)B22
    #   P3 = A11(B11 - B21)
    #   P4 = A22(B22 - B12)
    #   P5 = (A11 + A12)B22
    #   P6 = (A21 - A11)(B11 + B12)
    #   P7 = (A12 - A22)(B21 + B22)
    # Then:
    #   C11 = P1 - P2 + P3 + P6
    #   C12 = P4 + P5
    #   C21 = P3 + P7
    #   C22 = P1 - P5 + P4 + P6
    #
    # Each Pk corresponds to a term. Let's enumerate:
    # P1: A part (1,1),(2,2); B part (1,1),(2,2); C contributions: C11(+1), C22(-1)
    # P2: A part (1,1),(2,2); B part (2,2); C contributions: C11(-1), C22(-1)  [wait: C22 has -P5+P4+P6, C11 has -P2]
    # Let me re-check C22: P1 - P5 + P4 + P6. So P1(+1), P5(-1), P4(+1), P6(+1)
    # C11: P1 - P2 + P3 + P6 -> P1(+1), P2(-1), P3(+1), P6(+1)
    # C12: P4 + P5 -> P4(+1), P5(+1)
    # C21: P3 + P7 -> P3(+1), P7(+1)
    #
    # Let's define each Strassen term by (A-indices, B-indices, C-indices with signs):
    strassen_blocks = [
        # (A_row_indices, A_cols_indices, B_row_indices, B_cols_indices, C_entries)
        # Term 1: P1 = (A11+A22)(B11+B22)
        ([(1,1),(2,1)], [(1,1),(1,2),(2,1),(2,2)],  # wait, A11+A22 means A[0][0]+A[1][1] in 0-indexed
         None, None, [(1,1,+1), (2,2,-1)]),
    ]
    # This is getting complex. Let me just hardcode the 7 terms with explicit index lists.
    # 0-indexed local:
    # A[i][j] with i,j in {0,1}
    # B[j][k] with j,k in {0,1}
    # C[i][k] with i,k in {0,1}
    
    strassen_terms = [
        # P1: (A[0][0]+A[1][1]) * (B[0][0]+B[1][1])
        # A factors: A[0][0]=1, A[1][1]=1; B factors: B[0][0]=1, B[1][1]=1
        # C: C[0][0]+=1, C[1][1]-=1
        {'A': [(0,0,1),(1,1,1)], 'B': [(0,0,1),(1,1,1)], 'C': [(0,0,1),(1,1,-1)]},
        # P2: (A[0][0]+A[1][1]) * B[1][1]
        # A: A[0][0]=1, A[1][1]=1; B: B[1][1]=1
        # C: C[0][0]-=1, C[1][1]+=1  [from C11:-P2, C22:+P4... wait C22 has +P4, not +P2. Let me recheck]
        # C11 = P1 - P2 + P3 + P6, C22 = P1 - P5 + P4 + P6
        # So P2 appears only in C11 with sign -1.
        {'A': [(0,0,1),(1,1,1)], 'B': [(1,1,1)], 'C': [(0,0,-1)]},
        # P3: A[0][0] * (B[0][0]-B[0][1])
        # A: A[0][0]=1; B: B[0][0]=1, B[0][1]=-1
        # C: C[0][0]+=1, C[1][0]+=1
        {'A': [(0,0,1)], 'B': [(0,0,1),(0,1,-1)], 'C': [(0,0,1),(1,0,1)]},
        # P4: A[1][1] * (B[1][1]-B[1][0])  wait, P4 = A22(B22-B12) in 1-indexed = A[1][1]*(B[1][1]-B[0][1])? 
        # In 1-indexed: P4 = A22(B22 - B12). So B22=B[1][1], B12=B[0][1].
        # A: A[1][1]=1; B: B[1][1]=1, B[0][1]=-1
        # C: C[0][1]+=1, C[1][1]+=1
        {'A': [(1,1,1)], 'B': [(1,1,1),(0,1,-1)], 'C': [(0,1,1),(1,1,1)]},
        # P5: (A[0][0]+A[0][1]) * B[1][1]
        # A: A[0][0]=1, A[0][1]=1; B: B[1][1]=1
        # C: C[0][1]+=1, C[1][1]-=1
        {'A': [(0,0,1),(0,1,1)], 'B': [(1,1,1)], 'C': [(0,1,1),(1,1,-1)]},
        # P6: (A[1][0]-A[0][0]) * (B[0][0]+B[0][1])
        # A: A[1][0]=1, A[0][0]=-1; B: B[0][0]=1, B[0][1]=1
        # C: C[0][0]+=1, C[1][1]+=1
        {'A': [(1,0,1),(0,0,-1)], 'B': [(0,0,1),(0,1,1)], 'C': [(0,0,1),(1,1,1)]},
        # P7: (A[0][1]-A[1][1]) * (B[1][0]+B[1][1])
        # A: A[0][1]=1, A[1][1]=-1; B: B[1][0]=1, B[1][1]=1
        # C: C[1][0]+=1
        {'A': [(0,1,1),(1,1,-1)], 'B': [(1,0,1),(1,1,1)], 'C': [(1,0,1)]},
    ]

    def add_strassen_block(i0, j0, k0):
        global col
        for term in strassen_terms:
            # This term is a sum of products of sums. We need to express it as a sum of rank-1 terms.
            # Actually, each "term" in Strassen is itself a product of two linear forms, which is a sum of rank-1 tensors.
            # But in the tensor decomposition, each Pk corresponds to ONE rank-1 term in the decomposition? 
            # No! In Strassen, P1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
            # That's 4 rank-1 terms. But we only count P1 as ONE "multiplication", so ONE rank-1 tensor in the decomposition.
            # Wait, no. In the tensor decomposition of matrix multiplication, each rank-1 term is of the form u v^T w^*.
            # A product like (A11+A22)(B11+B22) expands to 4 products, so it corresponds to 4 rank-1 tensors, NOT 1.
            # I made a fundamental error. Strassen uses 7 multiplications, but each multiplication Pk is a product of two linear combinations, 
            # which when expanded gives multiple bilinear forms. The rank of the 2x2x2 multiplication tensor is 7, not 49.
            # The 7 rank-1 terms in the rank-7 decomposition are NOT the same as the 7 Strassen products.
            # Strassen's algorithm shows that you can compute the 4 products using 7 bilinear forms, but each bilinear form is NOT necessarily rank-1.
            # A bilinear form like (a+b)(c+d) = ac+ad+bc+bd is a sum of 4 rank-1 forms. So P1 is rank 4, not rank 1.
            # Therefore, the Strassen algorithm does NOT give a rank-7 decomposition directly. It gives an algorithm with 7 multiplications,
            # but the tensor rank of 2x2x2 is actually 7, and there IS a rank-7 decomposition, but it's not directly Strassen's 7 products.
            # I was confused. Let me use the known rank-7 decomposition for 2x2x2.
            # Actually, for the purpose of this task, I need to provide a valid decomposition. Let me just use the schoolbook for 2x2x2 (rank 8) 
            # and accept a higher total rank, OR find a known rank-7 decomposition.
            # For simplicity and correctness, let me use schoolbook for all sub-blocks.
            pass
        return col

    # Let me restart with schoolbook for all sub-blocks. Total rank = 2*5*6 = 60.
    # This is the same as the seed. I need a better approach.
    # Let me try a different strategy: use the fact that for small matrices, we can sometimes find better decompositions.
    # For now, let me just implement the schoolbook correctly and accept rank 60.
    
    # Reset
    U = [[0] * (n*m*p) for _ in range(n * m)]
    V = [[0] * (n*m*p) for _ in range(m * p)]
    W = [[0] * (n*m*p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
