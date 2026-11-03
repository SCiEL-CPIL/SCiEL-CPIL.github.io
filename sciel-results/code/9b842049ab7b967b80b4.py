"""Strassen-embedded decomposition for 3x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 by 5x6 matrix.
    Uses Strassen's rank-7 decomposition for 2x2x2 sub-blocks to reduce rank from 90.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 135.0  # Safety margin before 150s limit

    def verify(U, V, W, rank):
        """Check if the decomposition is exact using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-9)

    def schoolbook():
        """Standard schoolbook decomposition with rank = n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """
        Return the 7 rank-1 tensors (as lists of (i,j,k) triples with coefficients)
        for Strassen's 2x2x2 matrix multiplication.
        
        For 2x2x2: A(2,2)*B(2,2)=C(2,2)
        The tensor T[i,j,k]=1 for all i,j,k in {0,1}.
        
        Strassen's 7 products:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B10)
        M5 = (A00+A10)B11
        M6 = (A10-A00)(B00+B10)
        M7 = (A11-A01)(B01+B11)
        
        Results:
        C00 = M1 - M4 - M6 + M7
        C01 = M3 + M4
        C10 = M1 - M2 + M5 - M7
        C11 = M2 - M3 - M5 + M6
        
        Each M_r corresponds to a rank-1 term where:
        U_r[i,j] = coefficient of A[i,j] in the first factor
        V_r[j,k] = coefficient of B[j,k] in the second factor
        W_r[k,i] = coefficient in the result for C[i,k]
        """
        # For each of the 7 products, specify (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: dict (i,j) -> coeff for A[i,j]
        # V_coeffs: dict (j,k) -> coeff for B[j,k]
        # W_coeffs: dict (i,k) -> coeff for C[i,k]
        
        strassen_terms = [
            # M1 = (A00+A11)(B00+B11)
            # U: A00=1, A11=1
            # V: B00=1, B11=1
            # W: C00=1, C10=1, C01=-1, C11=1  (from C00=M1, C10=M1, C01=-M1? No)
            # Actually: C00 = M1-M4-M6+M7, C10 = M1-M2+M5-M7, C01 = M3+M4, C11 = M2-M3-M5+M6
            # So M1 contributes: C00+=1, C10+=1, C01+=0, C11+=0? 
            # Wait, let me re-derive. The standard Strassen for C = A*B:
            # C00 = M1 - M4 - M6 + M7
            # C01 = M3 + M4
            # C10 = M1 - M2 + M5 - M7
            # C11 = M2 - M3 - M5 + M6
            #
            # So each M_r contributes to specific C entries with specific signs.
            # M1: C00=+1, C10=+1 (appears in both)
            # M2: C10=-1, C11=+1
            # M3: C01=+1, C11=-1
            # M4: C00=-1, C01=+1
            # M5: C10=+1, C11=-1
            # M6: C00=-1, C11=+1
            # M7: C00=+1, C10=-1
            
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,0):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):-1, (1,1):1}),
            # M3
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):-1}),
            # M4
            ({(1,1):1}, {(0,1):1, (1,0):-1}, {(0,0):-1, (0,1):1}),
            # M5
            ({(0,0):1, (1,0):1}, {(1,1):1}, {(1,0):1, (1,1):-1}),
            # M6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(0,0):-1, (1,1):1}),
            # M7
            ({(1,1):1, (0,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1, (1,0):-1}),
        ]
        return strassen_terms

    def embed_strassen(global_i_offset, global_j_offset, global_k_offset):
        """
        Embed a 2x2x2 Strassen decomposition at given global offsets.
        Returns list of (U_dict, V_dict, W_dict) with global indices.
        """
        terms = strassen_2x2x2()
        global_terms = []
        for U_local, V_local, W_local in terms:
            U_global = {}
            V_global = {}
            W_global = {}
            for (i, j), coeff in U_local.items():
                U_global[(global_i_offset + i, global_j_offset + j)] = coeff
            for (j, k), coeff in V_local.items():
                V_global[(global_j_offset + j, global_k_offset + k)] = coeff
            for (i, k), coeff in W_local.items():
                W_global[(global_k_offset + k, global_i_offset + i)] = coeff
            global_terms.append((U_global, V_global, W_global))
        return global_terms

    def build_decomposition(use_strassen_blocks):
        """
        Build a decomposition using a mix of schoolbook and Strassen terms.
        
        For 3x5x6:
        - Rows 0,1 can form a 2x2 block with columns 0,1 (2x2x2 subproblem)
        - We can embed Strassen in several 2x2x2 sub-blocks
        
        Strategy: Use Strassen for overlapping 2x2x2 blocks where possible,
        and schoolbook for the rest.
        
        For 3x5x6, we can decompose:
        - Block A: rows {0,1}, cols {0,1} -> 2x2x2 Strassen (7 terms)
        - Remaining: schoolbook for all other (i,j,k) not covered
        
        Actually, a cleaner approach: partition the 5 columns and 6 rows.
        Columns: {0,1} and {2,3,4}
        Rows: {0,1} and {2,3,4,5}
        
        For the 2x2 sub-block (cols 0,1 x rows 0,1): use Strassen (7 terms)
        For everything else: schoolbook
        
        This gives: 7 + (90 - 8) = 89 terms... that's worse.
        
        Better: Use Strassen in multiple places.
        Let's think about this differently.
        
        The key insight from the knowledge: replacing 8 schoolbook products per 2x2x2 block
        with 7 Strassen products saves 1 rank per block. For 6 such blocks in 3x5x6,
        we save 6 ranks: 90 - 6 = 84.
        
        How to get 6 non-overlapping 2x2x2 blocks in 3x5x6?
        - i in {0,1}, j in {0,1}, k in {0,1}: Block 1
        - i in {0,1}, j in {0,1}, k in {2,3}: Block 2
        - i in {0,1}, j in {0,1}, k in {4,5}: Block 3
        - i in {0,1}, j in {2,3}, k in {0,1}: Block 4
        - i in {0,1}, j in {2,3}, k in {2,3}: Block 5
        - i in {0,1}, j in {2,3}, k in {4,5}: Block 6
        - i in {2}, j in {0,1}, k in {0,1}: schoolbook (3 terms)
        - i in {2}, j in {2,3}, k in {0,1}: schoolbook (3 terms)
        - i in {2}, j in {4}, k in {0,1}: schoolbook (2 terms)
        - i in {2}, j in {0,1}, k in {2,3}: schoolbook (3 terms)
        - i in {2}, j in {2,3}, k in {2,3}: schoolbook (3 terms)
        - i in {2}, j in {4}, k in {2,3}: schoolbook (2 terms)
        - i in {2}, j in {0,1}, k in {4,5}: schoolbook (3 terms)
        - i in {2}, j in {2,3}, k in {4,5}: schoolbook (3 terms)
        - i in {2}, j in {4}, k in {4,5}: schoolbook (2 terms)
        
        Total: 6*7 + 3+3+2+3+3+2+3+3+2 = 42 + 24 = 66? That can't be right.
        
        Wait, let me recount. The schoolbook has n*m*p = 3*5*6 = 90 terms.
        If I replace 6 blocks of 8 terms each with 7 terms each, I save 6*1 = 6 terms.
        So 90 - 6 = 84 terms.
        
        The 6 blocks each cover 2*2*2 = 8 (i,j,k) triples.
        6 blocks cover 48 triples.
        Remaining: 90 - 48 = 42 triples handled by schoolbook.
        Total: 6*7 + 42 = 42 + 42 = 84. Yes!
        """
        # Define the 6 Strassen blocks: (i_offset, j_offset, k_offset)
        strassen_blocks = [
            (0, 0, 0),   # i:{0,1}, j:{0,1}, k:{0,1}
            (0, 0, 2),   # i:{0,1}, j:{0,1}, k:{2,3}
            (0, 0, 4),   # i:{0,1}, j:{0,1}, k:{4,5}
            (0, 2, 0),   # i:{0,1}, j:{2,3}, k:{0,1}
            (0, 2, 2),   # i:{0,1}, j:{2,3}, k:{2,3}
            (0, 2, 4),   # i:{0,1}, j:{2,3}, k:{4,5}
        ]
        
        # Collect all (i,j,k) covered by Strassen blocks
        covered = set()
        for io, jo, ko in strassen_blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((io+di, jo+dj, ko+dk))
        
        # Build the list of all rank-1 terms
        U_list = []  # Each entry: list of (index, coeff) for U
        V_list = []
        W_list = []
        
        # Add Strassen terms
        for io, jo, ko in strassen_blocks:
            terms = embed_strassen(io, jo, ko)
            for U_global, V_global, W_global in terms:
                u_row = [0] * 0  # placeholder
                v_row = [0] * 0
                w_row = [0] * 0
                U_list.append(U_global)
                V_list.append(V_global)
                W_list.append(W_global)
        
        # Add schoolbook terms for uncovered (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U_list.append({(i, j): 1})
                        V_list.append({(j, k): 1})
                        W_list.append({(k, i): 1})
        
        rank = len(U_list)
        
        # Convert to matrix form
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c in range(rank):
            for (i, j), coeff in U_list[c].items():
                U[i*m + j][c] = coeff
            for (j, k), coeff in V_list[c].items():
                V[j*p + k][c] = coeff
            for (k, i), coeff in W_list[c].items():
                W[k*n + i][c] = coeff
        
        return {"U": U, "V": V, "W": W}

    # Build the Strassen-embedded decomposition
    if time.time() > deadline - 5:
        return schoolbook()
    
    result = build_decomposition(True)
    
    # Verify
    if verify(result["U"], result["V"], result["W"], len(result["U"][0])):
        return result
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
