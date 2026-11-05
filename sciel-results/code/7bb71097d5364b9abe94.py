"""Matrix multiplication 3,3,6 rank-minimal exact decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if not U or not V or not W:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def build_strassen_3x3x3_block(k_offset):
        """
        Build Strassen-based decomposition for one 3x3x3 block.
        Returns list of (u_dict, v_dict, w_dict) where each dict maps
        (row, col) -> coefficient for that factor in that term.
        
        The 3x3 matrices are partitioned as:
        A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
        B = [[B11, B12], [B21, B22]] similarly
        C = A*B, and we decompose each C[i,j,k] entry.
        
        Actually, for matrix multiplication C = A*B where A is 3x3, B is 3x3, C is 3x3:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        The tensor entry T[i,j,k] = 1 for all i,j,k (each product A[i,j]*B[j,k] appears once).
        
        We partition indices into {0,1} and {2}:
        - For the 2x2x2 sub-block (i,j,k in {0,1}), use Strassen (7 terms)
        - For terms involving index 2, use schoolbook
        """
        terms = []
        
        # Strassen for 2x2x2 sub-block (i,j,k all in {0,1})
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B11)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # In our tensor, T[i,j,k] = 1 for the product A[i,j]*B[j,k]
        # So for the 2x2x2 block, we need to decompose the 8 entries
        # using Strassen's 7 intermediate products.
        
        # Let me think about this differently. The tensor T[i,j,k] for i,j,k in {0,1,2}
        # has T[i,j,k] = 1. We want to write:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # (Note: W index is [k*n+i] which is k*3+i for n=3)
        
        # Actually the W factor has index [k*n + i] = [3k + i], so W is indexed by (k, i).
        # The tensor equation is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        # For a 3x3x3 block with k in {k_offset, k_offset+1, k_offset+2},
        # let's use local indices i, j, kk in {0,1,2} where kk = k - k_offset.
        
        # Strassen terms for the 2x2x2 core (i,j,kk in {0,1}):
        # Each Strassen term corresponds to a bilinear form.
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # This contributes to C[0,0], C[0,1], C[1,0], C[1,1]
        
        # In tensor form, for the 2x2x2 block:
        # T[i,j,kk] = 1 means the product A[i,j]*B[j,kk] has coefficient 1.
        
        # Strassen decomposition of the 2x2x2 multiplication tensor:
        # The 2x2x2 tensor has 8 entries, all = 1.
        # Strassen gives rank 7 decomposition.
        
        # Let me use the standard Strassen formulas and map to tensor factors.
        # For C = A*B (2x2), C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor T[i,j,k] = 1 for i,j,k in {0,1}.
        
        # Strassen:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b21-b11)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        # In tensor form, T[i,j,k] = 1 for all i,j,k in {0,1}.
        # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
        
        # For each Strassen term mc, we define U, V, W factors.
        # The key insight: each Strassen term mc is a product of two bilinear forms.
        # In the tensor decomposition, each term c has:
        # U[i,j,c] = coefficient of A[i,j] in the first bilinear form
        # V[j,k,c] = coefficient of B[j,k] in the second bilinear form  
        # W[k,i,c] = coefficient that maps to C[i,k]
        
        # Actually, let me reconsider. The standard tensor decomposition for matrix mult:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # where T[i,j,k] = 1 (the "matrix multiplication tensor").
        
        # For Strassen on 2x2x2:
        # Term 1: m1 = (a11+a22)(b11+b22)
        #   U[0,0,1]=1, U[1,1,1]=1
        #   V[0,0,1]=1, V[1,1,1]=1
        #   W[0,0,1]=1, W[0,1,1]=1, W[1,0,1]=1, W[1,1,1]=1  (contributes to all C[i,k])
        
        # Wait, that's not right. Let me be more careful.
        
        # The tensor equation is:
        # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = T[i,j,k] = 1
        
        # For Strassen term m1 = (a11+a22)(b11+b22):
        # This term appears in c11, c22 with coefficient +1, and in c11 with +1, c22 with +1.
        # Actually: c11 = m1 + m4 - m5 + m7, c22 = m1 - m2 + m3 + m6
        # So m1 appears in c11 (coeff +1) and c22 (coeff +1).
        
        # In the tensor, T[i,j,k] = 1 for the product a_ij * b_jk.
        # The Strassen decomposition rewrites the 8 products using 7 intermediates.
        
        # Let me just directly construct the factors.
        # For each Strassen term, define U, V, W such that the product
        # U[i,j,c] * V[j,k,c] * W[k,i,c] gives the right contribution.
        
        # Term 1: m1 = (a11+a22)(b11+b22)
        # U: a11 and a22 have coeff 1
        # V: b11 and b22 have coeff 1  (b_jk where j,k in {0,1}, so b00 and b11)
        # W: contributes to c00 and c11 (i=0,k=0 and i=1,k=1)
        # W[0,0,1]=1, W[1,1,1]=1
        terms.append({
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}
        })
        
        # Term 2: m2 = (a21+a22)(b11)
        # U: a11 and a12 have coeff 1  (a21 means a[1,0], a22 means a[1,1] in 0-indexed)
        # Wait, I'm getting confused with indexing. Let me use 0-indexed.
        # a11 = A[0,0], a12 = A[0,1], a21 = A[1,0], a22 = A[1,1]
        # b11 = B[0,0], b12 = B[0,1], b21 = B[1,0], b22 = B[1,1]
        
        # m2 = (a21+a22)*b11 = (A[1,0]+A[1,1])*B[0,0]
        # U: A[1,0] coeff 1, A[1,1] coeff 1
        # V: B[0,0] coeff 1
        # W: c21 = m2 + m4, so m2 contributes to c[1,0] (i=1,k=0)
        # W[0,1,2]=1  (k=0, i=1)
        terms.append({
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(0,1): 1}
        })
        
        # Term 3: m3 = a11*(b21-b11) = A[0,0]*(B[1,0]-B[0,0])
        # U: A[0,0] coeff 1
        # V: B[1,0] coeff 1, B[0,0] coeff -1
        # W: c12 = m3 + m5, c22 = m1-m2+m3+m6, so m3 contributes to c[0,1] and c[1,1]
        # W[1,0,3]=1, W[1,1,3]=1  (k=1,i=0 and k=1,i=1)
        terms.append({
            'U': {(0,0): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(1,0): 1, (1,1): 1}
        })
        
        # Term 4: m4 = a22*(b21-b11) = A[1,1]*(B[1,0]-B[0,0])
        # U: A[1,1] coeff 1
        # V: B[1,0] coeff 1, B[0,0] coeff -1
        # W: c11 = m1+m4-m5+m7, c21 = m2+m4, so m4 contributes to c[0,0] and c[1,0]
        # W[0,0,4]=1, W[0,1,4]=1
        terms.append({
            'U': {(1,1): 1},
            'V': {(1,0): 1, (0,0): -1},
            'W': {(0,0): 1, (0,1): 1}
        })
        
        # Term 5: m5 = (a11+a12)*b22 = (A[0,0]+A[0,1])*B[1,1]
        # U: A[0,0] coeff 1, A[0,1] coeff 1
        # V: B[1,1] coeff 1
        # W: c11 = ...-m5..., c12 = m3+m5, so m5 contributes to c[0,0] with -1 and c[0,1] with +1
        # W[0,0,5]=-1, W[1,0,5]=1
        terms.append({
            'U': {(0,0): 1, (0,1): 1},
            'V': {(1,1): 1},
            'W': {(0,0): -1, (1,0): 1}
        })
        
        # Term 6: m6 = (a21-a11)*(b11+b12) = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # U: A[1,0] coeff 1, A[0,0] coeff -1
        # V: B[0,0] coeff 1, B[0,1] coeff 1
        # W: c22 = ...+m6, so m6 contributes to c[1,1]
        # W[1,1,6]=1
        terms.append({
            'U': {(1,0): 1, (0,0): -1},
            'V': {(0,0): 1, (0,1): 1},
            'W': {(1,1): 1}
        })
        
        # Term 7: m7 = (a12-a22)*(b21+b22) = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # U: A[0,1] coeff 1, A[1,1] coeff -1
        # V: B[1,0] coeff 1, B[1,1] coeff 1
        # W: c11 = ...+m7, so m7 contributes to c[0,0]
        # W[0,0,7]=1
        terms.append({
            'U': {(0,1): 1, (1,1): -1},
            'V': {(1,0): 1, (1,1): 1},
            'W': {(0,0): 1}
        })
        
        # Now handle terms involving index 2 (i=2, j=2, or kk=2)
        # Schoolbook for these:
        # For each (i,j,kk) where at least one is 2, T[i,j,kk] = 1
        # We need one term per such (i,j,kk) triple.
        
        # i=2, j=0, kk=0: A[2,0]*B[0,0] -> C[2,0]
        terms.append({'U': {(2,0): 1}, 'V': {(0,0): 1}, 'W': {(0,2): 1}})
        # i=2, j=0, kk=1: A[2,0]*B[0,1] -> C[2,1]
        terms.append({'U': {(2,0): 1}, 'V': {(0,1): 1}, 'W': {(1,2): 1}})
        # i=2, j=1, kk=0: A[2,1]*B[1,0] -> C[2,0]
        terms.append({'U': {(2,1): 1}, 'V': {(1,0): 1}, 'W': {(0,2): 1}})
        # i=2, j=1, kk=1: A[2,1]*B[1,1] -> C[2,1]
        terms.append({'U': {(2,1): 1}, 'V': {(1,1): 1}, 'W': {(1,2): 1}})
        # i=2, j=2, kk=0: A[2,2]*B[2,0] -> C[2,0]
        terms.append({'U': {(2,2): 1}, 'V': {(2,0): 1}, 'W': {(0,2): 1}})
        # i=2, j=2, kk=1: A[2,2]*B[2,1] -> C[2,1]
        terms.append({'U': {(2,2): 1}, 'V': {(2,1): 1}, 'W': {(1,2): 1}})
        # i=0, j=2, kk=0: A[0,2]*B[2,0] -> C[0,0]
        terms.append({'U': {(0,2): 1}, 'V': {(2,0): 1}, 'W': {(0,0): 1}})
        # i=0, j=2, kk=1: A[0,2]*B[2,1] -> C[0,1]
        terms.append({'U': {(0,2): 1}, 'V': {(2,1): 1}, 'W': {(1,0): 1}})
        # i=1, j=2, kk=0: A[1,2]*B[2,0] -> C[1,0]
        terms.append({'U': {(1,2): 1}, 'V': {(2,0): 1}, 'W': {(0,1): 1}})
        # i=1, j=2, kk=1: A[1,2]*B[2,1] -> C[1,1]
        terms.append({'U': {(1,2): 1}, 'V': {(2,1): 1}, 'W': {(1,1): 1}})
        # i=0, j=0, kk=2: A[0,0]*B[0,2] -> C[0,2]
        terms.append({'U': {(0,0): 1}, 'V': {(0,2): 1}, 'W': {(2,0): 1}})
        # i=0, j=1, kk=2: A[0,1]*B[1,2] -> C[0,2]
        terms.append({'U': {(0,1): 1}, 'V': {(1,2): 1}, 'W': {(2,0): 1}})
        # i=1, j=0, kk=2: A[1,0]*B[0,2] -> C[1,2]
        terms.append({'U': {(1,0): 1}, 'V': {(0,2): 1}, 'W': {(2,1): 1}})
        # i=1, j=1, kk=2: A[1,1]*B[1,2] -> C[1,2]
        terms.append({'U': {(1,1): 1}, 'V': {(1,2): 1}, 'W': {(2,1): 1}})
        # i=2, j=0, kk=2: A[2,0]*B[0,2] -> C[2,2]
        terms.append({'U': {(2,0): 1}, 'V': {(0,2): 1}, 'W': {(2,2): 1}})
        # i=2, j=1, kk=2: A[2,1]*B[1,2] -> C[2,2]
        terms.append({'U': {(2,1): 1}, 'V': {(1,2): 1}, 'W': {(2,2): 1}})
        # i=2, j=2, kk=2: A[2,2]*B[2,2] -> C[2,2]
        terms.append({'U': {(2,2): 1}, 'V': {(2,2): 1}, 'W': {(2,2): 1}})
        
        return terms

    # Build the full decomposition for 3x3x6
    # Two blocks: k in {0,1,2} and k in {3,4,5}
    all_terms = []
    for block in range(2):
        k_offset = block * 3
        block_terms = build_strassen_3x3x3_block(k_offset)
        for term in block_terms:
            # Map local indices to global
            u_global = {}
            for (i, j), val in term['U'].items():
                u_global[(i, j)] = val
            
            v_global = {}
            for (j, kk), val in term['V'].items():
                v_global[(j, kk + k_offset)] = val
            
            w_global = {}
            for (kk, i), val in term['W'].items():
                w_global[(kk + k_offset, i)] = val
            
            all_terms.append({
                'U': u_global,
                'V': v_global,
                'W': w_global
            })
    
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(all_terms):
        for (i, j), val in term['U'].items():
            U[i * m + j][c] = val
        for (j, k), val in term['V'].items():
            V[j * p + k][c] = val
        for (k, i), val in term['W'].items():
            W[k * n + i][c] = val
    
    # Verify the decomposition
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if Strassen construction fails
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
