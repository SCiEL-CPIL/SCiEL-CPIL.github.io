import time
import numpy as np

DIMENSIONS = [3, 5, 6]
DEADLINE = 140.0

def _schoolbook_block(n, m, p, offset_u, offset_v, offset_w, rank_offset):
    """
    Generates factors for a schoolbook multiplication of n x m by m x p.
    Appends to U, V, W at the specified offsets.
    """
    U = []
    V = []
    W = []
    rank = n * m * p
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = 1.0
                v_val = 1.0
                w_val = 1.0
                # Indices in the global arrays
                # U index: (i, j) -> i * m_global + j? No, U is (n*m, rank).
                # The block U is at row i + offset_u, col rank_offset
                # We need to return the local factors to be placed in the global arrays.
                # It's easier to build the global arrays directly in the main function.
                pass
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Total rank estimate
    # Part 1: 3x2x6 -> 3 blocks of (7+4) = 33
    # Part 2: 3x3x6 -> 2 blocks of 27 = 54
    # Total 87
    total_rank = 87
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    rank_idx = 0
    
    # --- Part 1: j in {0, 1} ---
    # A is 3x2 (rows 0-2, cols 0-1)
    # B is 2x6 (rows 0-1, cols 0-5)
    # Split B into 3 blocks of 2 cols: {0,1}, {2,3}, {4,5}
    
    for k_block in range(3):
        k_start = k_block * 2
        k_end = k_start + 2
        
        # --- Sub-part 1a: i in {0, 1} (2x2 Strassen) ---
        # A_sub is 2x2 (rows 0-1, cols 0-1)
        # B_sub is 2x2 (rows 0-1, cols k_start to k_end)
        
        # Strassen 2x2x2 coefficients
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        
        # We need to map these to U, V, W.
        # U entries for A_sub:
        # a = U[0, r], b = U[1, r] (row 0 of A_sub)
        # c = U[2, r], d = U[3, r] (row 1 of A_sub)
        # Note: Global U index is i*m + j.
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 2
        # i=1, j=1 -> 3
        
        # V entries for B_sub:
        # e = V[0, r], f = V[1, r] (col 0 of B_sub, which is global col k_start)
        # g = V[2, r], h = V[3, r] (col 1 of B_sub, which is global col k_start+1)
        # Global V index is j*p + k.
        # j=0, k=k_start -> 0*6 + k_start
        # j=0, k=k_start+1 -> 0*6 + k_start+1
        # j=1, k=k_start -> 1*6 + k_start
        # j=1, k=k_start+1 -> 1*6 + k_start+1
        
        # W entries for C_sub (2x2):
        # C[0,0] = W[k_start*n + 0, r]
        # C[0,1] = W[k_start*n + 1, r]
        # C[1,0] = W[(k_start+1)*n + 0, r]
        # C[1,1] = W[(k_start+1)*n + 1, r]
        
        # Strassen factors (U, V, W) for the 7 terms
        # Each term is a rank-1 tensor.
        
        # Term 1: (a+d)(e+h)
        # U: a=1, d=1
        # V: e=1, h=1
        # W: C11=1, C12=1, C21=1, C22=1? No.
        # P1 contributes to all 4 outputs with specific signs.
        
        # Let's define the 7 Strassen terms explicitly.
        # a,b,c,d are A[0,0], A[0,1], A[1,0], A[1,1]
        # e,f,g,h are B[0,0], B[0,1], B[1,0], B[1,1]
        
        # P1: (a+d)(e+h)
        # U: A[0,0]=1, A[1,1]=1
        # V: B[0,0]=1, B[1,1]=1
        # W: C[0,0]=1, C[0,1]=1, C[1,0]=1, C[1,1]=1
        
        # P2: (c+d)e
        # U: A[1,0]=1, A[1,1]=1
        # V: B[0,0]=1
        # W: C[1,0]=1, C[1,1]=-1
        
        # P3: a(f-h)
        # U: A[0,0]=1
        # V: B[0,1]=1, B[1,1]=-1
        # W: C[0,1]=1, C[1,1]=1
        
        # P4: d(g-e)
        # U: A[1,1]=1
        # V: B[1,0]=1, B[0,0]=-1
        # W: C[0,0]=1, C[1,0]=1
        
        # P5: (a+b)h
        # U: A[0,0]=1, A[0,1]=1
        # V: B[1,1]=1
        # W: C[0,0]=-1, C[0,1]=1
        
        # P6: (c-a)(e+f)
        # U: A[1,0]=1, A[0,0]=-1
        # V: B[0,0]=1, B[0,1]=1
        # W: C[1,1]=1
        
        # P7: (b-d)(g+h)
        # U: A[0,1]=1, A[1,1]=-1
        # V: B[1,0]=1, B[1,1]=1
        # W: C[0,0]=1, C[1,1]=-1
        
        strassen_terms = [
            # (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
            # Indices are relative to the 2x2 block
            # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 1: 1, 2: 1, 3: 1}), # P1
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # P2
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # P3
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # P4
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # P5
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # P6
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: -1}), # P7
        ]
        
        for u_map, v_map, w_map in strassen_terms:
            # U
            for idx, val in u_map.items():
                i, j = idx // 2, idx % 2
                U[i * m + j][rank_idx] += val
            # V
            for idx, val in v_map.items():
                j, k = idx // 2, idx % 2
                V[j * p + (k_start + k)][rank_idx] += val
            # W
            for idx, val in w_map.items():
                k, i = idx // 2, idx % 2
                W[(k_start + k) * n + i][rank_idx] += val
            rank_idx += 1
            
        # --- Sub-part 1b: i in {2} (1x2 Schoolbook) ---
        # A_sub is 1x2 (row 2, cols 0-1)
        # B_sub is 2x2 (rows 0-1, cols k_start to k_end)
        # Rank 4: 1 * 2 * 2
        
        # Terms: A[2,0]*B[0,0], A[2,0]*B[0,1], A[2,1]*B[1,0], A[2,1]*B[1,1]
        # Wait, Schoolbook for 1x2 x 2x2:
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
        
        # Term 1: A[2,0] B[0,0] -> C[0,0]
        U[2 * m + 0][rank_idx] = 1.0
        V[0 * p + k_start][rank_idx] = 1.0
        W[k_start * n + 2][rank_idx] = 1.0
        rank_idx += 1
        
        # Term 2: A[2,0] B[0,1] -> C[0,1]
        U[2 * m + 0][rank_idx] = 1.0
        V[0 * p + (k_start + 1)][rank_idx] = 1.0
        W[(k_start + 1) * n + 2][rank_idx] = 1.0
        rank_idx += 1
        
        # Term 3: A[2,1] B[1,0] -> C[1,0]
        U[2 * m + 1][rank_idx] = 1.0
        V[1 * p + k_start][rank_idx] = 1.0
        W[k_start * n + 2][rank_idx] = 1.0
        rank_idx += 1
        
        # Term 4: A[2,1] B[1,1] -> C[1,1]
        U[2 * m + 1][rank_idx] = 1.0
        V[1 * p + (k_start + 1)][rank_idx] = 1.0
        W[(k_start + 1) * n + 2][rank_idx] = 1.0
        rank_idx += 1

    # --- Part 2: j in {2, 3, 4} ---
    # A is 3x3 (rows 0-2, cols 2-4)
    # B is 3x6 (rows 2-4, cols 0-5)
    # Split B into 2 blocks of 3 cols: {0,1,2}, {3,4,5}
    
    for k_block in range(2):
        k_start = k_block * 3
        k_end = k_start + 3
        
        # Schoolbook 3x3x3
        # 27 terms
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * m + (2 + j)][rank_idx] = 1.0
                    V[(2 + j) * p + (k_start + k)][rank_idx] = 1.0
                    W[(k_start + k) * n + i][rank_idx] = 1.0
                    rank_idx += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
