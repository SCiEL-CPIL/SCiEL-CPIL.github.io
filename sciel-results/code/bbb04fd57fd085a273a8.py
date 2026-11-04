"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill

def _make_laderman_3x3():
    """
    Laderman's algorithm for 3x3 matrix multiplication, rank 23.
    Returns (U, V, W) as 2D arrays of shape (9, 23).
    
    Convention: C[i,k] = sum_j A[i,j] B[j,k]
    U indexed by (i,j) -> i*3+j
    V indexed by (j,k) -> j*3+k
    W indexed by (k,i) -> k*3+i
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Laderman's 23 products from the original paper.
    # A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    #
    # The 23 multiplications:
    # p0  = a*j
    # p1  = b*m
    # p2  = c*p
    # p3  = d*k
    # p4  = e*n
    # p5  = f*q
    # p6  = g*l
    # p7  = h*r
    # p8  = i*o
    # p9  = (a+e)*(j+n)
    # p10 = (a+c)*(j+p)
    # p11 = (e+i)*(n+o)
    # p12 = (a+b+d+e)*(j+k+m+n)
    # p13 = (a+c+g+i)*(j+l+p+r)
    # p14 = (e+f+h+i)*(n+o+q+r)
    # p15 = (a+b+c+e+f+g+i)*(j+k+l+m+n+o+p+r)
    # p16 = (a-b+d-e)*(j-k+m-n)
    # p17 = (a-c+g-i)*(j-l+p-r)
    # p18 = (e-f+h-i)*(n-o+q-r)
    # p19 = (b+e-c-f)*(m+n-p-q)
    # p20 = (d+e-g-h)*(j+k-l-r)
    # p21 = (b+c-e-f)*(p+q-m-n)
    # p22 = (d+g-e-h)*(l+r-j-k)
    
    # Let me verify this is correct by checking the reconstruction.
    # C[i,k] should equal sum_j A[i,j]*B[j,k]
    
    # C[0,0] = a*j + b*m + c*p
    #   p0: a*j -> contributes 1 to C[0,0]
    #   p1: b*m -> contributes 1 to C[0,0]
    #   p2: c*p -> contributes 1 to C[0,0]
    #   p9: (a+e)(j+n) -> a*j + a*n + e*j + e*n -> contributes 1 to C[0,0] (from a*j)
    #   p12: (a+b+d+e)(j+k+m+n) -> a*j + ... -> contributes 1 to C[0,0]
    #   p15: ... -> a*j + ... -> contributes 1 to C[0,0]
    #   p16: (a-b+d-e)(j-k+m-n) -> a*j - a*k + ... -> contributes 1 to C[0,0]
    #   
    # This is getting complicated. Let me use a different, verified approach.
    # I'll use the well-known Laderman coefficients from a reliable source.
    
    # Actually, let me use a different strategy. I'll use the fact that
    # Laderman's algorithm can be described as follows:
    
    # The 23 terms are defined by their contributions to U, V, W.
    # Each term c has:
    #   U_c: 3x3 matrix (flattened to 9-vector)
    #   V_c: 3x3 matrix (flattened to 9-vector)  
    #   W_c: 3x3 matrix (flattened to 9-vector)
    # such that sum_c U_c ⊗ V_c ⊗ W_c = T where T[i,j,k] = 1 for all i,j,k.
    
    # Let me use a known valid set. I'll define it carefully.
    
    # Term 0: A[0,0] * B[0,0] -> C[0,0]
    U[0*3+0, 0] = 1; V[0*3+0, 0] = 1; W[0*3+0, 0] = 1
    
    # Term 1: A[0,1] * B[1,0] -> C[0,0]
    U[0*3+1, 1] = 1; V[1*3+0, 1] = 1; W[0*3+0, 1] = 1
    
    # Term 2: A[0,2] * B[2,0] -> C[0,0]
    U[0*3+2, 2] = 1; V[2*3+0, 2] = 1; W[0*3+0, 2] = 1
    
    # Term 3: A[1,0] * B[0,1] -> C[1,0]
    U[1*3+0, 3] = 1; V[0*3+1, 3] = 1; W[1*3+0, 3] = 1
    
    # Term 4: A[1,1] * B[1,1] -> C[1,1]
    U[1*3+1, 4] = 1; V[1*3+1, 4] = 1; W[1*3+1, 4] = 1
    
    # Term 5: A[1,2] * B[2,1] -> C[1,1]
    U[1*3+2, 5] = 1; V[2*3+1, 5] = 1; W[1*3+1, 5] = 1
    
    # Term 6: A[2,0] * B[0,2] -> C[2,0]
    U[2*3+0, 6] = 1; V[0*3+2, 6] = 1; W[2*3+0, 6] = 1
    
    # Term 7: A[2,1] * B[1,2] -> C[2,1]
    U[2*3+1, 7] = 1; V[1*3+2, 7] = 1; W[2*3+1, 7] = 1
    
    # Term 8: A[2,2] * B[2,2] -> C[2,2]
    U[2*3+2, 8] = 1; V[2*3+2, 8] = 1; W[2*3+2, 8] = 1
    
    # Now the 14 composite terms. I need to be very careful here.
    # Let me use a systematic approach based on the structure.
    
    # For the off-diagonal C elements, we need:
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    
    # And the diagonal elements also need additional terms:
    # C[0,0] already has terms 0,1,2. But we also need to account for the composite terms.
    # Actually, the 9 direct terms above give:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] ✓ (complete)
    # C[1,1] = A[1,1]*B[1,1] + A[1,2]*B[2,1] + A[1,0]*B[0,1] ✓ (complete)  
    # C[2,2] = A[2,2]*B[2,2] + A[2,1]*B[1,2] + A[2,0]*B[0,2] ✓ (complete)
    
    # Wait, that's not right. Let me recheck:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # Term 0: A[0,0]*B[0,0] -> C[0,0] ✓
    # Term 1: A[0,1]*B[1,0] -> C[0,0] ✓
    # Term 2: A[0,2]*B[2,0] -> C[0,0] ✓
    # So C[0,0] is complete with terms 0,1,2.
    
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # Term 3: A[1,0]*B[0,1] -> C[1,0] ✗ (this goes to C[1,0], not C[1,1])
    # Hmm, I made an error. Let me redo this.
    
    # The issue is that the "direct" terms don't naturally group by C element.
    # Let me use a completely different, verified Laderman construction.
    
    # I'll use the formulation from the original Laderman paper, carefully encoded.
    # From Laderman (1976), the 23 products are:
    
    # Reset and start fresh
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # I'll use a known valid Laderman decomposition.
    # Based on standard references, here are the 23 terms:
    
    # Let me use a computational verification approach.
    # I'll define candidate terms and verify them.
    
    # Actually, let me use a different, simpler approach:
    # Use 2x2 Strassen for the outer level and 3x3 schoolbook for inner.
    # That's what the current program does (rank 189).
    # 
    # To improve, I can try:
    # 1. 3x3 Laderman (23) * 2x2 schoolbook (4) = 92
    # 2. 2x2 Strassen (7) * 3x3 schoolbook (27) = 189 (current)
    # 3. 3x3 schoolbook (27) * 2x2 Strassen (7) = 189 (same)
    
    # Option 1 is much better if I can get a valid Laderman.
    # Let me try to construct it carefully and verify.
    
    # I'll use the following approach: define the 23 terms by their
    # contributions and verify that the sum equals the multiplication tensor.
    
    # For now, let me return None and use the fallback in construct().
    return None

def _make_laderman_3x3_v2():
    """
    Alternative Laderman 3x3 construction with verification.
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # I'll use a systematic construction.
    # The key idea: we need 23 rank-1 tensors that sum to T[i,j,k]=1.
    
    # Let me use the following verified set from a reliable source.
    # I'll encode it as a list of (U_entries, V_entries, W_entries)
    # where each entry is a list of (index, coefficient).
    
    # Based on the Laderman algorithm as implemented in various references:
    
    terms = [
        # Term 0: A[0,0]*B[0,0]
        {
            'U': [(0, 1)],  # A[0,0]
            'V': [(0, 1)],  # B[0,0]
            'W': [(0, 1)],  # C[0,0] (k=0, i=0)
        },
        # Term 1: A[0,1]*B[1,0]
        {
            'U': [(1, 1)],  # A[0,1]
            'V': [(3, 1)],  # B[1,0]
            'W': [(0, 1)],  # C[0,0]
        },
        # Term 2: A[0,2]*B[2,0]
        {
            'U': [(2, 1)],  # A[0,2]
            'V': [(6, 1)],  # B[2,0]
            'W': [(0, 1)],  # C[0,0]
        },
        # Term 3: A[1,0]*B[0,1]
        {
            'U': [(3, 1)],  # A[1,0]
            'V': [(1, 1)],  # B[0,1]
            'W': [(1, 1)],  # C[1,0] (k=0, i=1)
        },
        # Term 4: A[1,1]*B[1,1]
        {
            'U': [(4, 1)],  # A[1,1]
            'V': [(4, 1)],  # B[1,1]
            'W': [(4, 1)],  # C[1,1] (k=1, i=1)
        },
        # Term 5: A[1,2]*B[2,1]
        {
            'U': [(5, 1)],  # A[1,2]
            'V': [(7, 1)],  # B[2,1]
            'W': [(4, 1)],  # C[1,1]
        },
        # Term 6: A[2,0]*B[0,2]
        {
            'U': [(6, 1)],  # A[2,0]
            'V': [(2, 1)],  # B[0,2]
            'W': [(2, 1)],  # C[2,0] (k=0, i=2)
        },
        # Term 7: A[2,1]*B[1,2]
        {
            'U': [(7, 1)],  # A[2,1]
            'V': [(5, 1)],  # B[1,2]
            'W': [(5, 1)],  # C[2,1] (k=1, i=2)
        },
        # Term 8: A[2,2]*B[2,2]
        {
            'U': [(8, 1)],  # A[2,2]
            'V': [(8, 1)],  # B[2,2]
            'W': [(8, 1)],  # C[2,2] (k=2, i=2)
        },
        # Term 9: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        {
            'U': [(0, 1), (4, 1)],
            'V': [(0, 1), (4, 1)],
            'W': [(1, 1), (3, 1)],  # C[0,1] and C[1,0]
        },
        # Term 10: (A[0,0]+A[2,2])*(B[0,0]+B[2,2])
        {
            'U': [(0, 1), (8, 1)],
            'V': [(0, 1), (8, 1)],
            'W': [(2, 1), (6, 1)],  # C[0,2] and C[2,0]
        },
        # Term 11: (A[1,1]+A[2,2])*(B[1,1]+B[2,2])
        {
            'U': [(4, 1), (8, 1)],
            'V': [(4, 1), (8, 1)],
            'W': [(5, 1), (7, 1)],  # C[1,2] and C[2,1]
        },
        # Term 12: (A[0,0]+A[0,1]+A[1,0]+A[1,1])*(B[0,0]+B[0,1]+B[1,0]+B[1,1])
        {
            'U': [(0, 1), (1, 1), (3, 1), (4, 1)],
            'V': [(0, 1), (1, 1), (3, 1), (4, 1)],
            'W': [(0, 1), (1, 1), (3, 1), (4, 1)],  # C[0,0], C[0,1], C[1,0], C[1,1]
        },
        # Term 13: (A[0,0]+A[0,2]+A[2,0]+A[2,2])*(B[0,0]+B[0,2]+B[2,0]+B[2,2])
        {
            'U': [(0, 1), (2, 1), (6, 1), (8, 1)],
            'V': [(0, 1), (2, 1), (6, 1), (8, 1)],
            'W': [(0, 1), (2, 1), (6, 1), (8, 1)],  # C[0,0], C[0,2], C[2,0], C[2,2]
        },
        # Term 14: (A[1,1]+A[1,2]+A[2,1]+A[2,2])*(B[1,1]+B[1,2]+B[2,1]+B[2,2])
        {
            'U': [(4, 1), (5, 1), (7, 1), (8, 1)],
            'V': [(4, 1), (5, 1), (7, 1), (8, 1)],
            'W': [(4, 1), (5, 1), (7, 1), (8, 1)],  # C[1,1], C[1,2], C[2,1], C[2,2]
        },
        # Term 15: (A[0,0]+A[0,1]+A[1,0]+A[1,1]+A[0,2]+A[1,2]+A[2,0]+A[2,2])*(B[0,0]+B[0,1]+B[1,0]+B[1,1]+B[0,2]+B[1,2]+B[2,0]+B[2,2])
        {
            'U': [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (8, 1)],
            'V': [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (8, 1)],
            'W': [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (8, 1)],
        },
        # Term 16: (A[0,0]-A[0,1]+A[1,0]-A[1,1])*(B[0,0]-B[0,1]+B[1,0]-B[1,1])
        {
            'U': [(0, 1), (1, -1), (3, 1), (4, -1)],
            'V': [(0, 1), (1, -1), (3, 1), (4, -1)],
            'W': [(0, 1), (1, -1), (3, 1), (4, -1)],
        },
        # Term 17: (A[0,0]-A[0,2]+A[2,0]-A[2,2])*(B[0,0]-B[0,2]+B[2,0]-B[2,2])
        {
            'U': [(0, 1), (2, -1), (6, 1), (8, -1)],
            'V': [(0, 1), (2, -1), (6, 1), (8, -1)],
            'W': [(0, 1), (2, -1), (6, 1), (8, -1)],
        },
        # Term 18: (A[1,1]-A[1,2]+A[2,1]-A[2,2])*(B[1,1]-B[1,2]+B[2,1]-B[2,2])
        {
            'U': [(4, 1), (5, -1), (7, 1), (8, -1)],
            'V': [(4, 1), (5, -1), (7, 1), (8, -1)],
            'W': [(4, 1), (5, -1), (7, 1), (8, -1)],
        },
        # Term 19: (A[0,1]+A[1,1]-A[0,2]-A[1,2])*(B[1,0]+B[1,1]-B[2,0]-B[2,1])
        {
            'U': [(1, 1), (4, 1), (2, -1), (5, -1)],
            'V': [(3, 1), (4, 1), (6, -1), (7, -1)],
            'W': [(1, 1), (4, 1)],  # C[0,1], C[1,1]
        },
        # Term 20: (A[1,0]+A[1,1]-A[2,0]-A[2,1])*(B[0,0]+B[0,1]-B[0,2]-B[1,2])
        {
            'U': [(3, 1), (4, 1), (6, -1), (7, -1)],
            'V': [(0, 1), (1, 1), (2, -1), (5, -1)],
            'W': [(0, 1), (1, 1)],  # C[1,0], C[2,0]... wait
        },
        # Term 21: (A[0,1]+A[0,2]-A[1,1]-A[1,2])*(B[2,0]+B[2,1]-B[1,0]-B[1,1])
        {
            'U': [(1, 1), (2, 1), (4, -1), (5, -1)],
            'V': [(6, 1), (7, 1), (3, -1), (4, -1)],
            'W': [(2, 1), (5, 1)],  # C[0,2], C[1,2]
        },
        # Term 22: (A[1,0]+A[2,0]-A[1,1]-A[2,1])*(B[0,2]+B[1,2]-B[0,0]-B[1,0])
        {
            'U': [(3, 1), (6, 1), (4, -1), (7, -1)],
            'V': [(2, 1), (5, 1), (0, -1), (3, -1)],
            'W': [(2, 1), (5, 1)],  # C[2,0], C[2,1]
        },
    ]
    
    for c, term in enumerate(terms):
        for idx, coeff in term['U']:
            U[idx, c] += coeff
        for idx, coeff in term['V']:
            V[idx, c] += coeff
        for idx, coeff in term['W']:
            W[idx, c] += coeff
    
    # Verify
    max_err = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_val = U[i*3+j, :]
                v_val = V[j*3+k, :]
                w_val = W[k*3+i, :]
                s = np.dot(u_val, v_val * w_val)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        return None
    
    return U, V, W

def _make_2x2_schoolbook():
    """
    Schoolbook 2x2 matrix multiplication, rank 4.
    Returns (U, V, W) as 2D arrays of shape (4, 4).
    """
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c = i*4 + j*2 + k  # column index
                U[i*2+j, c] = 1
                V[j*2+k, c] = 1
                W[k*2+i, c] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    
    # Fallback: Schoolbook 6x6 (Rank 216)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
        
    # Hybrid: 2x2 Strassen (Outer) x 3x3 Schoolbook (Inner) (Rank 189)
    def _hybrid_strassen_schoolbook():
        block_size = 3
        num_blocks = 2
        dim = n
        
        rank_inner = block_size ** 3 # 27
        rank_outer = 7
        total_rank = rank_inner * rank_outer
        
        U = [[0] * total_rank for _ in range(dim * dim)]
        V = [[0] * total_rank for _ in range(dim * dim)]
        W = [[0] * total_rank for _ in range(dim * dim)]
        
        col = 0
        
        steps = [
            {'A': { (0,0): 1, (1,1): 1 }, 'B': { (0,0): 1, (1,1): 1 }, 'C': { (0,0): 1, (1,1): 1 }},
            {'A': { (1,0): 1, (1,1): 1 }, 'B': { (0,0): 1 }, 'C': { (1,0): 1, (1,1): -1 }},
            {'A': { (0,0): 1 }, 'B': { (0,1): 1, (1,1): -1 }, 'C': { (0,1): 1, (1,1): 1 }},
            {'A': { (1,1): 1 }, 'B': { (1,0): 1, (0,0): -1 }, 'C': { (0,0): 1, (1,0): 1 }},
            {'A': { (0,0): 1, (0,1): 1 }, 'B': { (1,1): 1 }, 'C': { (0,0): -1, (0,1): 1 }},
            {'A': { (1,0): 1, (0,0): -1 }, 'B': { (0,0): 1, (0,1): 1 }, 'C': { (1,1): 1 }},
            {'A': { (0,1): 1, (1,1): -1 }, 'B': { (1,0): 1, (1,1): 1 }, 'C': { (0,0): 1 }}
        ]
        
        for step in steps:
            A_coeffs = step['A']
            B_coeffs = step['B']
            C_coeffs = step['C']
            
            for u in range(block_size):
                for v in range(block_size):
                    for w in range(block_size):
                        for (I, J), ca in A_coeffs.items():
                            if ca != 0:
                                i_global = I * block_size + u
                                j_global = J * block_size + v
                                idx_u = i_global * dim + j_global
                                U[idx_u][col] += ca
                        for (J, K), cb in B_coeffs.items():
                            if cb != 0:
                                j_global = J * block_size + v
                                k_global = K * block_size + w
                                idx_v = j_global * dim + k_global
                                V[idx_v][col] += cb
                        for (I, K), cc in C_coeffs.items():
                            if cc != 0:
                                i_global = I * block_size + u
                                k_global = K * block_size + w
                                idx_w = k_global * dim + i_global
                                W[idx_w][col] += cc
                        col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try Laderman 3x3 (outer) x 2x2 schoolbook (inner) = rank 92
    def _hybrid_laderman_schoolbook():
        laderman = _make_laderman_3x3_v2()
        if laderman is None:
            return None
        
        U_l, V_l, W_l = laderman
        U_2, V_2, W_2 = _make_2x2_schoolbook()
        
        rank_l = U_l.shape[1]  # 23
        rank_2 = U_2.shape[1]  # 4
        total_rank = rank_l * rank_2  # 92
        
        U = np.zeros((36, total_rank))
        V = np.zeros((36, total_rank))
        W = np.zeros((36, total_rank))
        
        block_size = 2  # inner block size
        num_outer_blocks = 3  # 6/2 = 3
        
        for c_l in range(rank_l):
            for c_2 in range(rank_2):
                col = c_l * rank_2 + c_2
                
                # U: outer index (i_o, j_o) -> inner index (u, v)
                # Global: i = i_o * 2 + u, j = j_o * 2 + v
                # U_l index: i_o * 3 + j_o
                # U_2 index: u * 2 + v
                for i_o in range(3):
                    for j_o in range(3):
                        ul = U_l[i_o * 3 + j_o, c_l]
                        if ul != 0:
                            for u in range(2):
                                for v in range(2):
                                    u2 = U_2[u * 2 + v, c_2]
                                    if u2 != 0:
                                        i_global = i_o * 2 + u
                                        j_global = j_o * 2 + v
                                        U[i_global * 6 + j_global, col] += ul * u2
                
                # V: outer index (j_o, k_o) -> inner index (v, w)
                # Global: j = j_o * 2 + v, k = k_o * 2 + w
                # V_l index: j_o * 3 + k_o
                # V_2 index: v * 2 + w
                for j_o in range(3):
                    for k_o in range(3):
                        vl = V_l[j_o * 3 + k_o, c_l]
                        if vl != 0:
                            for v in range(2):
                                for w in range(2):
                                    v2 = V_2[v * 2 + w, c_2]
                                    if v2 != 0:
                                        j_global = j_o * 2 + v
                                        k_global = k_o * 2 + w
                                        V[j_global * 6 + k_global, col] += vl * v2
                
                # W: outer index (k_o, i_o) -> inner index (w, u)
                # Global: k = k_o * 2 + w, i = i_o * 2 + u
                # W_l index: k_o * 3 + i_o
                # W_2 index: w * 2 + u
                for k_o in range(3):
                    for i_o in range(3):
                        wl = W_l[k_o * 3 + i_o, c_l]
                        if wl != 0:
                            for w in range(2):
                                for u in range(2):
                                    w2 = W_2[w * 2 + u, c_2]
                                    if w2 != 0:
                                        k_global = k_o * 2 + w
                                        i_global = i_o * 2 + u
                                        W[k_global * 6 + i_global, col] += wl * w2
        
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try Laderman hybrid (rank 92)
    try:
        res = _hybrid_laderman_schoolbook()
        if res is not None:
            # Verify
            U_np = np.array(res["U"], dtype=float)
            V_np = np.array(res["V"], dtype=float)
            W_np = np.array(res["W"], dtype=float)
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    u_row = U_np[i * m + j, :]
                    for k in range(p):
                        v_row = V_np[j * p + k, :]
                        w_row = W_np[k * n + i, :]
                        s = np.dot(u_row, v_row * w_row)
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-10:
                return res
    except Exception:
        pass
    
    # Try Hybrid (189)
    try:
        res = _hybrid_strassen_schoolbook()
        # Verify
        U_np = np.array(res["U"], dtype=float)
        V_np = np.array(res["V"], dtype=float)
        W_np = np.array(res["W"], dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_np[i * m + j, :]
                for k in range(p):
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    s = np.dot(u_row, v_row * w_row)
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-10:
            return res
    except Exception:
        pass
        
    # Fallback to Schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
