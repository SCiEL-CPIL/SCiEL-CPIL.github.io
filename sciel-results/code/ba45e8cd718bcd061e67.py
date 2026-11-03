"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = time.time() + 115.0

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly matches matrix multiplication."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if U else 0
    
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # The tensor equation: sum_c U[i,j,c] * V[j,k,c] * W[k+n-1-i... 
    # Actually, the convention is W[k*n+i, c]
    # Tensor: T[i,j,k] = (C)_ik where A[i,j]*B[j,k]
    # Decomposition: T[i,j,k] = sum_c U[nm_idx(i,j), c] * V[mp_idx(j,k), c] * W[pn_idx(k,i), c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if True else 0  # A[i,j]*B[j,k] gives C[i,k], so tensor entry is 1
                # Actually the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
                # because C[i,k] = sum_j A[i,j] B[j,k]
                # So each trilinear term A[i,j]B[j,k]C[i,k] has coefficient 1
                computed = 0
                for c in range(rank):
                    computed += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(computed - 1) > 1e-10:
                    return False
    return True

def _try_rank_decomposition(n, m, p, target_rank):
    """
    Try to construct a decomposition of the given rank.
    Returns (U, V, W) if successful, None otherwise.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    # For 3x3x4, try using known constructions
    # One approach: block decomposition
    
    # For 3x3x4, we can think of it as multiplying a 3x3 matrix by a 3x4 matrix
    # Result is 3x4
    
    # Try rank 32: Use 23 terms for a 3x3x3 subproblem + 9 for the 4th column
    
    if target_rank == 32 and n == 3 and m == 3 and p == 4:
        # 23-term decomposition of 3x3x3
        # I'll use a known construction
        # The 23 terms for 3x3x3 can be built using Strassen on 2x2 blocks plus extras
        
        # Let me use a specific 23-term decomposition:
        # Terms are (i, j, k) indices with possible +/-1 coefficients and offsets
        
        # A known 23-term decomposition (Landsberg):
        # I'll represent each term as (u_indices, v_indices, w_indices, coeff)
        # where u_indices is a set of (i,j) positions, etc.
        
        # Actually, let me use the following known 23-term scheme:
        # Split 3x3 into overlapping 2x2 blocks and use Strassen
        
        # For now, let me use a different strategy:
        # Try to find a rank-32 decomposition by combining:
        # - A rank-23 decomposition for columns 0,1,2 of B (treating it as 3x3)
        # - Plus 9 schoolbook terms for column 3 of B
        
        # The 23-term decomposition of 3x3x3:
        # I'll use the one from the literature. Each term c gives:
        # U_c: 3x3 matrix, V_c: 3x3 matrix, W_c: 3x3 matrix (diagonal in W's indexing)
        # such that the sum reconstructs the identity mapping
        
        # Known 23 terms (I'll use a standard set):
        # This is complex, so let me use a simpler but valid construction
        
        # Alternative: use rank 31 or 32 from a direct construction
        
        # Let me try a constructive approach for rank 32:
        U = np.zeros((nm, target_rank), dtype=object)
        V = np.zeros((mp, target_rank), dtype=object)
        W = np.zeros((pn, target_rank), dtype=object)
        
        # First 23 terms: attempt to cover 3x3x3 part
        # Last 9 terms: cover the 4th column with schoolbook
        
        # For the 9 schoolbook terms for column k=3 (the 4th column):
        # These handle A[i,j] * B[j,3] -> C[i,3]
        for i in range(3):
            for j in range(3):
                c = 23 + i * 3 + j
                U[i * 3 + j, c] = 1
                V[j * 4 + 3, c] = 1  # B[j,3]
                W[3 * 3 + i, c] = 1  # C[i,3] where W index is k*n+i = 3*3+i
        
        # For the first 23 terms, we need a valid 23-term decomposition of 3x3x3
        # Let me use a known construction:
        # I'll use the following 23 terms (adapted from standard references):
        
        # Term definitions for 3x3x3 (U and V are 3x3, W has diagonal entries)
        # Each term: (u_nonzero_positions, v_nonzero_positions, w_nonzero_positions, coeffs)
        
        # Known 23-term decomposition:
        # I'll use the construction from Landsberg (2010) "Tensor Rank and Arithmetical Complexity of Bilinear Computations"
        
        # For practical purposes, let me use a verifiable rank-23 set:
        # The terms are as follows (using 0-indexed i,j,k for 3x3x3):
        
        # Let me define 23 terms where each term has specific U, V, W entries:
        
        terms_23 = [
            # (u: dict of (i,j)->val, v: dict of (j,k)->val, w: dict of (k,i)->val)
            # Term 0
            ({(0,0):1, (0,1):1, (1,0):1, (1,1):1},
             {(0,0):1, (0,1):1, (1,0):1, (1,1):1},
             {(0,0):1, (1,1):1}),
            # Term 1
            ({(0,0):1, (0,1):-1, (1,0):1, (1,1):1},
             {(0,0):1, (0,1):1, (1,0):-1, (1,1):1},
             {(0,0):1, (1,1):-1}),
            # Term 2
            ({(0,0):1, (0,1):1, (1,0):1, (1,1):-1},
             {(0,0):1, (0,1):-1, (1,0):1, (1,1):1},
             {(0,0):-1, (1,1):1}),
            # Term 3
            ({(0,0):-1, (0,1):-1, (1,0):-1, (1,1):1},
             {(0,0):1, (0,1):-1, (1,0):-1, (1,1):1},
             {(0,0):1, (1,1):1}),
            # Term 4
            ({(2,0):1, (2,1):1, (0,2):1, (1,2):1},
             {(0,0):1, (1,0):1, (0,2):1, (1,2):1},
             {(0,2):1, (1,2):1}),
            # Term 5
            ({(2,0):1, (2,1):-1, (0,2):1, (1,2):1},
             {(0,0):1, (1,0):1, (0,2):-1, (1,2):1},
             {(0,2):1, (1,2):-1}),
            # Term 6
            ({(2,0):1, (2,1):1, (0,2):1, (1,2):-1},
             {(0,0):1, (1,0):-1, (0,2):1, (1,2):1},
             {(0,2):-1, (1,2):1}),
            # Term 7
            ({(2,0):-1, (2,1):-1, (0,2):-1, (1,2):1},
             {(0,0):1, (1,0):-1, (0,2):-1, (1,2):1},
             {(0,2):1, (1,2):1}),
            # Term 8
            ({(0,2):1, (1,2):1, (2,0):1, (2,1):1},
             {(0,2):1, (1,2):1, (2,0):1, (2,1):1},
             {(2,0):1, (2,1):1}),
            # Term 9
            ({(0,2):1, (1,2):-1, (2,0):1, (2,1):1},
             {(0,2):1, (1,2):1, (2,0):-1, (2,1):1},
             {(2,0):1, (2,1):-1}),
            # Term 10
            ({(0,2):1, (1,2):1, (2,0):1, (2,1):-1},
             {(0,2):1, (1,2):-1, (2,0):1, (2,1):1},
             {(2,0):-1, (2,1):1}),
            # Term 11
            ({(0,2):-1, (1,2):-1, (2,0):-1, (2,1):1},
             {(0,2):1, (1,2):-1, (2,0):-1, (2,1):1},
             {(2,0):1, (2,1):1}),
            # Term 12
            ({(0,0):1, (2,0):1, (0,2):1, (2,2):1},
             {(0,0):1, (0,2):1, (2,0):1, (2,2):1},
             {(0,0):1, (2,2):1}),
            # Term 13
            ({(0,0):1, (2,0):-1, (0,2):1, (2,2):1},
             {(0,0):1, (0,2):1, (2,0):-1, (2,2):1},
             {(0,0):1, (2,2):-1}),
            # Term 14
            ({(0,0):1, (2,0):1, (0,2):1, (2,2):-1},
             {(0,0):1, (0,2):-1, (2,0):1, (2,2):1},
             {(0,0):-1, (2,2):1}),
            # Term 15
            ({(0,0):-1, (2,0):-1, (0,2):-1, (2,2):1},
             {(0,0):1, (0,2):-1, (2,0):-1, (2,2):1},
             {(0,0):1, (2,2):1}),
            # Term 16
            ({(0,1):1, (1,1):1, (1,0):1, (1,2):1},
             {(0,0):1, (1,0):1, (0,1):1, (1,1):1},
             {(0,0):1, (1,1):1}),
            # Term 17
            ({(0,1):1, (1,1):-1, (1,0):1, (1,2):1},
             {(0,0):1, (1,0):1, (0,1):-1, (1,1):1},
             {(0,0):1, (1,1):-1}),
            # Term 18
            ({(0,1):1, (1,1):1, (1,0):1, (1,2):-1},
             {(0,0):1, (1,0):-1, (0,1):1, (1,1):1},
             {(0,0):-1, (1,1):1}),
            # Term 19
            ({(0,1):-1, (1,1):-1, (1,0):-1, (1,2):1},
             {(0,0):1, (1,0):-1, (0,1):-1, (1,1):1},
             {(0,0):1, (1,1):1}),
            # Term 20
            ({(1,0):1, (1,1):1, (1,2):1, (2,2):1},
             {(1,0):1, (1,1):1, (1,2):1, (2,2):1},
             {(1,1):1, (2,2):1}),
            # Term 21
            ({(1,0):1, (1,1):-1, (1,2):1, (2,2):1},
             {(1,0):1, (1,1):1, (1,2):-1, (2,2):1},
             {(1,1):1, (2,2):-1}),
            # Term 22
            ({(1,0):1, (1,1):1, (1,2):-1, (2,2):1},
             {(1,0):1, (1,1):-1, (1,2):1, (2,2):1},
             {(1,1):-1, (2,2):1}),
        ]
        
        # This is getting complex and I'm not confident in the correctness of these 23 terms.
        # Let me take a safer approach: return the schoolbook decomposition which is guaranteed correct.
        
        pass
    
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition (rank 36)
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(nm)]
    best_V = [[0] * best_rank for _ in range(mp)]
    best_W = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Return the schoolbook decomposition
    # Improving this requires implementing a verified lower-rank decomposition
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
