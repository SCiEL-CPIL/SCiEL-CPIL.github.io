"""Matrix multiplication 4x4x6 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Uses Strassen's algorithm for the 2x2x2 base case, combined with a split-base-case
    approach for the 2x2x3 sub-problem (rank-4 schoolbook 2x2x1 + rank-7 Strassen 2x2x2).
    
    Strategy:
    - 4x4x6 = (2x2 blocks) where each 2x2 block of A multiplies 2x4 blocks of B
    - For each 2x2x4 sub-problem: split B into two 2x2 blocks
      - First 2x2x2: Strassen rank 7
      - Second 2x2x2: Strassen rank 7
    - Total: 4 (A blocks) * 2 (B blocks per A row pair) * 7 = 56? 
      Wait, let me think more carefully.
    
    Actually, let's think about the tensor structure:
    - A is 4x4, B is 4x6, C is 4x6
    - C[i,k] = sum_j A[i,j] * B[j,k]
    
    Using Strassen-style decomposition:
    Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2
    Split B into 2x3 blocks: B = [[B11, B12], [B21, B22]] where B11,B12 are 2x3, etc.
    Actually B is 4x6, so split rows into 2+2, columns into 3+3:
    B = [[B11, B12], [B21, B22]] where each Bij is 2x3
    
    C = [[C11, C12], [C21, C22]] where Cij is 2x3
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    Each product like A11*B11 is a 2x2 * 2x3 = 2x3 matrix product.
    
    For 2x2x3: We can use the split-base-case:
    - 2x2x1 (first column of B): schoolbook rank 4
    - 2x2x2 (second and third columns... wait, 2x2x2 means 2x2 * 2x2)
    
    Actually, 2x2x3 means A is 2x2, B is 2x3, C is 2x3.
    Split B into B[:,0] (2x1) and B[:,1:3] (2x2).
    A*B[:,0]: 2x2 * 2x1, schoolbook rank 4
    A*B[:,1:3]: 2x2 * 2x2, Strassen rank 7
    Total for 2x2x3: 4 + 7 = 11
    
    So for each of the 4 block products (Aij * Bkl), we need rank 11.
    Total rank = 4 * 11 = 44.
    
    Let me implement this.
    """
    start_time = time.time()
    DEADLINE = start_time + 120  # safety margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # We'll build U, V, W as lists of lists
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    # Strategy: Use Strassen for 2x2x2 blocks, schoolbook for 2x2x1
    # 
    # Block decomposition:
    # A (4x4) -> 2x2 blocks of 2x2: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
    # B (4x6) -> 2x3 blocks: B[0:2,0:3], B[0:2,3:6], B[2:4,0:3], B[2:4,3:6]
    # C (4x6) -> 2x3 blocks: C[0:2,0:3], C[0:2,3:6], C[2:4,0:3], C[2:4,3:6]
    #
    # C[0:2,0:3] = A[0:2,0:2]*B[0:2,0:3] + A[0:2,2:4]*B[2:4,0:3]
    # C[0:2,3:6] = A[0:2,0:2]*B[0:2,3:6] + A[0:2,2:4]*B[2:4,3:6]
    # C[2:4,0:3] = A[2:4,0:2]*B[0:2,0:3] + A[2:4,2:4]*B[2:4,0:3]
    # C[2:4,3:6] = A[2:4,0:2]*B[0:2,3:6] + A[2:4,2:4]*B[2:4,3:6]
    
    # For each 2x2x3 product, we use rank 11 (4 schoolbook + 7 Strassen)
    # Total rank = 4 (C blocks) * 2 (A blocks per C block) * 11 = 88? 
    # Wait no. Let me recount.
    #
    # Each C block (2x3) is a sum of 2 products of type (2x2)*(2x3).
    # Each (2x2)*(2x3) product uses rank 11.
    # So each C block uses 2 * 11 = 22 rank.
    # 4 C blocks: 4 * 22 = 88. That's worse than schoolbook (96)? No, 88 < 96.
    #
    # Actually wait, the schoolbook for 4x4x6 is 4*4*6 = 96.
    # 88 < 96, so this is an improvement.
    
    # Let me implement the 2x2x3 decomposition:
    # For A (2x2) * B (2x3) = C (2x3):
    # Split B = [B_left (2x1), B_right (2x2)]
    # C_left = A * B_left: schoolbook rank 4
    # C_right = A * B_right: Strassen rank 7
    #
    # Schoolbook for 2x2x1:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # Rank 4 terms:
    # term0: A[0,0]*B[0,0] -> C[0,0]
    # term1: A[0,1]*B[1,0] -> C[0,0]
    # term2: A[1,0]*B[0,0] -> C[1,0]
    # term3: A[1,1]*B[1,0] -> C[1,0]
    
    # Strassen for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_term(U, V, W, rank_offset, a_idx, b_idx, c_idx, coeff_a=1, coeff_b=1, coeff_c=1):
        """Add a rank-1 term to the decomposition."""
        U[a_idx][rank_offset] += coeff_a
        V[b_idx][rank_offset] += coeff_b
        W[c_idx][rank_offset] += coeff_c
    
    def add_strassen_2x2x2(U, V, W, rank_offset, a_base, b_base):
        """
        Add Strassen rank-7 terms for 2x2x2.
        a_base: (i0, j0) top-left of A block in (i,j) coords
        b_base: (j0, k0) top-left of B block in (j,k) coords
        Returns new rank_offset.
        """
        i0, j0 = a_base
        jb, k0 = b_base
        
        # M1 = (A00+A11)*(B00+B11)
        U[i0*m + j0][rank_offset] += 1
        U[(i0+1)*m + (j0+1)][rank_offset] += 1
        V[jb*p + k0][rank_offset] += 1
        V[(jb+1)*p + (k0+1)][rank_offset] += 1
        W[k0*n + i0][rank_offset] += 1
        W[k0*n + (i0+1)][rank_offset] += 1
        W[(k0+1)*n + i0][rank_offset] += 1
        W[(k0+1)*n + (i0+1)][rank_offset] += 1
        rank_offset += 1
        
        # M2 = (A10+A11)*B00
        U[(i0+1)*m + j0][rank_offset] += 1
        U[(i0+1)*m + (j0+1)][rank_offset] += 1
        V[jb*p + k0][rank_offset] += 1
        W[(k0+1)*n + (i0+1)][rank_offset] += 1
        rank_offset += 1
        
        # M3 = A00*(B01-B11)
        U[i0*m + j0][rank_offset] += 1
        V[jb*p + (k0+1)][rank_offset] += 1
        V[(jb+1)*p + (k0+1)][rank_offset] -= 1
        W[k0*n + i0][rank_offset] += 1
        W[(k0+1)*n + i0][rank_offset] -= 1
        rank_offset += 1
        
        # M4 = A11*(B10-B00)
        U[(i0+1)*m + (j0+1)][rank_offset] += 1
        V[(jb+1)*p + k0][rank_offset] += 1
        V[jb*p + k0][rank_offset] -= 1
        W[(k0+1)*n + (i0+1)][rank_offset] += 1
        W[k0*n + (i0+1)][rank_offset] -= 1
        rank_offset += 1
        
        # M5 = (A00+A01)*B11
        U[i0*m + j0][rank_offset] += 1
        U[i0*m + (j0+1)][rank_offset] += 1
        V[(jb+1)*p + (k0+1)][rank_offset] += 1
        W[k0*n + i0][rank_offset] += 1
        W[(k0+1)*n + i0][rank_offset] += 1
        rank_offset += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[(i0+1)*m + j0][rank_offset] += 1
        U[i0*m + j0][rank_offset] -= 1
        V[jb*p + k0][rank_offset] += 1
        V[jb*p + (k0+1)][rank_offset] += 1
        W[(k0+1)*n + (i0+1)][rank_offset] += 1
        W[(k0+1)*n + i0][rank_offset] -= 1
        rank_offset += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[i0*m + (j0+1)][rank_offset] += 1
        U[(i0+1)*m + (j0+1)][rank_offset] -= 1
        V[(jb+1)*p + k0][rank_offset] += 1
        V[(jb+1)*p + (k0+1)][rank_offset] += 1
        W[k0*n + i0][rank_offset] += 1
        W[k0*n + (i0+1)][rank_offset] -= 1
        rank_offset += 1
        
        return rank_offset
    
    def add_2x2x3_decomp(U, V, W, rank_offset, a_base, b_base):
        """
        Add rank-11 terms for 2x2x3 decomposition.
        a_base: (i0, j0) top-left of A (2x2) block
        b_base: (j0, k0) top-left of B (2x3) block
        Returns new rank_offset.
        """
        i0, j0 = a_base
        jb, k0 = b_base
        
        # Schoolbook part for B[:,0] (2x1): rank 4
        # term0: A[0,0]*B[0,0]
        U[i0*m + j0][rank_offset] += 1
        V[jb*p + k0][rank_offset] += 1
        W[k0*n + i0][rank_offset] += 1
        rank_offset += 1
        
        # term1: A[0,1]*B[1,0]
        U[i0*m + (j0+1)][rank_offset] += 1
        V[(jb+1)*p + k0][rank_offset] += 1
        W[k0*n + i0][rank_offset] += 1
        rank_offset += 1
        
        # term2: A[1,0]*B[0,0]
        U[(i0+1)*m + j0][rank_offset] += 1
        V[jb*p + k0][rank_offset] += 1
        W[k0*n + (i0+1)][rank_offset] += 1
        rank_offset += 1
        
        # term3: A[1,1]*B[1,0]
        U[(i0+1)*m + (j0+1)][rank_offset] += 1
        V[(jb+1)*p + k0][rank_offset] += 1
        W[k0*n + (i0+1)][rank_offset] += 1
        rank_offset += 1
        
        # Strassen part for B[:,1:3] (2x2): rank 7
        rank_offset = add_strassen_2x2x2(U, V, W, rank_offset, (i0, j0), (jb, k0+1))
        
        return rank_offset
    
    # Now build the full decomposition
    # We need to track the total rank first
    # Each C block (2x3) = sum of 2 (2x2x3) products
    # Each (2x2x3) = rank 11
    # Total rank = 4 C_blocks * 2 A_blocks * 11 = 88
    
    total_rank = 88
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # C[0:2, 0:3] = A[0:2,0:2]*B[0:2,0:3] + A[0:2,2:4]*B[2:4,0:3]
    rank_offset = add_2x2x3_decomp(U, V, W, rank_offset, (0, 0), (0, 0))
    rank_offset = add_2x2x3_decomp(U, V, W, rank_offset, (0, 2), (2, 0))
    
    # C[0:2, 3:6] = A[0:2,0:2]*B[0:2,3:6] + A[0:2,2:4]*B[2:4,3:6]
    rank_offset = add_2x2x3_decomp(U, V, W, rank_offset, (0, 0), (0, 3))
    rank_offset = add_2x2x3_decomp(U, V, W, rank_offset, (0, 2), (2, 3))
    
    # C[2:4, 0:3] = A[2:4,0:2]*B[0:2,0:3] + A[2:4,2:4]*B[2:4,0:3]
    rank_offset = add_2x2x3_decomp(U, V, W, rank_offset, (2, 0), (0, 0))
    rank_offset = add_2x2x3_decomp(U, V, W, rank_offset, (2, 2), (2, 0))
    
    # C[2:4, 3:6] = A[2:4,0:2]*B[0:2,3:6] + A[2:4,2:4]*B[2:4,3:6]
    rank_offset = add_2x2x3_decomp(U, V, W, rank_offset, (2, 0), (0, 3))
    rank_offset = add_2x2x3_decomp(U, V, W, rank_offset, (2, 2), (2, 3))
    
    if rank_offset != total_rank:
        raise ValueError(f"Rank mismatch: expected {total_rank}, got {rank_offset}")
    
    return {"U": U, "V": V, "W": W}
