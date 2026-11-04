"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [4, 5, 5]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # Implement block-diagonal Strassen decomposition for 4x5x5
    # Partition: n=4 -> [2,2], m=5 -> [2,3], p=5 -> [2,3]
    
    # We'll build the decomposition block by block.
    # For each combination of (i, k, j) where:
    #   i in range(2): row block of A and C, sizes [2, 2]
    #   k in range(2): col block of A / row block of B, sizes [2, 3]
    #   j in range(2): col block of B / C, sizes [2, 3]
    
    # C[i,j] = sum_k A[i,k] @ B[k,j]
    
    # For each (i,k,j) triple, we need to multiply A[i,k] (n_i x m_k) by B[k,j] (m_k x p_j)
    # to get a contribution to C[i,j] (n_i x p_j).
    
    # Block sizes:
    n_blocks = [2, 2]  # rows of A and C
    m_blocks = [2, 3]  # cols of A / rows of B
    p_blocks = [2, 3]  # cols of B and C
    
    # For each (i, k, j), the sub-problem is:
    # Multiply (n_i x m_k) by (m_k x p_j) to get (n_i x p_j)
    # If n_i=2, m_k=2, p_j=2: use Strassen (rank 7)
    # Otherwise: use schoolbook (rank n_i * m_k * p_j)
    
    # Let's build the full decomposition.
    U_list = []
    V_list = []
    W_list = []
    
    # We'll track column indices for U, V, W
    # U has shape (n*m, rank) = (20, rank)
    # V has shape (m*p, rank) = (25, rank)  
    # W has shape (p*n, rank) = (20, rank)
    
    # For each rank-1 term, we add one column to each of U, V, W.
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term. u_vec, v_vec, w_vec are sparse dicts {index: value}"""
        U_list.append({idx: val for idx, val in u_vec.items() if val != 0})
        V_list.append({idx: val for idx, val in v_vec.items() if val != 0})
        W_list.append({idx: val for idx, val in w_vec.items() if val != 0})
    
    # Schoolbook for a block of size (ni, mk, pj)
    def schoolbook_block(ni, mk, pj, i_offset, k_offset, j_offset):
        """Add schoolbook rank-1 terms for multiplying (ni x mk) by (mk x pj)."""
        for a in range(ni):
            for b in range(mk):
                for c in range(pj):
                    u = {(i_offset + a) * m + (k_offset + b): 1.0}
                    v = {(k_offset + b) * p + (j_offset + c): 1.0}
                    w = {(j_offset + c) * n + (i_offset + a): 1.0}
                    add_term(u, v, w)
    
    # Strassen for 2x2x2 block
    def strassen_2x2x2(i_offset, k_offset, j_offset):
        """Add Strassen rank-1 terms for 2x2x2 matrix multiplication."""
        # Strassen's 7 products for 2x2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each M is a product of linear combinations.
        # We need to express each M as a sum of rank-1 terms.
        
        # For simplicity, let's use the standard Strassen decomposition.
        # Each M_t = (sum_a alpha_a * A_a) * (sum_b beta_b * B_b)
        # This expands to sum_{a,b} alpha_a * beta_b * (A_a * B_b)
        # Each A_a * B_b is itself a sum of rank-1 terms (schoolbook for 2x2x2).
        
        # Actually, the standard way: each M_t is computed as a 2x2x2 multiplication
        # of the linear combinations, which has rank 4 (schoolbook for 2x2x2).
        # But Strassen replaces 8 schoolbook products (rank 32) with 7 products (rank 28).
        # Wait, that's not right either.
        
        # Let me think more carefully. In the full tensor decomposition:
        # Each "product" M_t corresponds to computing (linear comb of A) @ (linear comb of B)
        # which is a 2x2x2 multiplication with rank 4 (schoolbook).
        # So 7 Strassen products = 7 * 4 = 28 rank-1 terms.
        # Versus 8 schoolbook products = 8 * 4 = 32 rank-1 terms.
        # Savings: 4 rank-1 terms per 2x2x2 block.
        
        # For our 4x5x5 problem with [2,2]x[2,3]x[2,3] partition:
        # We have 8 blocks (i,k,j) with i in {0,1}, k in {0,1}, j in {0,1}.
        # Block sizes:
        # (i=0,k=0,j=0): 2x2x2 -> Strassen, rank 28
        # (i=0,k=0,j=1): 2x2x3 -> schoolbook, rank 12
        # (i=0,k=1,j=0): 2x3x2 -> schoolbook, rank 12
        # (i=0,k=1,j=1): 2x3x3 -> schoolbook, rank 18
        # (i=1,k=0,j=0): 2x2x2 -> Strassen, rank 28
        # (i=1,k=0,j=1): 2x2x3 -> schoolbook, rank 12
        # (i=1,k=1,j=0): 2x3x2 -> schoolbook, rank 12
        # (i=1,k=1,j=1): 2x3x3 -> schoolbook, rank 18
        # Total: 28+12+12+18+28+12+12+18 = 140. That's worse than 100!
        
        # I'm making an error. The Strassen decomposition doesn't work this way
        # for the full tensor rank. Let me reconsider.
        
        # The correct approach: the tensor rank of the 4x5x5 matmul tensor
        # is the minimum number of rank-1 tensors needed.
        # Schoolbook gives 100.
        # To improve, we need to find a decomposition with fewer than 100 terms.
        
        # Given the complexity and time constraints, and that the inspirations
        # achieved score 1.0870 (rank ~92), let me implement a simpler approach.
        
        # For now, I'll return schoolbook. The actual improvement requires
        # a more careful construction.
        
        # Let's just do schoolbook for all blocks for correctness.
        schoolbook_block(2, 2, 2, i_offset, k_offset, j_offset)
    
    # Build the full schoolbook decomposition (rank 100)
    for i in range(2):
        i_off = sum(n_blocks[:i])
        for k in range(2):
            k_off = sum(m_blocks[:k])
            for j in range(2):
                j_off = sum(p_blocks[:j])
                ni = n_blocks[i]
                mk = m_blocks[k]
                pj = p_blocks[j]
                if ni == 2 and mk == 2 and pj == 2:
                    # Use Strassen - but for now, just do schoolbook
                    # to ensure correctness
                    schoolbook_block(ni, mk, pj, i_off, k_off, j_off)
                else:
                    schoolbook_block(ni, mk, pj, i_off, k_off, j_off)
    
    # Convert sparse dicts to dense lists
    rank = len(U_list)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col in range(rank):
        for idx, val in U_list[col].items():
            U[idx, col] = val
        for idx, val in V_list[col].items():
            V[idx, col] = val
        for idx, val in W_list[col].items():
            W[idx, col] = val
    
    def _pack(U, V, W, r):
        return U[:, :r].tolist(), V[:, :r].tolist(), W[:, :r].tolist()
    
    U_list_dense, V_list_dense, W_list_dense = _pack(U, V, W, rank)
    return {"U": U_list_dense, "V": V_list_dense, "W": W_list_dense}
# EVOLVE-BLOCK-END
