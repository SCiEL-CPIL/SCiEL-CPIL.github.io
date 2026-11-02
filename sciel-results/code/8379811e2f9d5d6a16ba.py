"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # First, build the known rank-28 Strassen decomposition as fallback
    def strassen_decomposition():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(u_idx, v_idx, w_idx, col_offset):
            U[u_idx[0]][col_offset + 0] += 1
            U[u_idx[3]][col_offset + 0] += 1
            V[v_idx[0]][col_offset + 0] += 1
            V[v_idx[3]][col_offset + 0] += 1
            W[w_idx[0]][col_offset + 0] += 1
            W[w_idx[3]][col_offset + 0] += 1
            
            U[u_idx[2]][col_offset + 1] += 1
            U[u_idx[3]][col_offset + 1] += 1
            V[v_idx[0]][col_offset + 1] += 1
            W[w_idx[2]][col_offset + 1] += 1
            W[w_idx[3]][col_offset + 1] -= 1
            
            U[u_idx[0]][col_offset + 2] += 1
            V[v_idx[1]][col_offset + 2] += 1
            V[v_idx[3]][col_offset + 2] -= 1
            W[w_idx[1]][col_offset + 2] += 1
            W[w_idx[3]][col_offset + 2] += 1
            
            U[u_idx[3]][col_offset + 3] += 1
            V[v_idx[2]][col_offset + 3] += 1
            V[v_idx[0]][col_offset + 3] -= 1
            W[w_idx[0]][col_offset + 3] += 1
            W[w_idx[2]][col_offset + 3] += 1
            
            U[u_idx[0]][col_offset + 4] += 1
            U[u_idx[1]][col_offset + 4] += 1
            V[v_idx[3]][col_offset + 4] += 1
            W[w_idx[0]][col_offset + 4] -= 1
            W[w_idx[1]][col_offset + 4] += 1
            
            U[u_idx[2]][col_offset + 5] += 1
            U[u_idx[0]][col_offset + 5] -= 1
            V[v_idx[0]][col_offset + 5] += 1
            V[v_idx[1]][col_offset + 5] += 1
            W[w_idx[3]][col_offset + 5] += 1
            
            U[u_idx[1]][col_offset + 6] += 1
            U[u_idx[3]][col_offset + 6] -= 1
            V[v_idx[2]][col_offset + 6] += 1
            V[v_idx[3]][col_offset + 6] += 1
            W[w_idx[0]][col_offset + 6] += 1
        
        u1 = [0, 1, 4, 5]
        v1 = [0, 1, 4, 5]
        w1 = [0, 2, 1, 3]
        add_strassen_block(u1, v1, w1, 0)
        
        u2 = [2, 3, 6, 7]
        v2 = [8, 9, 12, 13]
        w2 = [0, 2, 1, 3]
        add_strassen_block(u2, v2, w2, 7)
        
        u3 = [0, 1, 4, 5]
        v3 = [2, 3, 6, 7]
        w3 = [4, 6, 5, 7]
        add_strassen_block(u3, v3, w3, 14)
        
        u4 = [2, 3, 6, 7]
        v4 = [10, 11, 14, 15]
        w4 = [4, 6, 5, 7]
        add_strassen_block(u4, v4, w4, 21)
        
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The tensor T[i,j,k,l] = delta_{j,k} for C[i,l] = sum_j A[i,j] * B[j,l]
        # Decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # Indexing: U[i*m+j, r], V[j*p+l, r], W[l*n+i, r]
        
        # Check: sum_r U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r] == delta_{j,k}... 
        # Actually the matrix multiplication tensor is T[i,j,k,l] = 1 if j==k else 0
        # where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
        # But our decomposition is: C[i,l] = sum_r W[l*n+i, r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_k V[k*p+l,r]*B[k,l])
        # This should equal sum_jk A[i,j]*B[j,l]*delta_{j,k} = sum_j A[i,j]*B[j,l]
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r] = delta_{j,?}
        # Wait, let me reconsider. The standard formulation:
        # C[i,l] = sum_j A[i,j] B[j,l]
        # = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_k V[k*p+l, r] B[k,l]) W[l*n+i, r]
        # For this to equal sum_j A[i,j] B[j,l], we need:
        # sum_r U[i*m+j, r] V[k*p+l, r] W[l*n+i, r] = delta_{j,k}
        
        # Test with random matrices
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute via decomposition
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            a_lin = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    a_lin[i] += U_arr[i*m + j, r] * A[i, j]
            b_lin = np.zeros(p)
            for l in range(p):
                for k in range(m):
                    b_lin[l] += V_arr[k*p + l, r] * B[k, l]
            for i in range(n):
                for l in range(p):
                    C_decomp[i, l] += W_arr[l*n + i, r] * a_lin[i] * b_lin[l]
        
        if not np.allclose(C_decomp, C_expected, atol=tol):
            return False
        return True
    
    # Start with Strassen rank-28 as baseline
    best_result = strassen_decomposition()
    best_rank = 28
    
    # Try to find lower rank decompositions by coupling Strassen blocks
    # Strategy: For blocks that share A inputs, try to merge their Strassen decompositions
    
    # The four blocks are:
    # Block 1: A[0:2,0:2] x B[0:2,0:2] -> C[0:2,0:2]
    # Block 2: A[0:2,2:4] x B[2:4,0:2] -> C[0:2,0:2]
    # Block 3: A[0:2,0:2] x B[0:2,2:4] -> C[0:2,2:4]
    # Block 4: A[0:2,2:4] x B[2:4,2:4] -> C[0:2,2:4]
    
    # Blocks 1&3 share A[0:2,0:2], Blocks 2&4 share A[0:2,2:4]
    # Blocks 1&2 share C[0:2,0:2], Blocks 3&4 share C[0:2,2:4]
    
    # Try a coupled approach: treat C[0:2,0:4] = A[0:2,0:4] x B[0:4,0:2]
    # This is a 2x4 x 4x2 multiplication, which has schoolbook rank 16
    # And C[0:2,2:4] = A[0:2,0:4] x B[0:4,2:4], also 2x4 x 4x2
    
    # For 2x4 x 4x2, can we do better than schoolbook 16?
    # Strassen on 2x2 blocks: A = [A11 A12], B = [B11; B21] (4x2)
    # C = A11*B11 + A12*B21 (sum of two 2x2 products)
    # Rank of sum of two 2x2 products: at most 14, possibly less
    
    # Let's try to find a rank-r decomposition for 2x4 x 4x2
    # U: (2*4, r) = (8, r), V: (4*2, r) = (8, r), W: (2*2, r) = (4, r)
    
    def find_low_rank_2x4x4x2(max_rank=14, num_trials=500):
        """Try to find a low-rank decomposition for 2x4 x 4x2 matrix mult."""
        nn, mm, pp = 2, 4, 2  # n=2, m=4, p=2
        for r in range(7, max_rank + 1):  # Start from Strassen-like 7
            if time.time() > deadline - 10:
                break
            # Try random integer/half-integer entries
            for trial in range(num_trials):
                if time.time() > deadline - 5:
                    break
                # Generate random sparse patterns
                U_trial = np.zeros((nn * mm, r), dtype=float)
                V_trial = np.zeros((mm * pp, r), dtype=float)
                W_trial = np.zeros((pp * nn, r), dtype=float)
                
                # Use a structured approach based on Strassen-like ideas
                # For 2x4 x 4x2: A is 2x4, B is 4x2, C is 2x2
                # C = A[:,0:2] @ B[0:2,:] + A[:,2:4] @ B[2:4,:]
                
                # Try to use 7 multiplications by coupling
                # This is equivalent to computing two 2x2 products with shared structure
                
                if r == 7:
                    # Try a specific coupled Strassen-like decomposition
                    # A = [a00 a01 | a02 a03]
                    #     [a10 a11 | a12 a13]
                    # B = [b00 b01]
                    #     [b10 b11]
                    #     [b20 b21]
                    #     [b30 b31]
                    # C = A[:,:2]@B[:2,:] + A[:,2:]@B[2:4,:]
                    #   = [a00 a01; a10 a11]@[b00 b01; b10 b11] + [a02 a03; a12 a13]@[b20 b21; b30 b31]
                    
                    # Try to find 7 bilinear forms
                    # This is a research problem; let's try a few structured patterns
                    
                    # Pattern 1: Direct Strassen on each 2x2 block, but share some forms
                    # M1 = (a00+a11)(b00+b10)  -- but b10 is from different block
                    # This doesn't directly work.
                    
                    pass  # Skip for now, try other approaches
                
                # Generate random half-integer entries
                scale = 1.0
                for r_idx in range(r):
                    # U column
                    nnz_u = np.random.randint(1, 4)
                    rows_u = np.random.choice(nn * mm, nnz_u, replace=False)
                    for row in rows_u:
                        U_trial[row, r_idx] = np.random.choice([-1, 1]) * scale
                    
                    # V column  
                    nnz_v = np.random.randint(1, 4)
                    rows_v = np.random.choice(mm * pp, nnz_v, replace=False)
                    for row in rows_v:
                        V_trial[row, r_idx] = np.random.choice([-1, 1]) * scale
                    
                    # W column
                    nnz_w = np.random.randint(1, 3)
                    rows_w = np.random.choice(pp * nn, nnz_w, replace=False)
                    for row in rows_w:
                        W_trial[row, r_idx] = np.random.choice([-1, 1]) * scale
                
                # Verify
                A_test = np.random.randn(nn, mm)
                B_test = np.random.randn(mm, pp)
                C_expected = A_test @ B_test
                
                C_decomp = np.zeros((nn, pp))
                for r_idx in range(r):
                    a_lin = np.zeros(nn)
                    for i in range(nn):
                        for j in range(mm):
                            a_lin[i] += U_trial[i*mm + j, r_idx] * A_test[i, j]
                    b_lin = np.zeros(pp)
                    for l in range(pp):
                        for k in range(mm):
                            b_lin[l] += V_trial[k*pp + l, r_idx] * B_test[k, l]
                    for i in range(nn):
                        for l in range(pp):
                            C_decomp[i, l] += W_trial[l*nn + i, r_idx] * a_lin[i] * b_lin[l]
                
                if np.allclose(C_decomp, C_expected, atol=1e-10):
                    return r, U_trial, V_trial, W_trial
        
        return None
    
    # Try to find lower-rank decomposition for the full problem
    # by decomposing into two 2x4x4x2 subproblems and coupling them
    
    # For now, let's try a more systematic search on the full problem
    # with a time budget
    
    search_start = time.time()
    search_budget = 100  # seconds
    
    # Try ranks from 24 to 27 (below current best of 28)
    for target_rank in range(24, min(28, 29)):
        if time.time() > deadline - 10:
            break
        
        # For each rank, try a few structured approaches
        # Approach: Start from Strassen and try to merge blocks
        
        if target_rank <= 27:
            # Try to merge blocks 1 and 3 (which share A inputs)
            # Combined: A[0:2,0:2] x [B[0:2,0:2] | B[0:2,2:4]] -> C[0:2,0:4]
            # This is 2x2 x 2x4 -> 2x4, schoolbook rank 8, Strassen can do 7
            
            # If we can do 2x2 x 2x4 in rank 7, and 2x2 x 2x4 in rank 7 for the other half,
            # total would be 14, but we need to handle the B[2:4,:] parts too.
            
            # Actually, let me reconsider the structure:
            # C = A @ B where A is 2x4, B is 4x4
            # Partition A = [A_L | A_R] (each 2x2), B = [[B_TL, B_TR], [B_BL, B_BR]] (each 2x2)
            # C = A_L @ [B_TL; B_BL] + A_R @ [B_TR; B_BR]
            #   = A_L @ B_TL + A_L @ B_BL + A_R @ B_TR + A_R @ B_BR
            # Wait, that's not right. Let me redo:
            # A is 2x4: A = [a00 a01 a02 a03; a10 a11 a12 a13]
            # B is 4x4: B = [b00 b01 b02 b03; b10 b11 b12 b13; b20 b21 b22 b23; b30 b31 b32 b33]
            # C[i,l] = sum_{j=0}^{3} A[i,j] B[j,l]
            
            # C[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
            # C[0,1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
            # C[0,2] = a00*b02 + a01*b12 + a02*b22 + a03*b32
            # C[0,3] = a00*b03 + a01*b13 + a02*b23 + a03*b33
            # C[1,0] = a10*b00 + a11*b10 + a12*b20 + a13*b30
            # C[1,1] = a10*b01 + a11*b11 + a12*b21 + a13*b31
            # C[1,2] = a10*b02 + a11*b12 + a12*b22 + a13*b32
            # C[1,3] = a10*b03 + a11*b13 + a12*b23 + a13*b33
            
            # The Strassen approach partitions into 2x2 blocks and uses 7 mults per block.
            # To do better, we need to share multiplications across blocks.
            
            # One idea: Use the fact that C[0,:] and C[1,:] share the same A rows.
            # For row 0 of C: C[0,l] = sum_j a0j * B[j,l]
            # For row 1 of C: C[1,l] = sum_j a1j * B[j,l]
            
            # These are independent linear combinations of B's columns, so sharing
            # across rows doesn't help directly.
            
            # Sharing across columns of B: C[:,0:2] and C[:,2:4] both use all of A.
            # C[:,0:2] = A @ B[:,0:2] (2x4 @ 4x2)
            # C[:,2:4] = A @ B[:,2:4] (2x4 @ 4x2)
            
            # If we can compute 2x4 @ 4x2 in rank r, then total rank is 2r.
            # Schoolbook for 2x4@4x2 is 16, so 2*16=32 (matches schoolbook 32).
            # Strassen gives 2*14=28 (if 2x4@4x2 can be done in 14).
            # To beat 28, we need 2x4@4x2 in rank < 14, i.e., <= 13.
            
            # Can 2x4 @ 4x2 be done in rank 13 or less?
            # The tensor for 2x4@4x2 has dimension 2*4*4*2 = 64.
            # Schoolbook rank is 2*4*2 = 16.
            # Strassen: partition A into two 2x2 blocks, B into two 2x2 blocks.
            # C = A1@B1 + A2@B2 where A1,A2 are 2x2, B1,B2 are 2x2.
            # Sum of two 2x2 products: rank at most 14, possibly 13 or less.
            
            # Let's try to find a rank-13 or lower decomposition for 2x4@4x2.
            # This is a well-studied problem. The rank of the sum of two 2x2 matrix
            # multiplication tensors is known to be 14 (same as two independent Strassen).
            # Actually, I'm not sure it can be done in less than 14.
            
            # Let's try a computational search for rank 13 decomposition of 2x4@4x2.
            pass
    
    # Given the difficulty of finding a lower rank, let's try a more targeted search
    # for the full 2x4@4x4 problem with ranks 24-27
    
    # Use a local search / perturbation approach starting from Strassen
    # Try to eliminate some rank terms by finding linear dependencies
    
    # For now, return the Strassen rank-28 solution as the best known
    # A more sophisticated search would be needed to find rank < 28
    
    return best_result
# EVOLVE-BLOCK-END
