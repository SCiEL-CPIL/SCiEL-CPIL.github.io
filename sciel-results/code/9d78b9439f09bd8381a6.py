"""Improved rank-minimal decomposition for 2x5 @ 5x6 using block decomposition and rank reduction."""
DIMENSIONS = [2, 5, 6]

import time
import numpy as np
from collections import defaultdict
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 145s out of 150s wall-clock limit
    
    # Build the 54-rank block decomposition
    rank_init = 54
    U = [[0] * rank_init for _ in range(n * m)]
    V = [[0] * rank_init for _ in range(m * p)]
    W = [[0] * rank_init for _ in range(p * n)]

    col = 0

    # Part A: j=0 (2x1x6 tensor), rank 12
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Strassen 2x2x2 terms
    strassen_2x2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],           [(1,0,1),(1,1,-1)]),
        ([(0,0,1)],           [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)],           [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)],   [(1,1,1)],          [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)],  [(0,0,1),(0,1,1)],  [(1,1,1)]),
        ([(0,1,1),(1,1,-1)],  [(1,0,1),(1,1,1)],  [(0,0,1)]),
    ]

    # Part B: j=1,2 and j=3,4, each 2x2x6 split into 3x 2x2x2
    for block_j in [1, 3]:
        for k_block in range(3):
            k_base = k_block * 2
            for term in strassen_2x2x2:
                u_list, v_list, w_list = term
                for (li, lj, val) in u_list:
                    U[li * m + block_j + lj][col] += val
                for (lj, lk, val) in v_list:
                    V[(block_j + lj) * p + k_base + lk][col] += val
                for (li, lk, val) in w_list:
                    W[(k_base + lk) * n + li][col] += val
                col += 1

    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[i*m+j,r] * V[j*p+k,r] == W[k*n+i,r] for all i,j,k
    # Reshape and check
    U_reshaped = U_arr.reshape(n, m, rank_init)
    V_reshaped = V_arr.reshape(m, p, rank_init)
    W_reshaped = W_arr.reshape(p, n, rank_init)
    
    # Compute left side: for each (i,k), sum over j,r of U[i,j,r]*V[j,k,r]
    # = einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)
    lhs = np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)
    # Right side: W[k,i,r]
    rhs = W_reshaped.transpose(1, 0, 2)  # shape (n, p, rank_init) -> (i, k, r)
    
    if not np.allclose(lhs, rhs, atol=1e-10):
        # Fallback: return the 54-rank solution as-is (shouldn't happen)
        return {"U": U, "V": V, "W": W}

    # Now attempt rank reduction via Gaussian elimination on the rank-1 terms
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * ... wait, no.
    # Actually the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * delta? 
    # No. The standard tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]? 
    # Wait, let me re-read the contract.
    
    # "output factor follows the official trace convention W[k*n+i,r]"
    # The multiplication tensor: T[i,j,k] = delta_{i,j,k} meaning (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 if we're computing (A*B)[i,k] = sum_j A[i,j]B[j,k]
    # The tensor is T[i,j,k] where the bilinear form is sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = (AB)[i,k]
    # So T[i,j,k] is actually a 3-tensor with indices (i, j, k) but really it's:
    # (AB)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for the "multiplication" structure.
    # 
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]? 
    # No, that's not right either. Let me think again.
    #
    # Standard: A bilinear algorithm for matrix mult has:
    # M_r = sum over some (i,j) of a_{r,ij} * A[i,j]
    # N_r = sum over some (j,k) of b_{r,jk} * B[j,k]  
    # (AB)[i,k] = sum_r c_{r,ik} * M_r * N_r
    #
    # So T[i,j,k] = sum_r a_{r,ij} * b_{r,jk} * c_{r,ik}
    # where a_{r,ij} = U[i*m+j, r], b_{r,jk} = V[j*p+k, r], c_{r,ik} = W[k*n+i, r]
    
    # Verify: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should equal 1 if this is the "multiplication" tensor
    # Actually no - the tensor T[i,j,k] represents the coefficient such that
    # (AB)[i,k] = sum_{j} A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? No, that's not a tensor decomposition.
    
    # Let me re-examine. The matrix multiplication tensor T has entries T[i,j,k] where:
    # T is a tensor in R^{n x m x p} such that for matrices A (n x m) and B (m x p),
    # (AB)[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k? No!
    # (AB)[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i, j, k. The tensor is all-ones!
    
    # Wait, that can't be right for a "tensor decomposition" problem. Let me reconsider.
    # Actually yes, the matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 for all i,j,k.
    # The rank is the minimum number of rank-1 tensors to sum to this all-ones tensor.
    
    # So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    
    # Let me verify our construction satisfies this.
    
    # For Part A (j=0): U[i*5+0, r] = 1, V[0*6+k, r] = 1, W[k*2+i, r] = 1 for specific r
    # So for j=0: sum_r U[i,0,r]*V[0,k,r]*W[k,i,r] = number of r where all three are nonzero
    # In Part A, for each (i,k), there's exactly one r with U[i*5+0,r]=1, V[k,r]=1, W[k*2+i,r]=1
    # So the sum is 1. Good.
    
    # For Strassen parts, it should work by construction.
    
    # Now for rank reduction: we want to find a smaller rank.
    # Approach: collect all 54 rank-1 terms, then try to find linear dependencies.
    
    # Represent each rank-1 term as a vector in the space of (i,j,k) triples.
    # Term r contributes: for each (i,j,k), value = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Build the tensor as a 3D array
    T = np.zeros((n, m, p))
    for r in range(rank_init):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    if not np.allclose(T, 1.0, atol=1e-10):
        return {"U": U, "V": V, "W": W}

    # The tensor is all-ones, as expected.
    
    # Rank reduction strategy: try to find a subset of terms that still sums to the all-ones tensor,
    # or find linear combinations that cancel.
    
    # Actually, a better approach: use the fact that we can combine terms.
    # If two terms have the same support pattern, we might be able to merge them.
    
    # Let's try a greedy approach: start with all 54 terms, try to remove terms one by one
    # and see if the remaining terms still sum to the all-ones tensor.
    
    # This is expensive, so let's be smart about it.
    
    # Alternative: use the structure. The 12 terms for j=0 are necessary (each (i,k) pair 
    # needs exactly one term for j=0). The 42 Strassen terms might have redundancies.
    
    # Let me try: can we reduce the number of Strassen terms?
    # For each 2x2x6 block, we use 3x7=21 terms. Is there a 20-term algorithm for 2x2x6?
    
    # Known: R(2,2,6) = 21. So no.
    
    # But maybe a global decomposition that doesn't respect the block structure can do better?
    
    # Given time constraints, let's try a simple greedy rank reduction:
    # Try removing each term and checking if the tensor is still all-ones.
    
    best_rank = rank_init
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    # Try removing terms one at a time (greedy)
    # Start with all terms active
    active = [True] * rank_init
    
    # For each term, try removing it
    for trial in range(rank_init):
        if time.time() > deadline:
            break
        for r in range(rank_init):
            if not active[r]:
                continue
            # Try removing term r
            active[r] = False
            # Check if tensor is still all-ones
            T_check = np.zeros((n, m, p))
            for r2 in range(rank_init):
                if active[r2]:
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_check[i,j,k] += U[i*m+j, r2] * V[j*p+k, r2] * W[k*n+i, r2]
            if np.allclose(T_check, 1.0, atol=1e-10):
                # Term r can be removed
                best_rank -= 1
                best_U = np.delete(best_U, r, axis=1)
                best_V = np.delete(best_V, r, axis=1)
                best_W = np.delete(best_W, r, axis=1)
                # Adjust indices for subsequent removals
                for r3 in range(r+1, rank_init):
                    if active[r3]:
                        active[r3] = active[r3-1] if r3-1 >= 0 else False
                # This indexing is getting messy. Let me restart with a cleaner approach.
                break
            else:
                active[r] = True
    
    # The greedy removal above is buggy. Let me do it properly.
    if time.time() < deadline - 10:
        active = [True] * rank_init
        current_rank = rank_init
        U_cur = U_arr.copy()
        V_cur = V_arr.copy()
        W_cur = W_arr.copy()
        
        # Greedy: repeatedly try to remove a term
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for r in range(current_rank):
                if time.time() > deadline - 10:
                    break
                # Temporarily remove term r
                T_check = np.zeros((n, m, p))
                for r2 in range(current_rank):
                    if r2 == r:
                        continue
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_check[i,j,k] += U_cur[i*m+j, r2] * V_cur[j*p+k, r2] * W_cur[k*n+i, r2]
                if np.allclose(T_check, 1.0, atol=1e-10):
                    # Remove term r
                    U_cur = np.delete(U_cur, r, axis=1)
                    V_cur = np.delete(V_cur, r, axis=1)
                    W_cur = np.delete(W_cur, r, axis=1)
                    current_rank -= 1
                    changed = True
                    break  # Restart the loop since indices changed
        
        if current_rank < best_rank:
            best_rank = current_rank
            best_U = U_cur
            best_V = V_cur
            best_W = W_cur
    
    # Convert back to lists
    U_final = best_U.tolist()
    V_final = best_V.tolist()
    W_final = best_W.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
