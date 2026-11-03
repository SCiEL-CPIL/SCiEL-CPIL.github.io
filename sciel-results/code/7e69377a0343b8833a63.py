"""Rewritten program for 3x3x5 matrix multiplication rank minimization."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin before 150s kill

    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15

    def schoolbook():
        """Return the rank-nmp schoolbook decomposition (rank 45)."""
        r = n * m * p
        U = [[0] * r for _ in range(nm)]
        V = [[0] * r for _ in range(mp)]
        W = [[0] * r for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, atol=1e-9):
        """Verify that the decomposition exactly reconstructs the tensor."""
        r = len(U[0]) if U else 0
        T = np.zeros((n, m, p))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        return np.allclose(T, np.ones((n, m, p)), atol=atol)

    def strassen_2x2x2_terms():
        """Return the 7 rank-1 tensor terms for 2x2x2 matrix multiplication.
        Each term is (U_2x2, V_2x2, W_2x2) such that:
        T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0,1}
        """
        # Correct Strassen 2x2x2 decomposition
        # U, V, W are 2x2 matrices for each term
        terms = [
            # c=0
            (np.array([[1, 0], [0, -1]], dtype=float),
             np.array([[0, 0], [1, 1]], dtype=float),
             np.array([[0, 1], [0, 0]], dtype=float)),
            # c=1
            (np.array([[0, 0], [1, 1]], dtype=float),
             np.array([[1, 0], [0, 0]], dtype=float),
             np.array([[1, 0], [0, 0]], dtype=float)),
            # c=2
            (np.array([[1, 0], [0, 0]], dtype=float),
             np.array([[0, 1], [0, -1]], dtype=float),
             np.array([[0, 0], [1, 0]], dtype=float)),
            # c=3
            (np.array([[0, 0], [0, 1]], dtype=float),
             np.array([[-1, 0], [1, 0]], dtype=float),
             np.array([[1, 0], [0, -1]], dtype=float)),
            # c=4
            (np.array([[1, 0], [0, 1]], dtype=float),
             np.array([[1, 0], [0, 1]], dtype=float),
             np.array([[1, 0], [1, 1]], dtype=float)),
            # c=5
            (np.array([[0, 1], [0, -1]], dtype=float),
             np.array([[0, 0], [1, 1]], dtype=float),
             np.array([[0, 0], [0, 1]], dtype=float)),
            # c=6
            (np.array([[0, 0], [1, -1]], dtype=float),
             np.array([[1, 1], [0, 0]], dtype=float),
             np.array([[1, 0], [0, 0]], dtype=float)),
        ]
        
        # Verify the decomposition
        U_arr = np.zeros((2, 2, 7))
        V_arr = np.zeros((2, 2, 7))
        W_arr = np.zeros((2, 2, 7))
        for c, (u, v, w) in enumerate(terms):
            U_arr[:, :, c] = u
            V_arr[:, :, c] = v
            W_arr[:, :, c] = w
        
        T_check = np.zeros((2, 2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for c in range(7):
                        T_check[i, j, k] += U_arr[i, j, c] * V_arr[j, k, c] * W_arr[k, i, c]
        
        if not np.allclose(T_check, np.ones((2, 2, 2)), atol=1e-9):
            return None
        
        return terms

    def build_strassen_decomposition():
        """Build a decomposition using Strassen for 2x2 blocks and schoolbook for the rest."""
        strassen_terms = strassen_2x2x2_terms()
        if strassen_terms is None:
            return None
        
        # We'll use Strassen for two disjoint 2x2x2 blocks
        # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
        # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
        
        # Collect all terms as (U_dict, V_dict, W_dict)
        all_terms = []
        current_col = 0
        
        # Helper to add a Strassen block with offset
        def add_strassen_block(i_offset, j_offset, k_offset, start_col):
            terms_added = []
            for c, (u, v, w) in enumerate(strassen_terms):
                U_dict = {}
                V_dict = {}
                W_dict = {}
                
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = i_offset + i_local
                        j_global = j_offset + j_local
                        U_dict[(i_global * m + j_global, start_col + c)] = u[i_local, j_local]
                
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = j_offset + j_local
                        k_global = k_offset + k_local
                        V_dict[(j_global * p + k_global, start_col + c)] = v[j_local, k_local]
                
                for k_local in range(2):
                    for i_local in range(2):
                        k_global = k_offset + k_local
                        i_global = i_offset + i_local
                        W_dict[(k_global * n + i_global, start_col + c)] = w[k_local, i_local]
                
                terms_added.append((U_dict, V_dict, W_dict))
            return terms_added, start_col + 7
        
        # Add Block 1
        terms1, next_col = add_strassen_block(0, 0, 0, 0)
        all_terms.extend(terms1)
        
        # Add Block 2
        terms2, next_col = add_strassen_block(0, 0, 2, next_col)
        all_terms.extend(terms2)
        
        # Schoolbook for remaining entries
        # Covered by Strassen: 
        # Block 1: i<2, j<2, k<2
        # Block 2: i<2, j<2, 2<=k<4
        remaining = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not ((i < 2 and j < 2 and k < 2) or (i < 2 and j < 2 and 2 <= k < 4)):
                        remaining.append((i, j, k))
        
        for idx, (i, j, k) in enumerate(remaining):
            col = next_col + idx
            U_dict = {(i * m + j, col): 1}
            V_dict = {(j * p + k, col): 1}
            W_dict = {(k * n + i, col): 1}
            all_terms.append((U_dict, V_dict, W_dict))
        
        # Combine all terms
        rank = next_col + len(remaining)
        
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        for U_dict, V_dict, W_dict in all_terms:
            for (row, col), val in U_dict.items():
                if val != 0:
                    U[row][col] = val
            for (row, col), val in V_dict.items():
                if val != 0:
                    V[row][col] = val
            for (row, col), val in W_dict.items():
                if val != 0:
                    W[row][col] = val
        
        return U, V, W, rank

    def try_rank_search(min_rank, max_rank):
        """Try to find a decomposition of the given rank using random search."""
        best_result = None
        
        for rank in range(min_rank, max_rank + 1):
            if time.time() > deadline - 10:
                break
            
            # Try multiple random initializations
            for attempt in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Random initialization with small values
                U = np.random.randn(nm, rank) * 0.1
                V = np.random.randn(mp, rank) * 0.1
                W = np.random.randn(pn, rank) * 0.1
                
                # Gradient descent to minimize ||T_decomposed - T_target||^2
                T_target = np.ones((n, m, p))
                
                for step in range(100):
                    if time.time() > deadline - 10:
                        break
                    
                    # Compute gradient
                    T_decomp = np.zeros((n, m, p))
                    dU = np.zeros((nm, rank))
                    dV = np.zeros((mp, rank))
                    dW = np.zeros((pn, rank))
                    
                    for c in range(rank):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                                    T_decomp[i, j, k] += val
                                    err = T_decomp[i, j, k] - T_target[i, j, k]
                                    dU[i * m + j, c] += err * V[j * p + k, c] * W[k * n + i, c]
                                    dV[j * p + k, c] += err * U[i * m + j, c] * W[k * n + i, c]
                                    dW[k * n + i, c] += err * U[i * m + j, c] * V[j * p + k, c]
                    
                    # Update
                    lr = 0.01
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                    
                    # Check convergence
                    if np.allclose(T_decomp, T_target, atol=1e-6):
                        # Convert to lists
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        if verify_decomposition(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                
                # Final check
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if verify_decomposition(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # First, try the Strassen + schoolbook hybrid (should give rank ~38)
    result = build_strassen_decomposition()
    if result is not None:
        U, V, W, rank = result
        if verify_decomposition(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Try random search for lower ranks
    # Known lower bound for 3x3x5 is around 20-25
    result = try_rank_search(20, 35)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
