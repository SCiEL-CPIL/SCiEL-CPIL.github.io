import time
import numpy as np
import itertools

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        r = len(U[0]) if U else 0
        if not U or not V or not W:
            return False
        if len(V[0]) != r or len(W[0]) != r:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def try_rank(rank):
        """
        Attempts to find a decomposition of the given rank by searching 
        for a valid sparsity pattern for U and V, then solving for W.
        We restrict U and V entries to {-1, 0, 1} and W to integers/half-integers.
        """
        if rank < 20: # Lower bound heuristic
            return None
            
        # We will use a randomized local search on the sparsity masks of U and V.
        # We fix the values to be random +/- 1 where the mask is 1.
        # Then we solve for W. If W is close to integers/half-integers, we accept.
        
        # To make it faster, we will try a few different "densities" and restarts.
        # The key insight is that for small ranks, the tensors must be sparse.
        
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            # Random initialization of U and V
            # We want U to be 6xrank, V to be 15xrank
            # Let's bias towards sparse matrices
            density_u = 0.3
            density_v = 0.4
            
            U = np.zeros((6, rank), dtype=int)
            V = np.zeros((15, rank), dtype=int)
            
            # Generate random masks
            mask_u = np.random.rand(6, rank) < density_u
            mask_v = np.random.rand(15, rank) < density_v
            
            # Assign random +/- 1
            U_vals = np.random.choice([-1, 1], size=(6, rank))
            V_vals = np.random.choice([-1, 1], size=(15, rank))
            
            U[mask_u] = U_vals[mask_u]
            V[mask_v] = V_vals[mask_v]
            
            # Construct the linear system for W
            # The tensor T_{ijk} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
            # We want T_{ijk} = 1 for all i,j,k (since it's matrix multiplication identity tensor? 
            # No, wait. The task is to decompose the matrix multiplication tensor.
            # The tensor M_{i,j,k} for matrix multiplication A(i,j) B(j,k) -> C(i,k) 
            # is usually defined such that C_{ik} = sum_j A_{ij} B_{jk}.
            # The "tensor" of multiplication is the set of coefficients.
            # Actually, the standard tensor decomposition of matrix multiplication 
            # represents the operation C = AB as a sum of rank-1 tensors.
            # The entries of the multiplication tensor are 1 if the indices match the 
            # multiplication pattern, 0 otherwise.
            # Specifically, for multiplying n x m by m x p:
            # The tensor T has dimensions n x m x p? No, the factors are U (nm x r), V (mp x r), W (pn x r).
            # The product sum_c U_{(i,j),c} V_{(j,k),c} W_{(k,i),c} should equal 1 
            # if (i,j,k) corresponds to a term in the sum?
            # Let's check the schoolbook.
            # Schoolbook sets U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1.
            # Sum over col: for a specific (i,j,k), only the col where 
            # U row is i*m+j, V row is j*p+k, W row is k*n+i contributes.
            # In schoolbook, each col is unique to a (i,j,k) triple.
            # So for the triple (i,j,k), the sum is 1.
            # For any other triple, the sum is 0.
            # So the target tensor T_{i,j,k} is 1 for all i in [0,n), j in [0,m), k in [0,p)?
            # No, that's not right.
            # The matrix multiplication tensor T_{ijk} is 1 if the indices are valid 
            # in the contraction?
            # Actually, the tensor of matrix multiplication M_{n,m,p} is defined 
            # on indices (i, j, k) where i is row of A, j is col of A/row of B, k is col of B.
            # The entry T_{ijk} is 1.
            # Wait, if T_{ijk} = 1 for all i,j,k, then the decomposition is trivial?
            # No, the indices are coupled.
            # Let's look at the verify function in the seed.
            # It checks: s = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            # And requires s == 1 for ALL i, j, k.
            # This implies the target tensor is the all-ones tensor of size n x m x p?
            # If so, the rank is just the rank of the all-ones tensor, which is 1?
            # No, that can't be. Matrix multiplication is not just summing all products.
            # C_{ik} = sum_j A_{ij} B_{jk}.
            # This is a bilinear form.
            # The tensor decomposition represents the bilinear map.
            # The "tensor" is the set of coefficients of the monomials A_{ij} B_{jk} in the expressions for C_{ik}.
            # There is exactly one monomial A_{ij} B_{jk} for each j.
            # So for a fixed i, k, the coefficient of A_{ij} B_{jk} is 1.
            # The tensor T has indices (i, j, k) corresponding to the term A_{ij} B_{jk}.
            # The value is 1.
            # So yes, the target is T_{ijk} = 1 for all i, j, k.
            # BUT, the indices i, j, k are not independent in the output W.
            # W is indexed by (k, i).
            # The term U_{ij,c} V_{jk,c} W_{ki,c} contributes to the coefficient of A_{ij} B_{jk} in C_{ki}?
            # No, C is indexed by (i, k) in standard math, but here W is (p*n, rank) i.e. (k, i).
            # So the output is C_{ki}.
            # The sum is over c.
            # For a specific term A_{ij} B_{jk}, it appears in the computation of C_{ki} 
            # with coefficient sum_c U_{ij,c} V_{jk,c} W_{ki,c}.
            # We want this coefficient to be 1.
            # We also want the coefficient of A_{ij'} B_{jk} in C_{ki} to be 0 for j' != j.
            # And coefficient of A_{ij} B_{jk'} in C_{ki} to be 0 for k' != k.
            # And coefficient of A_{ij} B_{jk} in C_{k'i'} to be 0 for (k',i') != (k,i).
            
            # So the condition is:
            # sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1 if (i,j,k) is the specific triple?
            # No, the triple (i,j,k) defines the input pair. The output index is (k,i).
            # For a fixed output index (k,i), the coefficient of input pair (j) is 1.
            # So for a fixed i, k, and varying j:
            # sum_c U_{ij,c} V_{jk,c} W_{ki,c} should be 1 for the specific j?
            # No, C_{ki} = sum_j A_{ij} B_{jk}.
            # So for a fixed i, k, the term A_{ij} B_{jk} should have coefficient 1.
            # The term A_{ij'} B_{jk} (with different j') should have coefficient 0.
            # So:
            # For all i, k, j:
            # sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1
            # Is this correct?
            # If i=0, k=0, j=0: sum U_{00,c} V_{00,c} W_{00,c} = 1.
            # If i=0, k=0, j=1: sum U_{01,c} V_{10,c} W_{00,c} = 0.
            # If i=0, k=1, j=0: sum U_{00,c} V_{01,c} W_{10,c} = 0.
            
            # Let's re-read the verify function in the seed.
            # for i in range(n):
            #   for j in range(m):
            #     for k in range(p):
            #       s = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            #       if s != 1: return False
            # This checks if the sum is 1 for ALL i, j, k.
            # This implies that for EVERY combination of i, j, k, the coefficient is 1.
            # This would mean C_{ki} = sum_j A_{ij} B_{jk} + sum_{j' != j} A_{ij'} B_{jk} ... ?
            # No, if the sum is 1 for all i,j,k, then:
            # For fixed i, k, the sum over j of (coefficient of A_{ij}B_{jk}) is not constrained.
            # The constraint is on the individual terms.
            # If sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1 for ALL i,j,k,
            # then C_{ki} = sum_j A_{ij} B_{jk} * 1 = sum_j A_{ij} B_{jk}.
            # This IS correct matrix multiplication!
            # The "tensor" is indeed the all-ones tensor on the indices (i,j,k) 
            # when viewed as the coefficient of the monomial A_{ij}B_{jk} in the expression for C_{ki}.
            # Wait, if i=0, k=0, j=0, coeff is 1.
            # If i=0, k=0, j=1, coeff is 1.
            # Then C_{00} = A_{00}B_{00} + A_{01}B_{10} + ...
            # This is exactly C_{00}.
            # So yes, the target is T_{ijk} = 1 for all i,j,k.
            
            # So we need to find U, V, W such that:
            # sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1 for all i,j,k.
            
            # This is a system of n*m*p = 30 equations.
            # Unknowns: W_{ki,c} (10 * rank).
            # U and V are fixed in this iteration.
            # We solve for W.
            
            A_mat = np.zeros((30, rank * 10))
            b = np.ones(30)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i*m+j, :].astype(float)
                        v_row = V[j*p+k, :].astype(float)
                        coeffs = u_row * v_row
                        # W is indexed by (k, i). 
                        # W_flat index for (k, i) and column c is (k*n + i)*rank + c.
                        # The equation is sum_c coeffs[c] * W[(k*n+i)*rank + c] = 1
                        start_idx = (k*n+i) * rank
                        A_mat[idx, start_idx:start_idx+rank] = coeffs
                        idx += 1
            
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b, rcond=None)
            except:
                continue
                
            W_sol = x.reshape(10, rank)
            
            # Check if W is close to integers or half-integers
            # Try rounding to integers
            W_int = np.round(W_sol)
            if np.allclose(W_int, W_sol, atol=1e-4):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_int.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
            # Try rounding to half-integers
            W_half = np.round(W_sol * 2) / 2
            if np.allclose(W_half, W_sol, atol=1e-4):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_half.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
            # Local Search: flip a bit in U or V and retry
            # This is expensive, so we only do a few steps
            # We will skip local search for now to keep it fast, 
            # but rely on many random restarts with different densities.
            
        return None

    # Try ranks from low to high
    # Known bounds: Rank(2x3x5) is at most 24? Or 25?
    # Strassen on 2x2 part (rank 7) + 1x1 part (rank 5) + cross terms?
    # Let's try from 20 up to 29.
    
    for r in range(20, 30):
        if time.time() > deadline:
            break
        res = try_rank(r)
        if res:
            return res

    return schoolbook()
