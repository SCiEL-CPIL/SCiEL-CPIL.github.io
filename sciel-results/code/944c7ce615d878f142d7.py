"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100.0
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook solution
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank solution via random search
    # The tensor T_{ijk} = 1 if (i,j,k) is in the multiplication support
    # We need U (nm x r), V (mp x r), W (pn x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k (since we're multiplying 3x3 by 3x5)
    # Actually, the matrix multiplication tensor for C = A*B where A is n x m, B is m x p
    # has T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
    # For standard matrix mult, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # So it's the all-ones tensor of shape (n, m, p)
    
    # Wait, that's not right. The tensor is T[i,j,k] = 1 for all valid (i,j,k)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is all-ones.
    
    # For all-ones tensor, the rank-1 decomposition is trivial:
    # T[i,j,k] = 1 = 1*1*1, so rank 1!
    # But that can't be right for matrix multiplication...
    
    # Actually, the tensor for matrix multiplication is NOT all-ones in the sense that
    # we need U, V, W such that the product reconstructs the multiplication.
    # The condition is: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T[i,j,k] = 1 for all (i,j,k) means the tensor is the all-ones tensor.
    # But the rank of the all-ones tensor is 1!
    
    # This is confusing. Let me re-read the problem.
    # The problem says "matrix multiplication 3,3,5"
    # The standard matrix multiplication tensor has T[i,j,k] = 1 if the (i,k) output
    # depends on A[i,j]*B[j,k]. For standard multiplication, this is all (i,j,k).
    
    # But the rank of the all-ones tensor of size n*m*p is 1, which is clearly wrong
    # for matrix multiplication rank.
    
    # I think I'm misunderstanding the tensor structure. Let me think again.
    # The matrix multiplication tensor M^{n,m,p} is defined by:
    # M^{n,m,p} = sum_{i,j,k} e_i^{(n)} ⊗ e_j^{(m)} ⊗ e_k^{(p)}
    # where the tensor lives in (R^n ⊗ R^m) ⊗ R^p or similar.
    
    # Actually, the standard formulation is:
    # We want to compute C = AB where A is n x m and B is m x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T ∈ R^{n} ⊗ R^{m} ⊗ R^{p} with T[i,j,k] = 1.
    
    # The rank of this tensor is the matrix multiplication rank.
    # For the all-ones tensor, the rank is indeed not 1 in the tensor rank sense
    # because tensor rank is different from matrix rank.
    
    # Wait, no. If T[i,j,k] = 1 for all i,j,k, then T = u ⊗ v ⊗ w where
    # u_i = 1, v_j = 1, w_k = 1. So tensor rank is 1.
    
    # This must be wrong. Let me reconsider.
    
    # I think the issue is that the matrix multiplication tensor is NOT the all-ones tensor.
    # Let me look at this more carefully.
    
    # The matrix multiplication map is: (A, B) -> AB
    # Where A ∈ R^{n×m}, B ∈ R^{m×p}, AB ∈ R^{n×p}
    # The bilinear map is: f(A,B) = AB
    # The associated tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}
    # T[α, β, γ] where α indexes A, β indexes B, γ indexes AB
    # T[α, β, γ] = 1 if γ = (i,k) and α = (i,j), β = (j,k) for some j
    # Otherwise T[α, β, γ] = 0
    
    # So T is NOT all-ones. It's a specific tensor.
    
    # For n=3, m=3, p=5:
    # A has 9 entries, B has 15 entries, AB has 15 entries
    # T is a 9 x 15 x 15 tensor
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'
    # So T[(i,j), (j,k), (i,k)] = 1, and 0 otherwise
    
    # This is the standard matrix multiplication tensor.
    
    # For this tensor, the rank is the matrix multiplication rank.
    # For 2x2x2, it's 7 (Strassen).
    # For 3x3x3, it's at most 23 (Strassen-like), known to be <= 23.
    # For 3x3x5, it's unknown but likely around 40-45.
    
    # The schoolbook rank is n*m*p = 3*3*5 = 45.
    
    # Let me try a randomized approach to find a lower rank.
    
    def try_rank(r):
        if time.time() > deadline - 10:
            return None
        # Try random U, V and solve for W
        for attempt in range(50):
            if time.time() > deadline - 5:
                return None
            U = np.random.randint(-2, 3, size=(n*m, r)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, r)).astype(float)
            # Build the linear system for W
            # For each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i,j,k]
            # T[i,j,k] = 1 if this is a valid multiplication term
            # For standard matrix mult, all (i,j,k) with i<n, j<m, k<p are valid
            # So T[i,j,k] = 1 for all i,j,k
            
            # We have r unknowns W[k*n+i, c] for c in range(r)
            # But W has shape (p*n, r), so p*n*r unknowns
            # And we have n*m*p equations
            
            # This is underdetermined if r < n*m*p
            # Let's set up the system
            # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # Let's build the matrix A and vector b
            # A[eq, unknown] where eq = (i,j,k) index, unknown = (k*n+i, c)
            num_eqs = n * m * p
            num_unknowns = p * n * r
            
            A = np.zeros((num_eqs, num_unknowns))
            b = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq = (i * m + j) * p + k
                        for c in range(r):
                            unknown = (k * n + i) * r + c
                            A[eq, unknown] = U[i*m+j, c] * V[j*p+k, c]
            
            # Solve least squares
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if the solution is close to integer/half-integer
            W_flat = W_flat.reshape(p*n, r)
            
            # Try to round to half-integers
            W_rounded = np.round(W_flat * 2) / 2
            
            # Check if the rounded W gives an exact solution
            # Reconstruct and check
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U[i*m+j, c] * V[j*p+k, c] * W_rounded[k*n+i, c]
                        max_err = max(max_err, abs(val - 1))
            
            if max_err < 1e-9:
                # Found a valid solution
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_rounded.tolist()
                }
        
        return None
    
    # Try ranks from 40 down to 25
    for r in range(40, 24, -1):
        if time.time() > deadline - 5:
            break
        result = try_rank(r)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
