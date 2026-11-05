"""Matrix multiplication 4x5x5 - Numerical Rank Minimization Strategy."""
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    schoolbook_rank = n * m * p  # 100
    
    # 1. Build Schoolbook Baseline (Guaranteed Valid)
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
                
    def verify_numeric(U, V, W, rank):
        """Check if decomposition is valid within numerical tolerance."""
        if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
            return False
        # Compute tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        # We check if T matches the identity tensor for matmul
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over rank dimension
                    # U row: i*m+j
                    # V row: j*p+k
                    # W row: k*n+i
                    val = np.dot(U[i*m+j, :], V[j*p+k, :])
                    # Wait, the tensor definition is:
                    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                    # My dot above is sum_r U_{i,j,r} V_{j,k,r}. This is NOT the tensor.
                    # The tensor is a 3-way sum.
                    # Let's compute properly.
                    pass
        
        # Efficient check:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is hard to vectorize simply without einsum or loops.
        # Given small dimensions, loops are fine for verification.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-5:
                        return False
        return True

    # Verify Schoolbook
    assert verify_numeric(U_sb, V_sb, W_sb, schoolbook_rank), "Schoolbook failed"
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = time.time() + 140  # Safety margin
    
    # 2. Numerical Search using Iterative Hard Thresholding (IHT)
    # We attempt to find a decomposition of rank R < 100.
    # We start with a random initialization of rank R and refine.
    
    # Rational Recovery: Given a numerical matrix, find the closest matrix with entries in {0, 0.5, 1, 1.5, ...}
    # Actually, the problem allows "integer/half-integer".
    # Let's try to recover to the set {0, 0.5, 1} first, or just round to nearest 0.5.
    
    def rationalize(mat, tol=1e-4):
        """Round to nearest 0.5 and check if it's close to the original."""
        # Scale by 2 to make integers
        scaled = mat * 2
        rounded = np.rint(scaled)
        # Check error
        if np.max(np.abs(scaled - rounded)) > tol * 10: # Heuristic tolerance
            return None
        return rounded / 2.0

    # IHT Algorithm for Tensor Rank Minimization
    # Objective: Minimize || T - sum_r U_r V_r W_r ||_F subject to rank R
    # We use a simple gradient-free or projected gradient approach?
    # Given the non-convexity, we'll use a "Rank-1 Update" style search or random restarts with local search.
    
    # A more robust simple approach: 
    # 1. Pick a target rank R.
    # 2. Initialize U, V, W randomly.
    # 3. Iterate: 
    #    a. Compute residual R = T - sum U_r V_r W_r
    #    b. Update U, V, W to minimize residual (e.g., alternating least squares is hard for 3-way with product constraint)
    #    c. Project to rank R (truncation)
    
    # Actually, for matrix multiplication, there is a known structure.
    # Let's try a different numerical method: 
    # "Tensor Rank-1 updates"
    
    def try_rank(R, num_restarts=50, time_limit=10):
        """Try to find a valid decomposition of rank R."""
        start_time = time.time()
        rng = np.random.default_rng(42)
        
        # Target Tensor T
        # T[i,j,k] = 1 if (i,j,k) is a valid index? No.
        # The tensor for matmul C = A B is:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # The tensor T has indices (i, j, k) corresponding to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
        # Wait, the decomposition is:
        # C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_k V_{j,k,r} B_{j,k}) ... No.
        
        # Standard decomposition form:
        # C_{i,k} = sum_r U_{i,r}^{(A)} V_{k,r}^{(B)} ... 
        # The task defines:
        # U shape (n*m, rank) -> U[i*m+j, r]
        # V shape (m*p, rank) -> V[j*p+k, r]
        # W shape (p*n, rank) -> W[k*n+i, r]
        
        # The tensor equation is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} ? No.
        # The tensor T_{i,j,k} for matrix multiplication is 1 if the indices are valid, but the value is 1 for all (i,j,k) in the product?
        # No, the tensor T represents the bilinear map.
        # T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # Because A_{i,j} * B_{j,k} contributes 1 * 1 = 1 to the sum for C_{i,k}.
        # So we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i,j,k.
        
        # This is a system of n*m*p = 100 equations.
        # We want to find U, V, W with rank R such that this holds.
        
        # Let's use a random search with local refinement.
        # Since R < 100, we have fewer variables than equations if R is small.
        # For R=99, we have 99*(20+25+20) = 5940 variables, 100 equations. Highly underdetermined.
        # So a solution likely exists for R close to 100.
        
        # Strategy: 
        # Start with schoolbook (rank 100).
        # Try to zero out one column and solve for adjustments in the others.
        # This is a linear system!
        
        # Let U, V, W be the schoolbook matrices.
        # We want to modify them to U', V', W' of rank 99.
        # This is equivalent to finding a vector in the null space of the tensor?
        
        # Actually, let's just use the linear algebra approach for Rank 99.
        # The condition is:
        # sum_r U'[i,j,r] V'[j,k,r] W'[k,i,r] = 1
        
        # If we take the schoolbook and remove column 99 (corresponding to i=3, j=4, k=4),
        # we lose the term U[3,4,99]V[4,4,99]W[4,3,99] = 1.
        # We need to add a correction delta_U, delta_V, delta_W (supported on the remaining 99 cols)
        # such that sum_{r<99} (U_r + dU_r)(V_r + dV_r)(W_r + dW_r) = 1.
        # This is non-linear.
        
        # However, if we assume small perturbations:
        # sum U V W + sum (dU V W + U dV W + U V dW) = 1
        # Since sum U V W = 1 - 1 = 0 (for the missing term) or 1 (for others),
        # For the missing term (i=3, j=4, k=4):
        # 0 + sum_{r<99} (dU_{3,4,r} V_{4,4,r} W_{4,3,r} + U_{3,4,r} dV_{4,4,r} W_{4,3,r} + U_{3,4,r} V_{4,4,r} dW_{4,3,r}) = 1
        
        # For all other terms (i,j,k) != (3,4,4):
        # 1 + sum_{r<99} (...) = 1  =>  sum_{r<99} (...) = 0
        
        # This is a linear system in the variables dU, dV, dW!
        # Number of variables: 99 * (20 + 25 + 20) = 5940.
        # Number of equations: 100.
        # We can find a solution easily using least squares (np.linalg.lstsq).
        
        # Let's implement this Linearization approach for Rank 99.
        
        def find_rank_99():
            # Remove last column of schoolbook
            U0 = U_sb[:, :99].copy()
            V0 = V_sb[:, :99].copy()
            W0 = W_sb[:, :99].copy()
            
            # Variables: dU (20x99), dV (25x99), dW (20x99)
            # We flatten them into a vector x of size 5940.
            # Equations: 100.
            # Matrix A of size 100 x 5940.
            # b of size 100.
            
            # For each equation (i,j,k):
            # LHS = sum_r (dU_{i,j,r} V0_{j,k,r} W0_{k,i,r} + U0_{i,j,r} dV_{j,k,r} W0_{k,i,r} + U0_{i,j,r} V0_{j,k,r} dW_{k,i,r})
            
            # Note: U0, V0, W0 are sparse (mostly 0s, one 1 per column).
            # This sparsity makes A very sparse.
            
            A = np.zeros((100, 5940))
            b = np.zeros(100)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i*m*p + j*p + k
                        # Target: 1 if (i,j,k) != (3,4,4), else 1 (since we removed the term that provided 1)
                        # Wait.
                        # Original sum = 1.
                        # New sum = sum_{r<99} (U0+dU)(V0+dV)(W0+dW).
                        # For (3,4,4): Original sum was 1 (from col 99). New base sum is 0.
                        # We need New sum = 1. So Linearized sum must be 1.
                        # For others: Original sum was 1 (from col r). New base sum is 1.
                        # We need New sum = 1. So Linearized sum must be 0.
                        
                        if i == 3 and j == 4 and k == 4:
                            b[eq_idx] = 1.0
                        else:
                            b[eq_idx] = 0.0
                        
                        # Fill A row
                        # Terms involving dU:
                        for r in range(99):
                            coeff = V0[j*p+k, r] * W0[k*n+i, r]
                            if coeff != 0:
                                # dU index: i*m+j, r
                                var_idx = (i*m+j)*99 + r
                                A[eq_idx, var_idx] += coeff
                            
                            coeff_v = U0[i*m+j, r] * W0[k*n+i, r]
                            if coeff_v != 0:
                                var_idx = 20*99 + (j*p+k)*99 + r
                                A[eq_idx, var_idx] += coeff_v
                                
                            coeff_w = U0[i*m+j, r] * V0[j*p+k, r]
                            if coeff_w != 0:
                                var_idx = 45*99 + (k*n+i)*99 + r
                                A[eq_idx, var_idx] += coeff_w
            
            # Solve A x = b using least squares
            # Since A is underdetermined, we want the minimum norm solution.
            # np.linalg.lstsq gives min norm.
            try:
                x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                return None
            
            # Check if the solution is "small" enough to justify the linear approximation
            # or if the resulting U, V, W are close to 0.5/1 integers.
            # Actually, we just need to check if the resulting decomposition is valid.
            
            dU = x[:20*99].reshape(20, 99)
            dV = x[20*99:45*99].reshape(25, 99)
            dW = x[45*99:].reshape(20, 99)
            
            U1 = U0 + dU
            V1 = V0 + dV
            W1 = W0 + dW
            
            # Verify
            if verify_numeric(U1, V1, W1, 99):
                # Try to rationalize
                # The values might be irrational.
                # The problem requires integer/half-integer entries.
                # If the linear solution is not close to half-integers, we can't use it directly.
                # However, if we find a valid numerical solution, we can try to snap to half-integers
                # and re-verify.
                
                U_r = rationalize(U1)
                V_r = rationalize(V1)
                W_r = rationalize(W1)
                
                if U_r is not None and V_r is not None and W_r is not None:
                    if verify_numeric(U_r, V_r, W_r, 99):
                        return U_r, V_r, W_r
                
                # If rationalization fails, the linear solution might be valid numerically
                # but not in the required format.
                # The task says "integer/half-integer entries".
                # If we can't snap, we can't return it.
                
            return None
        
        # Try Rank 99
        if time.time() < deadline - 10:
            res = find_rank_99()
            if res:
                U99, V99, W99 = res
                # Final check
                if verify_numeric(U99, V99, W99, 99):
                    return {"U": U99.tolist(), "V": V99.tolist(), "W": W99.tolist()}
        
        # If Rank 99 fails (e.g. due to rationalization), try Rank 98?
        # Rank 98 would require removing 2 columns. The linearization might break down.
        # Let's stick to Rank 99 as it's the most likely candidate for a "small" improvement.
        
        # Fallback: Schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
