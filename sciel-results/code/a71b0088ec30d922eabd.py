import scipy
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = 145  # seconds, leave margin for safety

    # 1. Build the Target Tensor T (2x3 * 3x3)
    # T[a, b, c] = 1 if (a,b,c) corresponds to a valid multiplication term
    # Indices: a in [0, nm), b in [0, mp), c in [0, pn)
    # a = i*m + j (i in 0..n-1, j in 0..m-1)
    # b = j*p + k (j in 0..m-1, k in 0..p-1)
    # c = k*n + i (k in 0..p-1, i in 0..n-1)
    # Note: The middle index j must match in a and b.
    
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # 2. Define a known low-rank decomposition for 3x3 matrix multiplication
    # Standard rank is 9. Strassen-like methods for 3x3 exist with rank 14 or 15.
    # However, we are multiplying 2x3 by 3x3.
    # We can view this as two independent 2x3 multiplications? No, the 3x3 is shared.
    # Actually, (2x3) * (3x3) = (2x3).
    # We can split the 2x3 into two 1x3 rows.
    # Row 0 of A (1x3) * B (3x3) = Row 0 of C (1x3)
    # Row 1 of A (1x3) * B (3x3) = Row 1 of C (1x3)
    
    # A 1x3 * 3x3 multiplication can be seen as a vector-matrix product.
    # The tensor for 1x3 * 3x3 -> 1x3 has rank 9 (schoolbook).
    # Is there a lower rank?
    # The tensor is $T_{j,k,l} = \delta_{j,l}$? No.
    # $C_l = \sum_j A_j B_{j,l}$.
    # This is the standard matrix multiplication tensor for 1x3 * 3x3.
    # The rank of $M_{1,3,3}$ is 9.
    # Since the two rows of A are independent and B is shared, the total rank is at most $9+9=18$.
    # But can we share terms?
    # The tensor for the full 2x3 * 3x3 is the direct sum of two 1x3 * 3x3 tensors?
    # Yes, because the output indices for row 0 and row 1 are disjoint in the first dimension.
    # The middle dimension (B) is the same, but the tensor structure is a block diagonal if we permute indices?
    # Let's check the tensor structure.
    # $T_{(i,j), (j',k), (k',i')} = \delta_{j,j'} \delta_{k,k'}$.
    # If we fix i, the tensor is $T_{j, j',k, k'} = \delta_{j,j'} \delta_{k,k'}$.
    # This is exactly the tensor for $1 \times 3 \times 3$ multiplication.
    # The rank of $M_{1,3,3}$ is 9.
    # So the rank of $M_{2,3,3}$ is at most $9+9=18$.
    # Is it lower?
    # The rank of $M_{n,m,p}$ is generally not additive in n.
    # However, for $M_{1,m,p}$, the rank is $mp$.
    # For $M_{2,3,3}$, the rank is known to be 16? Or 17?
    # Let's try to construct a rank 16 or 17 decomposition by finding linear dependencies.
    
    # Strategy: 
    # 1. Start with the schoolbook rank 18 decomposition.
    # 2. Try to find linear combinations of the rank-1 terms that cancel out to form fewer terms?
    # No, rank reduction requires finding a different decomposition.
    # 
    # Alternative Strategy: 
    # Use the known rank-14 decomposition for $3 \times 3$? No, that's for 3x3 * 3x3.
    # 
    # Let's try a different approach: 
    # The problem is equivalent to finding the rank of the tensor $T \in \mathbb{R}^{6 \times 9 \times 6}$.
    # We can use a "tensor network" approach or just brute-force search for integer/half-integer solutions 
    # with a small rank.
    
    # Given the time constraint and the difficulty of finding a rank < 18 analytically without a specific 
    # known decomposition, let's try to verify if rank 16 or 17 is achievable by checking 
    # if the schoolbook decomposition can be "compressed".
    
    # Actually, a simpler heuristic: 
    # The rank of $M_{2,3,3}$ is 16. (This is a known result in algebraic complexity, 
    # specifically that $R(M_{2,3,3}) \le 16$).
    # Let's try to find such a decomposition using a randomized search with integer constraints 
    # but guided by the structure.
    
    # Since finding an exact rank-16 decomposition from scratch is hard, 
    # let's see if we can improve upon the schoolbook (18) by checking for 
    # specific symmetries or using a known algorithm.
    
    # For now, let's implement a robust search for rank 16 and 17 using 
    # a local search on the integer lattice, but initialized smartly.
    
    # However, the prompt asks for a "MATERIALLY DIFFERENT" strategy.
    # The current one is L-BFGS on continuous variables.
    # The new one will be: **Algebraic Verification of a Constructed Candidate**.
    
    # We will construct a candidate decomposition using the following idea:
    # The tensor $M_{2,3,3}$ can be decomposed using the rank-9 decomposition of $M_{1,3,3}$ 
    # for each row, but we look for cross-terms.
    
    # Let's try a different known decomposition:
    # The rank of $M_{2,2,2}$ is 7 (Strassen).
    # $M_{2,3,3}$ contains $M_{2,2,2}$ as a sub-tensor.
    
    # Let's try to find a rank 16 decomposition by solving a system of linear equations 
    # for the entries of U, V, W assuming a specific structure.
    
    # To stay within time limits, we will:
    # 1. Return schoolbook (18) as a guaranteed fallback.
    # 2. Attempt a quick check for rank 17 and 16 using a randomized integer search 
    #    with a very tight budget (5 seconds).
    # 3. If found, return it.
    
    # But wait, the directive says "Keep this design and investigate improvements within its family".
    # The family is "finding a rank-minimal exact tensor decomposition".
    # The "different strategy" is the method of finding it.
    
    # Let's implement a **Lattice Search with Pruning**.
    # We assume entries are in {-1, 0, 1, 0.5, 1.5, ...}.
    # We try rank 16.
    # We fix U and V to random small integers, and solve for W?
    # No, it's non-linear.
    
    # Let's try a **Deterministic Construction** based on the fact that 
    # $M_{2,3,3}$ is a sub-tensor of $M_{3,3,3}$? No.
    
    # Actually, the most robust "different" strategy that is fast is:
    # **Use the known rank-14 decomposition for 3x3 multiplication to build a candidate for 2x3*3x3?**
    # No, dimensions don't match directly.
    
    # Let's stick to a **High-Precision Local Search** but with a **Different Initialization**.
    # The current program uses random Gaussian init.
    # We will use an initialization based on the schoolbook decomposition, 
    # then perturb it and use a **Gradient-Free Local Search** (Nelder-Mead) 
    # on the squared error, but with a **Constraint Projection** to integer/half-integers 
    # at the end.
    
    # Even better: **Simulated Annealing on the Rank**.
    # Start with rank 18.
    # Try to merge two terms into one?
    
    # Given the complexity, let's try to find if there is a known rank 16 decomposition 
    # in the literature that we can hardcode or derive.
    # A known result: $R(M_{2,3,3}) = 16$.
    # We can try to find it by searching for a solution to:
    # $T = \sum_{r=1}^{16} u_r \otimes v_r \otimes w_r$
    
    # Let's use a **Genetic Algorithm** or **Random Restart Local Search** on the integer grid.
    
    def decompose_to_tensor(U, V, W):
        r = U.shape[1]
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return T

    def check_exact(U, V, W, T_target, tol=1e-9):
        T_approx = decompose_to_tensor(U, V, W)
        return np.allclose(T_approx, T_target, atol=tol)

    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((nm, rank_schoolbook))
    V_schoolbook = np.zeros((mp, rank_schoolbook))
    W_schoolbook = np.zeros((pn, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_decomp = (U_schoolbook, V_schoolbook, W_schoolbook)

    # Try to find rank 16 or 17
    # We will use a randomized search with a very small number of iterations 
    # to avoid timeout, but with a smarter initialization.
    
    def try_rank_int_search(r, max_iters=500):
        if time.time() - start_time > deadline - 5:
            return None
            
        # Initialize with random integers in {-1, 0, 1}
        # This is a heuristic, not guaranteed to find the solution.
        
        for _ in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
                
            U = np.random.randint(-1, 2, (nm, r))
            V = np.random.randint(-1, 2, (mp, r))
            W = np.random.randint(-1, 2, (pn, r))
            
            # Quick check
            if check_exact(U, V, W, T_target):
                return (U, V, W)
                
        return None

    # The integer search is unlikely to find rank 16 quickly.
    # Let's try a **Continuous Optimization with Rounding** but with a **Better Objective**.
    # Minimize the nuclear norm or Frobenius norm of the error?
    # The current program minimizes Frobenius norm squared.
    
    # Let's try **L-BFGS-B** again but with a **Tighter Tolerance** and **More Iterations** 
    # for rank 16, and if it fails, return schoolbook.
    # This is essentially the same family, but we change the parameters significantly 
    # to "investigate improvements within its family".
    
    # Improvement: Use **Multiple Restarts** with **Different Scales** and **Seeds**.
    
    def try_rank_opt(r, max_attempts=20):
        if time.time() - start_time > deadline - 5:
            return None
            
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 5:
                break
                
            # Scale matters. Try scale 1.0 and 0.5
            scale = 1.0 if attempt % 2 == 0 else 0.5
            U_init = np.random.randn(nm, r) * scale
            V_init = np.random.randn(mp, r) * scale
            W_init = np.random.randn(pn, r) * scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def error_function(params):
                nm_r = nm * r
                mp_r = mp * r
                pn_r = pn * r
                
                U_flat = params[:nm_r]
                V_flat = params[nm_r:nm_r+mp_r]
                W_flat = params[nm_r+mp_r:nm_r+mp_r+pn_r]
                
                U = U_flat.reshape(nm, r)
                V = V_flat.reshape(mp, r)
                W = W_flat.reshape(pn, r)
                
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.sum((T_approx - T_target) ** 2)
                return error
            
            try:
                result = scipy.optimize.minimize(
                    error_function, 
                    params_init, 
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50}
                )
            except Exception:
                continue
                
            if result.fun < 1e-10:
                nm_r = nm * r
                mp_r = mp * r
                pn_r = pn * r
                
                U_opt = result.x[:nm_r].reshape(nm, r)
                V_opt = result.x[nm_r:nm_r+mp_r].reshape(mp, r)
                W_opt = result.x[nm_r+mp_r:nm_r+mp_r+pn_r].reshape(pn, r)
                
                # Round to nearest half-integer
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                if check_exact(U_rounded, V_rounded, W_rounded, T_target):
                    return (U_rounded, V_rounded, W_rounded)
                
                # Try integer rounding
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                if check_exact(U_int, V_int, W_int, T_target):
                    return (U_int, V_int, W_int)
                
        return None

    # Try rank 16 first, then 17
    for r in [16, 17]:
        if time.time() - start_time > deadline - 10:
            break
            
        decomp = try_rank_opt(r, max_attempts=10)
        if decomp is not None:
            best_rank = r
            best_decomp = decomp
            break

    def to_lists(mat):
        return mat.tolist()
        
    if best_rank == rank_schoolbook:
        return {"U": to_lists(U_schoolbook), "V": to_lists(V_schoolbook), "W": to_lists(W_schoolbook)}
    else:
        U, V, W = best_decomp
        return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}
# EVOLVE-BLOCK-END
