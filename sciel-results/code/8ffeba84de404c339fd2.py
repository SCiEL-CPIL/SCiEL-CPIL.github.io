"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 145.0
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct the known Rank 18 Solution (Fallback)
    # Strategy: Strassen for cols (0,1), Strassen for cols (3,4), Schoolbook for col 2.
    # This avoids the gap in the middle for Strassen pairs, but any pairing works for Rank 18.
    # Let's use (0,1) and (2,3) and Schoolbook for 4 to be consistent with previous logic, 
    # or (0,1) and (3,4) and Schoolbook for 2. 
    # Actually, (0,1) and (2,3) leaves 4. (0,1) and (3,4) leaves 2.
    # It doesn't matter for the rank, but let's stick to (0,1) and (2,3) + Schoolbook(4) 
    # as it was in the previous working code.
    
    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(t_offset, k_start):
            # Strassen products M1..M7
            # M1: (A00+A11)(B00+B11)
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            V[0*p + k_start][t_offset+0] = 1
            V[1*p + k_start+1][t_offset+0] = 1
            W[k_start*n + 0][t_offset+0] = 1
            W[(k_start+1)*n + 1][t_offset+0] = 1
            
            # M2: (A10+A11)B00
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            V[0*p + k_start][t_offset+1] = 1
            W[k_start*n + 1][t_offset+1] = 1
            W[(k_start+1)*n + 1][t_offset+1] = -1
            
            # M3: A00(B01-B11)
            U[0][t_offset+2] = 1
            V[0*p + k_start+1][t_offset+2] = 1
            V[1*p + k_start+1][t_offset+2] = -1
            W[(k_start+1)*n + 0][t_offset+2] = 1
            W[(k_start+1)*n + 1][t_offset+2] = 1
            
            # M4: A11(B10-B00)
            U[3][t_offset+3] = 1
            V[1*p + k_start][t_offset+3] = 1
            V[0*p + k_start][t_offset+3] = -1
            W[k_start*n + 0][t_offset+3] = 1
            W[k_start*n + 1][t_offset+3] = 1
            
            # M5: (A00+A01)B11
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            V[1*p + k_start+1][t_offset+4] = 1
            W[k_start*n + 0][t_offset+4] = -1
            W[(k_start+1)*n + 0][t_offset+4] = 1
            
            # M6: (A10-A00)(B00+B01)
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            V[0*p + k_start][t_offset+5] = 1
            V[0*p + k_start+1][t_offset+5] = 1
            W[(k_start+1)*n + 1][t_offset+5] = 1
            
            # M7: (A01-A11)(B10+B11)
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1
            V[1*p + k_start][t_offset+6] = 1
            V[1*p + k_start+1][t_offset+6] = 1
            W[k_start*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0) # Cols 0, 1
        add_strassen_block(7, 2) # Cols 2, 3
        
        # Schoolbook for Col 4
        t = 14
        U[0][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[1][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        
        U[2][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 1][t] = 1
        t += 1
        
        U[3][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 1][t] = 1
        
        return {"U": U, "V": V, "W": W}

    sol_18 = get_rank_18_solution()
    
    # 2. Attempt to find Rank 17 Solution
    # Strategy: Start from Rank 18, force one term to be zero, and optimize the rest.
    # We try different terms to zero out. The "least important" term is often a schoolbook term.
    
    def error_func(params, rank, U_init, V_init, W_init):
        U = params[:n*m*rank].reshape((n*m, rank))
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape((m*p, rank))
        W = params[n*m*rank + m*p*rank :].reshape((p*n, rank))
        
        err = 0.0
        # Tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Target is 1 if i==j (wait, no, target is 1 if A[i,j] * B[j,k] contributes to C[i,k])
        # The tensor for matrix mult M_{n,m,p} has entries T_{i,j,k} = 1 if j is the inner index?
        # No, the standard embedding is:
        # T[i, j, k] = 1 if the term A[i,j]B[j,k] appears in C[i,k].
        # So for fixed i,k, the sum over j of A[i,j]B[j,k] is C[i,k].
        # The tensor entry T[i,j,k] is 1 if we are computing C[i,k] using A[i,j] and B[j,k].
        # Wait, the indices in the problem statement are:
        # U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # Index of U: i*m + j ? Or i*n + j?
        # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # Standard convention for U: row index is i*m + j? 
        # Let's check the seed code:
        # U[0][t] ... U[3][t] ...
        # n=2, m=2. U has 4 rows.
        # In get_rank_18_solution:
        # U[0] corresponds to A[0,0]?
        # U[1] corresponds to A[0,1]?
        # U[2] corresponds to A[1,0]?
        # U[3] corresponds to A[1,1]?
        # This implies row index = i*m + j.
        
        # V index: j*p + k.
        # W index: k*n + i.
        
        # Target Tensor T[i, j, k] is 1 if this combination of indices is part of the multiplication.
        # C[i, k] = sum_j A[i, j] B[j, k].
        # So for every i, j, k, the term A[i,j]B[j,k] contributes to C[i,k].
        # Thus, T[i, j, k] = 1 for ALL i, j, k?
        # No, that's not right.
        # The tensor T is defined such that C = sum_c (A * U_c) (V_c * B) ... no.
        # C = sum_c U_c A V_c B W_c^T ...
        # Let's look at the check in the previous code:
        # target = 1.0 if j == jp else 0.0
        # This check was for the 2x2x3 subproblem where it seemed to assume a specific structure?
        # No, the previous code's check was:
        # for i in range(n_sub):
        #   for j in range(m_sub):
        #     for jp in range(m_sub):
        #       for k in range(p_sub):
        #         target = 1.0 if j == jp else 0.0
        # This is WRONG for general matrix multiplication.
        # The target tensor for M_{n,m,p} is T[i,j,k] = 1 for all i,j,k?
        # No.
        # If T[i,j,k]=1 for all, then sum_c U[i,j,c]V[j,k,c]W[k,i,c] = 1 for all i,j,k.
        # This would mean A[i,j]B[j,k] is always 1? No.
        
        # Let's re-read the tensor definition.
        # The multiplication tensor $\mathcal{M}_{n,m,p}$ is the tensor such that
        # $\langle \mathcal{M}, A \otimes B \otimes C \rangle = \text{tr}(ABC)$? No.
        # It is the tensor where $T_{i,j,k} = 1$ if the index $j$ connects $i$ and $k$.
        # Actually, the standard definition is:
        # $T_{i,j,k} = 1$ for all $i \in \{0..n-1\}, j \in \{0..m-1\}, k \in \{0..p-1\}$.
        # Wait, if $T_{i,j,k}=1$ for all, then the decomposition requires:
        # $\sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1$ for ALL i,j,k.
        # This is NOT matrix multiplication.
        
        # Correct Tensor for Matrix Multiplication $C = AB$:
        # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
        # We want to express $C_{i,k}$ as $\sum_c ( \sum_j U_{i,j,c} A_{i,j} ) ( \sum_k V_{j,k,c} B_{j,k} ) ...$
        # No, the bilinear form is:
        # $C_{i,k} = \sum_c U_{i,c}^{(1)} A_{i, \cdot} \cdot U_{i,c}^{(2)} ...$
        
        # Let's look at the structure of U, V, W again.
        # U is (n*m, rank). Row $i*m+j$ corresponds to $A_{i,j}$.
        # V is (m*p, rank). Row $j*p+k$ corresponds to $B_{j,k}$.
        # W is (p*n, rank). Row $k*n+i$ corresponds to $C_{k,i}$? Or $C_{i,k}$?
        # The problem says "output factor follows the official trace convention W[k*n+i,r]".
        # Usually, $C_{i,k}$ is stored. If W index is $k*n+i$, then W row corresponds to $C_{k,i}$?
        # If W row is $k*n+i$, and we sum $U_{i,j,c} V_{j,k,c} W_{k,i,c}$,
        # then the result is associated with index pair $(i,k)$ via the W term?
        
        # Let's verify the target condition.
        # We want $\sum_c U_{i,m+j,c} V_{j,p+k,c} W_{k,n+i,c} = \delta_{\text{something}}$?
        # No, we want the sum to be 1 if the indices $(i,j,k)$ form a valid term in $C_{i,k} = \sum_j A_{i,j}B_{j,k}$.
        # Every combination of $i, j, k$ is a valid term in the sum for $C_{i,k}$.
        # So, for ANY $i, j, k$, the term $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
        # Therefore, the tensor entry $T_{i,j,k}$ should be 1 for ALL $i,j,k$.
        
        # Let's check the previous code's check again:
        # `target = 1.0 if j == jp else 0.0`
        # This implies the previous code was solving a DIFFERENT problem or was WRONG.
        # If the previous code scored 1.111, it means it produced a valid Rank 18 solution.
        # Did the evaluator accept it? Yes.
        # So the evaluator's check must match the code's check?
        # Or the code's check was wrong but the solution happened to be valid?
        # Unlikely.
        
        # Let's re-read the task: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 5 matrix."
        # Standard tensor $\mathcal{M}_{2,2,5}$.
        # Entries are $T_{i,j,k}$.
        # If I use Strassen, I know the coefficients.
        # In Strassen for 2x2x2:
        # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This term contributes to $C_{00}$ and $C_{11}$.
        # So $T_{0,0,0}$? No.
        # The decomposition is:
        # $C_{i,k} = \sum_c \sum_j U_{i,j,c} V_{j,k,c} W_{k,i,c} A_{i,j} B_{j,k}$? No.
        
        # The bilinear decomposition is:
        # $C_{i,k} = \sum_c U_{i,c} A_{i, \dots} ...$
        
        # Let's assume the standard definition:
        # $\sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c} = 1$ for all $i,j,k$.
        # If this is the target, then the Rank 18 solution provided in the inspiration should satisfy this.
        # Let's test the Strassen block for M1:
        # $U_{0,0} = 1, U_{3,0} = 1$ (A00, A11)
        # $V_{0,0} = 1, V_{1,1} = 1$ (B00, B11) -> Note: V index $j*p+k$. $j=0,k=0 \to 0$. $j=1,k=1 \to 1*2+1=3$?
        # In the code: `V[0*p + k0][t]` and `V[1*p + k1][t]`.
        # For 2x2x2 block, $p=2$.
        # $k0=0, k1=1$.
        # $V[0][t]=1, V[3][t]=1$.
        # $W[0][t]=1, W[3][t]=1$?
        # In code: `W[k0*n + 0][t]` -> $W[0][t]=1$. `W[k1*n + 1][t]` -> $W[1*2+1=3][t]=1$.
        
        # So the term is $1 \cdot 1 \cdot 1$ for indices:
        # U row 0 (A00), V row 0 (B00), W row 0 (C00? index 0 = 0*2+0).
        # U row 3 (A11), V row 3 (B11), W row 3 (C11? index 3 = 1*2+1).
        
        # So this term contributes to:
        # $i=0, j=0, k=0 \implies T_{0,0,0} = 1$.
        # $i=1, j=1, k=1 \implies T_{1,1,1} = 1$.
        
        # Does Strassen have other terms for M1?
        # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This expands to $A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
        # In the tensor decomposition, we don't expand. We just have one term $c=0$.
        # The contribution to $C_{i,k}$ is determined by the W vector.
        # Wait, the formula is:
        # $C_{i,k} = \sum_c W_{k*n+i, c} \left( \sum_j U_{i*m+j, c} A_{i,j} \right) \left( \sum_j V_{j*p+k, c} B_{j,k} \right)$?
        # No, that's not right either.
        
        # The correct bilinear form for matrix multiplication tensor is:
        # $T_{i,j,k} = 1$ for all $i,j,k$.
        # And the decomposition is:
        # $T_{i,j,k} = \sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}$.
        
        # Let's check if the Strassen solution satisfies $T_{i,j,k}=1$ for all $i,j,k$.
        # For M1 ($c=0$):
        # $U_{0,0}=1, U_{3,0}=1$. Others 0.
        # $V_{0,0}=1, V_{3,0}=1$. Others 0.
        # $W_{0,0}=1, W_{3,0}=1$. Others 0.
        
        # Product for $(i,j,k)$:
        # If $i=0, j=0, k=0$: $U_{0,0} V_{0,0} W_{0,0} = 1 \cdot 1 \cdot 1 = 1$.
        # If $i=0, j=0, k=1$: $U_{0,0} V_{1,0} W_{1,0}$. $V_{1,0}=0$. So 0.
        # But $T_{0,0,1}$ should be 1?
        # $C_{0,1} = A_{0,0}B_{0,1} + A_{0,1}B_{1,1}$.
        # So yes, $A_{0,0}B_{0,1}$ is a term. So $T_{0,0,1}$ should be 1.
        # But the Strassen term M1 does NOT contribute to $T_{0,0,1}$.
        # It contributes to $T_{0,0,0}$ and $T_{1,1,1}$ and cross terms?
        
        # Ah, the Strassen algorithm uses LINEAR COMBINATIONS.
        # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This single product $M_1$ is used to reconstruct $C$.
        # $C_{00} = M_1 + M_4 - M_5 - M_7$.
        # $C_{01} = M_3 + M_5$.
        # etc.
        
        # The tensor decomposition represents the LINEAR MAP.
        # The tensor $T$ is defined such that $C = T(A,B)$.
        # $T_{i,j,k}$ is the coefficient of $A_{i,j} B_{j,k}$ in the expression for $C_{i,k}$?
        # No, $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
        # So $T_{i,j,k} = 1$ for all $i,j,k$.
        
        # The decomposition is:
        # $\sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = T_{i,j,k} = 1$.
        
        # So, for the Strassen solution, does $\sum_{c=0}^6 U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1$ for all $i,j,k$?
        # Let's check $(i=0, j=0, k=1)$.
        # We need the sum to be 1.
        # $U_{0,0,c}$ is non-zero for $c=0,2,4,5$.
        # $V_{0,1,c}$ is non-zero for $c=2,5,6$.
        # $W_{1,0,c}$ is non-zero for $c=2,4,6$.
        
        # Let's calculate the product for $c$ where all three are non-zero.
        # Intersection of indices:
        # $c=2$: $U_{0,0,2}=1, V_{0,1,2}=1, W_{1,0,2}=1$. Product = 1.
        # $c=4$: $U_{0,0,4}=1, V_{0,1,4}=0$. Product 0.
        # $c=5$: $U_{0,0,5}=-1, V_{0,1,5}=1, W_{1,0,5}=0$. Product 0.
        # $c=6$: $U_{0,0,6}=0$. Product 0.
        
        # So sum is 1. Correct!
        
        # So the target is indeed $T_{i,j,k} = 1$ for all $i,j,k$.
        
        # The previous code's check `target = 1.0 if j == jp else 0.0` was WRONG.
        # It was checking a different condition.
        # However, the code returned a valid solution.
        # Why?
        # Because the code constructed the Strassen blocks manually, which ARE correct for $T_{i,j,k}=1$.
        # The `check_exact_223` function was only used to verify the *optimized* sub-solution.
        # Since the optimization likely failed or wasn't used, the fallback (manual Strassen) was returned.
        # The manual Strassen is correct.
        
        # So, for my new optimization attempt, I MUST use the correct target: $T_{i,j,k}=1$.

    def error_full(params, rank):
        U = params[:n*m*rank].reshape((n*m, rank))
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape((m*p, rank))
        W = params[n*m*rank + m*p*rank :].reshape((p*n, rank))
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    err += (val - 1.0) ** 2
        return err

    def check_exact_full(U, V, W, rank, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integers(arr, tol=1e-5):
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    # Try to find Rank 17
    # Initialize with Rank 18 solution, set one term to 0, and optimize.
    # We try zeroing out each of the 18 terms one by one.
    
    U_18 = np.array(sol_18["U"], dtype=float)
    V_18 = np.array(sol_18["V"], dtype=float)
    W_18 = np.array(sol_18["W"], dtype=float)
    
    best_sol_17 = None
    
    if time.time() < DEADLINE - 30:
        for term_to_remove in range(18):
            if time.time() > DEADLINE - 5:
                break
            
            # Create initial guess for Rank 17
            # We take the Rank 18 solution, remove column `term_to_remove`, and keep the rest.
            # This is a valid initialization for Rank 17.
            
            U_init = np.delete(U_18, term_to_remove, axis=1)
            V_init = np.delete(V_18, term_to_remove, axis=1)
            W_init = np.delete(W_18, term_to_remove, axis=1)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize
            # Use L-BFGS-B for speed
            res = minimize(error_full, x0, args=(17,), method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False})
            
            if res.fun < 1e-8:
                U_opt = res.x[:n*m*17].reshape((n*m, 17))
                V_opt = res.x[n*m*17 : n*m*17 + m*p*17].reshape((m*p, 17))
                W_opt = res.x[n*m*17 + m*p*17 :].reshape((p*n, 17))
                
                # Snap to half-integers
                U_s = snap_to_half_integers(U_opt)
                V_s = snap_to_half_integers(V_opt)
                W_s = snap_to_half_integers(W_opt)
                
                if U_s is not None and V_s is not None and W_s is not None:
                    if check_exact_full(U_s, V_s, W_s, 17, tol=1e-6):
                        best_sol_17 = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                        break # Found one, stop

    if best_sol_17:
        return best_sol_17
    else:
        return sol_18

# EVOLVE-BLOCK-END
