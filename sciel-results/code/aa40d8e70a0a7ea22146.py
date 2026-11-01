"""
Strategy: Iterative Rank Reduction via Linear Solving.
Instead of just returning the schoolbook rank (60), we attempt to reduce the rank
by removing one term at a time and solving for the remaining terms to exactly
reproduce the matrix multiplication tensor. If a solution exists with integer
or half-integer precision, we accept it. If not, we revert and try another term.
This explores the space of lower-rank decompositions.
"""

import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    n, m, p = DIMENSIONS
    # Tensor T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # Actually, matrix multiplication tensor is T_{i,j,k} = 1 for all valid i,j,k
    # The equation is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
    # Note the index order in W is [k*n + i, r] per the prompt "W[k*n+i,r]".
    
    # Initial Schoolbook
    rank_init = n * m * p
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    current_rank = rank_init
    best_U, best_V, best_W = U, V, W

    # We will try to remove terms one by one.
    # A term r is defined by indices (i_r, j_r, k_r) in the schoolbook expansion.
    # If we remove term r, we have a system of linear equations for the remaining variables.
    # However, the system is nonlinear (trilinear). 
    # But if we fix two of the matrices, it's linear in the third.
    # Strategy: Fix U and V, solve for W? No, U and V change too.
    
    # Alternative Strategy: Randomized Search for Lower Rank
    # Since exact trilinear decomposition is hard, we use a heuristic:
    # Try random integer/half-integer matrices of rank R < current_rank.
    # Check if they multiply to the identity tensor.
    # Given the small size (2x5x6), the tensor has 60 entries.
    # Rank 60 is the schoolbook. Is there a rank < 60?
    # For 2x2x2, rank is 7. For 2x3x3, rank is 15? (Strassen for 2x2 is 7, schoolbook 8).
    # For 2x5x6, it is highly likely the rank is exactly 60 or very close.
    # However, we must try to find a better one.
    
    # Let's use a local search on the coefficients of a rank-R decomposition.
    # We start with R = current_rank - 1.
    # We define the tensor equation:
    # T[i,j,k] = sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We will attempt to solve this using a non-linear least squares approach
    # with integer/half-integer constraints, or simply random restarts with rounding.
    
    def verify(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        # Check all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_rank_reduction(target_rank):
        """
        Attempts to find a decomposition of the given target_rank.
        Uses a simple heuristic: take the schoolbook decomposition,
        randomly zero out (current_rank - target_rank) terms,
        then try to adjust the remaining terms slightly to compensate?
        This is difficult.
        
        Better heuristic:
        Randomly generate U, V, W with entries in {0, 1, -1, 0.5, -0.5}
        and check if they work.
        Given the size, this is a needle in a haystack, but for small ranks
        it might work if the rank is close to schoolbook.
        
        Actually, let's look at the structure.
        The tensor is rank 60.
        If we assume the rank is 60, we return 60.
        If we can find 59, we return 59.
        
        Let's try a specific algebraic trick.
        Can we merge two terms?
        Term (i,j,k) and (i',j',k')?
        Only if they share indices in a way that allows factoring.
        In schoolbook, each term is a basis tensor e_i ⊗ e_j ⊗ e_k.
        The tensor T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k.
        This is the all-ones tensor of size n x m x p.
        The rank of the all-ones tensor of size n x m x p is NOT necessarily n*m*p.
        Wait. The matrix multiplication tensor is NOT the all-ones tensor.
        
        Correction:
        The matrix multiplication tensor $T_{mm}$ has entries $T_{i,j,k} = 1$ if we are computing $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
        So $T_{i,j,k} = 1$ for ALL i, j, k in range.
        Yes, the matrix multiplication tensor IS the all-ones tensor of size $n \times m \times p$?
        No!
        $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
        The tensor maps $A \otimes B \to C$.
        The entry corresponding to $A_{i,j}, B_{j,k} \to C_{i,k}$ is 1.
        So the tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
        This is indeed the tensor of all ones.
        
        The rank of the all-ones tensor $J_{n,m,p}$ is known to be smaller than $nmp$ for some dimensions?
        For $2 \times 2 \times 2$, rank is 7 (Strassen). $2 \cdot 2 \cdot 2 = 8$.
        For $2 \times 3 \times 3$, rank is 15? $2 \cdot 3 \cdot 3 = 18$.
        For $2 \times 5 \times 6$, rank might be significantly lower than 60.
        
        So the current program (rank 60) is likely suboptimal.
        We need to find the minimal rank of the all-ones tensor $J_{2,5,6}$.
        
        Strategy:
        1. Check known results or use a search.
        2. Since we can't look up 2x5x6 specifically, we can try to construct one.
        
        Known: Rank of $J_{n,m,p}$.
        For $n=2$, the rank of $J_{2,m,p}$ is related to the rank of $J_{m,p}$?
        Actually, $J_{2,m,p}$ can be viewed as two copies of $J_{m,p}$ coupled.
        
        Let's try to find a decomposition for $J_{2,5,6}$ with rank < 60.
        
        We can use the fact that $J_{n,m,p}$ is the sum of basis tensors.
        
        Let's try a random search for a rank R decomposition with integer coefficients.
        We will iterate R from a lower bound up to 60.
        Lower bound: $\max(n, m, p)$? No, that's for something else.
        Lower bound for $J_{2,m,p}$:
        It is known that rank($J_{2,m,p}$) $\le$ rank($J_{m,p}$) + rank($J_{m,p}$)?
        
        Let's just do a randomized search for small R.
        """
        pass

    # Since we don't know the exact rank, and search is expensive,
    # and we have a 120s budget, let's try to find a rank < 60.
    
    # Heuristic:
    # The rank of $J_{2,m,p}$ is at most $m \cdot p + m \cdot p$? No.
    # Consider $J_{2,m,p}$.
    # $J_{2,m,p} = J_{1,m,p} \otimes e_1 + J_{1,m,p} \otimes e_2$? No.
    
    # Let's try to construct a rank $m \cdot p$ decomposition?
    # If we set $U[i, r]$ such that it doesn't depend on i?
    # If $U$ is constant in i, then $C_1$ and $C_2$ would be identical.
    # So U must depend on i.
    
    # Let's try rank $2 \cdot m \cdot p$? That's 60.
    
    # Is there a known result for $2 \times 5 \times 6$?
    # Rank of $J_{2,m,p}$ is $m \cdot p$ if one of m,p is 1?
    # If $m=1$, $J_{2,1,p}$ has rank 2 (since $C_1 = A_1 B_1$, $C_2 = A_2 B_1$).
    # Wait, if $m=1$, $A$ is $2 \times 1$, $B$ is $1 \times p$.
    # $C = A B^T$.
    # $C_{i,k} = A_{i,1} B_{1,k}$.
    # This is rank 1 as a bilinear map?
    # The tensor is $e_1 \otimes e_1 \otimes J_p + e_2 \otimes e_1 \otimes J_p$.
    # Rank is 2.
    # Schoolbook rank is $2 \cdot 1 \cdot p = 2p$.
    # So for $m=1$, rank is 2, not $2p$.
    
    # So for $m=5, p=6$, the rank should be much less than 60.
    
    # Let's try to find the rank of $J_{2,5,6}$.
    # General bound: Rank($J_{n,m,p}$) $\le n \cdot \text{Rank}(J_{m,p})$?
    # Actually, $J_{n,m,p}$ can be decomposed using the decomposition of $J_{m,p}$?
    # No, the indices are coupled.
    
    # Let's try a specific construction.
    # We want $\sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1$.
    
    # If we can find a decomposition of $J_{m,p}$ with rank $R_{mp}$,
    # can we lift it to $J_{2,m,p}$ with rank $2 R_{mp}$?
    # $J_{m,p} = \sum_{r=1}^{R_{mp}} v_{j,r} w_{k,r}$.
    # Then $J_{2,m,p}$?
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # This is not separable in i.
    
    # However, note that $J_{2,m,p}$ is the tensor for $2 \times m \times p$.
    # It is known that Rank($J_{2,m,p}$) = $2 \cdot \text{Rank}(J_{m,p})$?
    # Let's check $m=1, p=p$. Rank($J_{1,p}$) = 1 (since $J_{1,p}$ is just a vector of 1s, rank 1? No, $1 \times 1 \times p$ tensor is rank 1).
    # Then $2 \cdot 1 = 2$. Matches.
    
    # Let's check $m=2, p=2$.
    # Rank($J_{2,2}$) for $1 \times 2 \times 2$?
    # $1 \times 2 \times 2$ tensor is $J_{2,2}$. Rank is 1?
    # $C_{1,k} = \sum_j A_{1,j} B_{j,k}$.
    # This is just matrix multiplication $1 \times 2$ by $2 \times 2$.
    # The tensor is $e_1 \otimes J_{2,2}$.
    # Rank is Rank($J_{2,2}$).
    # $J_{2,2}$ (all ones $2 \times 2$) has rank 1?
    # $J_{2,2} = [1,1; 1,1] = [1,1]^T [1,1]$.
    # Yes, rank 1.
    # So Rank($J_{1,2,2}$) = 1.
    # Then Rank($J_{2,2,2}$) should be $2 \cdot 1 = 2$?
    # But we know Rank($J_{2,2,2}$) is 7 (Strassen).
    # So the formula $2 \cdot \text{Rank}(J_{m,p})$ is WRONG.
    
    # The coupling is non-trivial.
    
    # Let's try to search for a low rank decomposition using random integers.
    # Given the time constraint, we will try a few random ranks.
    
    def find_decomposition(target_rank):
        """
        Random search for integer decomposition.
        """
        n, m, p = DIMENSIONS
        # Number of variables: R * (n*m + m*p + p*n)
        # For R=20, vars = 20 * (10 + 30 + 12) = 20 * 52 = 1040.
        # Equations: 60.
        # Underdetermined.
        
        # We use a simple approach:
        # 1. Randomly initialize U, V, W with small integers.
        # 2. Solve for W given U, V?
        # The system is linear in W if U, V are fixed.
        # Eq: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
        # For fixed i,k: sum_r (sum_j U[i,j,r] V[j,k,r]) W[k,i,r] = 1.
        # Let M_{ik,r} = sum_j U[i,j,r] V[j,k,r].
        # Then sum_r M_{ik,r} W[k,i,r] = 1.
        # This is a linear system for W.
        # But W has p*n entries per r.
        # We have n*p equations (for each i,k).
        # Unknowns: R * p * n.
        
        # So for fixed U, V, we can solve for W.
        # Then we check if the solution W is "nice" (integer/half-integer).
        
        # Let's try this.
        
        for _ in range(100): # 100 restarts
            if time.time() > deadline - 5:
                break
                
            # Random U, V
            # Use small integers to keep numbers manageable
            U_rand = np.random.randint(-2, 3, (n*m, target_rank))
            V_rand = np.random.randint(-2, 3, (m*p, target_rank))
            
            # Construct M matrix
            # M[i, k, r] = sum_j U[i, j, r] V[j, k, r]
            # U is (n*m, R). Index i*m+j.
            # V is (m*p, R). Index j*p+k.
            
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for r in range(target_rank):
                        s = 0.0
                        for j in range(m):
                            s += U_rand[i*m+j, r] * V_rand[j*p+k, r]
                        M[i, k, r] = s
            
            # System: sum_r M[i, k, r] W[k, i, r] = 1
            # Let's flatten W for each r? No, W[k,i,r] is the variable.
            # For each (i,k), we have a linear equation in the vector W_r (length p*n).
            # Actually, W is a 3D array (p, n, R).
            # Let's treat W[r] as a vector of length p*n.
            # Eq for (i,k): sum_r M[i,k,r] * W_flat[r, k*n+i] = 1.
            
            # This is a system of n*p equations with R*p*n unknowns.
            # We can use least squares to find a solution W.
            
            # Construct A matrix: (n*p, R*p*n)
            # Row idx: i*p + k
            # Col idx: r*p*n + k*n + i
            
            A = np.zeros((n*p, target_rank * p * n))
            for i in range(n):
                for k in range(p):
                    row_idx = i * p + k
                    for r in range(target_rank):
                        col_idx = r * p * n + k * n + i
                        A[row_idx, col_idx] = M[i, k, r]
                        
            b = np.ones(n*p)
            
            # Solve A x = b
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Check residual
            if residuals and residuals[0] > 1e-9:
                continue
                
            # Reshape W
            W_sol = W_flat.reshape(target_rank, p, n)
            
            # Check if W_sol is close to integer or half-integer
            # Scale by 2 to check for integers
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            if np.allclose(W_sol, W_rounded / 2.0, atol=1e-6):
                # We found a valid half-integer solution!
                # Convert to the required format
                U_out = np.zeros((n*m, target_rank))
                V_out = np.zeros((m*p, target_rank))
                W_out = np.zeros((p*n, target_rank))
                
                # U and V are already integers from randint
                U_out = U_rand
                V_out = V_rand
                for r in range(target_rank):
                    for k in range(p):
                        for i in range(n):
                            W_out[k*n+i, r] = W_sol[r, k, i]
                            
                return U_out, V_out, W_out
            else:
                # Try to round W and check if it works with the original U, V
                # This might fail if rounding error is too high
                pass
                
        return None

    # Try ranks from 10 up to 59
    # Start from a reasonable guess.
    # For 2x5x6, maybe rank is around 20-30?
    
    # To save time, we'll try a few specific ranks.
    # If we find one, we return it.
    # If not, we return the schoolbook rank 60.
    
    # Note: The problem asks for "rank-minimal".
    # If we don't find a better one, we must return the best we have.
    # The schoolbook is always valid.
    
    # Let's try ranks 20, 30, 40, 50.
    # If we find a valid one, we can try to lower it further.
    
    found = None
    for r in [20, 25, 30, 40, 50, 55, 59]:
        if time.time() > deadline - 10:
            break
        res = find_decomposition(r)
        if res is not None:
            found = res
            # Don't break immediately, maybe we can find lower?
            # But finding lower is harder.
            # Let's just break and return this valid lower rank.
            break
            
    if found:
        U, V, W = found
        # Ensure integer/half-integer format is preserved in the return
        # The prompt says "integer/half-integer entries".
        # Our search ensures this.
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# Keep the original function name and signature
def construct():
    return construct()

# Wait, the function name in the seed is construct.
# I redefined it inside the block? No, I should define it once.

# Let's refactor to a single clean function.

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    # Schoolbook
    rank_init = n * m * p
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    def find_decomposition(target_rank):
        for _ in range(50):
            if time.time() > deadline - 5:
                break
                
            U_rand = np.random.randint(-2, 3, (n*m, target_rank))
            V_rand = np.random.randint(-2, 3, (m*p, target_rank))
            
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for r in range(target_rank):
                        s = 0.0
                        for j in range(m):
                            s += U_rand[i*m+j, r] * V_rand[j*p+k, r]
                        M[i, k, r] = s
            
            A = np.zeros((n*p, target_rank * p * n))
            for i in range(n):
                for k in range(p):
                    row_idx = i * p + k
                    for r in range(target_rank):
                        col_idx = r * p * n + k * n + i
                        A[row_idx, col_idx] = M[i, k, r]
                        
            b = np.ones(n*p)
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            if residuals and residuals[0] > 1e-9:
                continue
                
            W_sol = W_flat.reshape(target_rank, p, n)
            
            W_scaled = W_sol * 2
            W_rounded = np.round(W_scaled)
            if np.allclose(W_sol, W_rounded / 2.0, atol=1e-6):
                U_out = U_rand
                V_out = V_rand
                W_out = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    for k in range(p):
                        for i in range(n):
                            W_out[k*n+i, r] = W_sol[r, k, i]
                return U_out, V_out, W_out
        return None

    # Try to find a lower rank
    # We prioritize finding ANY rank < 60.
    # Try ranks in increasing order of likelihood?
    # Random search is better for lower ranks?
    # Let's try a few.
    
    for r in [30, 40, 50, 55, 59]:
        if time.time() > deadline - 10:
            break
        res = find_decomposition(r)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

