"""Matrix multiplication 2x3x6 tensor decomposition; DIRECT structural construction."""
DIMENSIONS = [2, 3, 6]

import time
import numpy as np
import itertools

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS
    
    # Target: Find a decomposition with rank < 36
    # We will search for R from 24 to 35.
    # If we find one, we return it.
    # If not, we return the trivial rank 36 decomposition.
    
    def is_valid_decomposition(U, V, W, tol=1e-9):
        # Check if U, V, W entries are integer or half-integer
        for mat in [U, V, W]:
            for val in mat:
                # val should be close to k/2
                two_val = val * 2
                if abs(two_val - round(two_val)) > tol:
                    return False
        # Check if decomposition is exact
        # T_ijk = sum_r U_ir V_jr W_kr
        # U shape (n, R), V shape (m, R), W shape (p, R)
        # Note: The problem statement says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
        # This is a flattened representation.
        # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ?
        # Let's re-read the interface.
        # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # "output factor follows the official trace convention W[k*n+i,r]"
        # This implies the tensor is $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$?
        # No, standard tensor decomposition for matrix mult is:
        # $C_{ik} = \sum_j A_{ij} B_{jk}$.
        # Tensor $T_{ijk}$ where indices are $i$ (row of A), $j$ (col of A / row of B), $k$ (col of B).
        # $T_{ijk} = 1$ if it's a valid term? No, the tensor IS the multiplication map.
        # The tensor $M_{n,m,p}$ has entries $M_{i,j,k} = 1$ if we view it as a trilinear form?
        # Actually, the tensor is defined such that $\langle M, A \otimes B \otimes C \rangle = \text{tr}(C A B)$.
        # The rank-1 decomposition is $M = \sum_r u_r \otimes v_r \otimes w_r$.
        # $u_r$ is a vector of size $nm$, $v_r$ of size $mp$, $w_r$ of size $pn$.
        # The condition is: for all A, B, C:
        # $\sum_{i,j,k} A_{ij} B_{jk} C_{ki} = \sum_r (\sum_{i,j} u_{r,ij} A_{ij}) (\sum_{j,k} v_{r,jk} B_{jk}) (\sum_{k,i} w_{r,ki} C_{ki})$.
        # This must hold.
        # Equivalently, the tensor $T$ with indices $I=(i,j), J=(j,k), K=(k,i)$?
        # No, the standard definition is:
        # $T_{(i,j), (j,k), (k,i)}$? No, the indices of the tensor are coupled.
        # The tensor $T \in \mathbb{R}^{nm \times mp \times pn}$?
        # Let's check the trace convention.
        # $\text{tr}(C A B) = \sum_{i,j,k} C_{ki} A_{ij} B_{jk}$.
        # So the tensor has entries $T_{I, J, K}$ where $I=(i,j)$, $J=(j,k)$, $K=(k,i)$.
        # But $i,j,k$ are not independent. $i$ is in $I$ and $K$. $j$ is in $I$ and $J$. $k$ is in $J$ and $K$.
        # This is the "matrix multiplication tensor".
        # The rank is the minimum $R$ such that $T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r$.
        # Where $u_r \in \mathbb{R}^{nm}$, $v_r \in \mathbb{R}^{mp}$, $w_r \in \mathbb{R}^{pn}$.
        # The entry $T_{I,J,K}$ is 1 if the indices are consistent (i.e., $I=(i,j), J=(j,k), K=(k,i)$) and 0 otherwise.
        # Wait, $I$ is a pair $(i,j)$. $J$ is a pair $(j,k)$. $K$ is a pair $(k,i)$.
        # So $T_{(i,j), (j,k), (k,i)} = 1$.
        # All other entries are 0.
        
        # Let's verify this interpretation.
        # If we have a decomposition $U, V, W$ of rank $R$.
        # $U$ is $nm \times R$. $U_{I, r}$.
        # $V$ is $mp \times R$. $V_{J, r}$.
        # $W$ is $pn \times R$. $W_{K, r}$.
        # The reconstructed tensor entry at $(I, J, K)$ is $\sum_r U_{I,r} V_{J,r} W_{K,r}$.
        # This must equal $T_{I,J,K}$.
        
        # So we need to check:
        # For all $i \in [0,n), j \in [0,m), k \in [0,p)$:
        # $I = i*m + j$
        # $J = j*p + k$
        # $K = k*n + i$
        # Sum_r U[I,r] V[J,r] W[K,r] == 1
        
        # For all other combinations of I, J, K that are NOT consistent (i.e., there is no i,j,k such that I=im+j, J=jp+k, K=kn+i), the sum must be 0.
        # Note: The set of valid (I,J,K) tuples is exactly the set of triples $(i*m+j, j*p+k, k*n+i)$.
        # There are $n*m*p$ such tuples.
        # The total number of possible (I,J,K) tuples is $nm \cdot mp \cdot pn = n^2 m^2 p^2$.
        # For $n=2, m=3, p=6$, this is $36 \cdot 18 \cdot 12 = 7776$.
        # We only need to check the 36 non-zero entries?
        # No, if the non-zero entries are correct, are the zero entries automatically zero?
        # Not necessarily. The decomposition might have "leaks" into other entries.
        # So we must check ALL entries or at least enough to ensure validity.
        # Checking all 7776 entries is fast.
        
        # However, we can optimize.
        # We just need to verify the 36 non-zero entries are 1.
        # And we can assume the rest are 0 if the rank is low enough? No.
        # We must verify the zero entries too.
        
        # Let's build the full tensor check.
        # This might be slow if done naively in Python for many iterations.
        # But R is small (<36).
        # We can use numpy broadcasting.
        
        # U: (nm, R)
        # V: (mp, R)
        # W: (pn, R)
        
        # Reconstructed tensor T_rec[I, J, K] = sum_r U[I, r] V[J, r] W[K, r]
        # This is a tensor of size (nm, mp, pn).
        # We can compute this using einsum or loops.
        # Given small sizes, loops are fine.
        
        # But we only need to check specific entries.
        # 1. The 36 valid entries must be 1.
        # 2. The other entries must be 0.
        
        # Let's check the 36 valid entries first.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    val = 0.0
                    for r in range(len(U[0])):
                        val += U[I][r] * V[J][r] * W[K][r]
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check a sample of invalid entries? Or all?
        # To be safe, check all.
        # nm=6, mp=18, pn=12. Total 1296? No.
        # nm=2*3=6. mp=3*6=18. pn=6*2=12.
        # 6 * 18 * 12 = 1296.
        # This is small.
        
        # Identify valid (I,J,K)
        valid_ijk = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    valid_ijk.add((I, J, K))
        
        for I in range(n * m):
            for J in range(m * p):
                for K in range(p * n):
                    if (I, J, K) in valid_ijk:
                        continue
                    val = 0.0
                    for r in range(len(U[0])):
                        val += U[I][r] * V[J][r] * W[K][r]
                    if abs(val) > tol:
                        return False
                        
        return True

    # Trivial decomposition
    rank_trivial = n * m * p
    U_triv = [[0] * rank_trivial for _ in range(n * m)]
    V_triv = [[0] * rank_trivial for _ in range(m * p)]
    W_triv = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                U_triv[I][col] = 1
                V_triv[J][col] = 1
                W_triv[K][col] = 1
                col += 1

    # Search for lower rank
    # We try R from 24 to 35
    # Use random initialization for U, V
    # Solve for W
    
    # The system is linear in W if U, V are fixed.
    # T_rec[I,J,K] = sum_r U[I,r] V[J,r] W[K,r]
    # For a fixed I, J, the equation is:
    # sum_r (U[I,r] V[J,r]) W[K,r] = T[I,J,K]
    # This is a system of linear equations for W[:, r]? No.
    # For each K, we have a variable W[K, r] for each r.
    # Actually, W is a matrix of size (pn, R).
    # The equation couples W[K, r] across r for a fixed K? No.
    # For a fixed (I, J), the sum is over r.
    # Let $A_{K, r} = U[I, r] V[J, r]$.
    # Then $\sum_r A_{K, r} W[K, r] = T[I, J, K]$.
    # This equation must hold for all K.
    # So for a fixed (I, J), we have a system of $p*n$ equations (for each K) with $R$ unknowns (the components of W[K, :]? No, W[K, r] is a single unknown for each K, r).
    # Wait, $W$ has entries $W[K, r]$.
    # The equation is $\sum_r A_{K, r} W[K, r] = T[I, J, K]$.
    # This is $p*n$ equations.
    # The unknowns are $W[K, r]$ for all $K, r$.
    # But the equations for different K are decoupled!
    # Equation for K=0: $\sum_r A_{0, r} W[0, r] = T[I, J, 0]$
    # Equation for K=1: $\sum_r A_{1, r} W[1, r] = T[I, J, 1]$
    # ...
    # So for each K, we have a linear system:
    # $A_K \cdot W_K^T = T_K$ where $A_K$ is a row vector of size R?
    # No, $A_{K, r}$ is a scalar.
    # The unknowns are $W[K, 0], \dots, W[K, R-1]$.
    # The equation is a single linear equation in R unknowns.
    # This is underdetermined.
    # We have $n*m$ pairs of (I, J).
    # So we have $n*m * p*n$ equations in total?
    # No, for each (I, J) and each K, we have one equation.
    # Total equations: $n*m * p*n = n^2 m p$.
    # Total unknowns: $p*n*R$.
    # For $n=2, m=3, p=6, R=28$:
    # Equations: $4 * 3 * 6 = 72$.
    # Unknowns: $6 * 2 * 28 = 336$.
    # Highly underdetermined.
    
    # We can pick a random solution for W for each (I, J, K) system?
    # But W[K, r] appears in equations for multiple (I, J).
    # Specifically, $W[K, r]$ appears in the equation for $(I, J, K)$.
    # So the variables are coupled across (I, J).
    # We can set up a global linear system for all W entries.
    # Variables: $W[K, r]$ for $K \in [0, pn), r \in [0, R)$.
    # Equations: For each $I \in [0, nm), J \in [0, mp), K \in [0, pn)$:
    # $\sum_r U[I, r] V[J, r] W[K, r] = T[I, J, K]$
    
    # This is a linear system $M \cdot W_{flat} = T_{flat}$.
    # We can solve this using least squares or null space.
    # Since it's underdetermined, we can find a particular solution.
    # We want the solution to have integer/half-integer entries.
    # This is a "rational solution" problem.
    
    # Strategy:
    # 1. Pick random U, V with integer entries.
    # 2. Build the linear system for W.
    # 3. Solve for W.
    # 4. Check if W is "nice".
    
    # To make W nice, we can scale U, V?
    # If we scale U by 1, V by 1, the system is fixed.
    # The solution W will be rational if U, V are integers.
    # We can check if the denominators are powers of 2.
    
    # Let's try this.
    
    import numpy.linalg as la
    
    for R in range(24, 36):
        if time.time() > deadline:
            break
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Generate random U, V
            # Entries from {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n*m, R))
            V = np.random.randint(-1, 2, size=(m*p, R))
            
            # Build linear system for W
            # Variables: W_flat of size (pn * R)
            # Equations: nm * mp * pn
            # But many T entries are 0.
            # We only need to satisfy the non-zero ones?
            # No, the zero ones must be 0 too.
            # So we include all equations.
            
            # M is (nm*mp*pn, pn*R)
            # This is huge: 6*18*12 = 1296 rows.
            # Cols: 12*R.
            # For R=28, 336 cols.
            # This is a small matrix.
            
            M = np.zeros((n*m*m*p*p*n, p*n*R))
            b = np.zeros(n*m*m*p*p*n)
            
            eq_idx = 0
            for I in range(n*m):
                for J in range(m*p):
                    for K in range(p*n):
                        # Determine T[I,J,K]
                        # Check if (I,J,K) is valid
                        # I = i*m + j, J = j*p + k, K = k*n + i
                        # i = I // m, j = I % m
                        # j2 = J // p, k = J % p
                        # k2 = K // n, i2 = K % n
                        # Valid if i==i2, j==j2, k==k2
                        i = I // m
                        j = I % m
                        j2 = J // p
                        k = J % p
                        k2 = K // n
                        i2 = K % n
                        
                        if i == i2 and j == j2 and k == k2:
                            b[eq_idx] = 1.0
                        else:
                            b[eq_idx] = 0.0
                        
                        # Fill M row
                        # Equation: sum_r U[I,r] V[J,r] W[K,r] = b[eq_idx]
                        # W is stored as W_flat[K*R + r]
                        for r in range(R):
                            M[eq_idx, K*R + r] = U[I, r] * V[J, r]
                        
                        eq_idx += 1
            
            # Solve M * W_flat = b
            # Use lstsq
            W_flat, residuals, rank, s = la.lstsq(M, b)
            
            # Check residual
            if residuals[0] > 1e-9:
                continue
            
            # Reshape W
            W = W_flat.reshape(p*n, R)
            
            # Check if W is nice
            # Convert to list for is_valid_decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if is_valid_decomposition(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return {"U": U_triv, "V": V_triv, "W": W_triv}
# EVOLVE-BLOCK-END
