"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # The tensor T[i,j,k] for matrix multiplication (A is n x m, B is m x p, C is n x p)
    # is defined such that C[i,k] = sum_j A[i,j] B[j,k].
    # In tensor decomposition form: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # Wait, the standard decomposition for matrix multiplication tensor M_{n,m,p} is:
    # M_{i,j,k} = delta_{j,k} is NOT correct for the tensor definition usually.
    # The multiplication tensor T has entries T_{i,j,k} such that:
    # (A B)_{i,k} = sum_{j} A_{i,j} B_{j,k} = sum_{j} A_{i,j} B_{j,k}
    # This corresponds to a 3-way tensor T where T[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # Actually, the tensor is T[i, j, k] = 1 if we are computing C[i,k] using A[i,j] and B[j,k].
    # So T[i, j, k] is non-zero only for specific i,j,k? No.
    # The tensor T is indexed by (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # The entry T[i,j,k] is 1 if the product A[i,j] * B[j,k] contributes to C[i,k].
    # This is always true for the specific indices.
    # However, the decomposition is:
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # For the schoolbook algorithm, we have one term for each (i,j,k) triple?
    # No, the schoolbook algorithm computes C[i,k] = sum_j A[i,j] B[j,k].
    # This is a sum of m terms for each (i,k).
    # The tensor T is sparse. T[i,j,k] is 1 if... wait.
    # The tensor T is defined such that the bilinear map (A,B) -> C is represented.
    # C_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k} ?? No.
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # So the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    # Thus, the tensor T has T[i,j,k] = 1 for all i,j,k? No, that would mean C_{i,k} gets contribution from all j for all i,k.
    # Yes, for a fixed (i,k), we sum over j.
    # So T[i,j,k] is 1 for all i in [0,n), j in [0,m), k in [0,p)?
    # No, the indices are coupled.
    # The tensor T is of size n x m x p.
    # The entry T[i,j,k] is 1 if the term A[i,j] B[j,k] is part of the computation for C[i,k].
    # This is true for all i,j,k.
    # So T is the all-ones tensor?
    # If T is all-ones, then rank(T) is 1? No.
    # Let's check the definition carefully.
    # Matrix multiplication tensor M_{n,m,p} is defined by M_{i,j,k} = 1 if the indices match?
    # Actually, the standard definition is that the tensor T is such that:
    # T = sum_{i,j,k} e_i^A \otimes e_j^B \otimes e_k^C * delta_{j,k} ? No.
    # Let's look at the schoolbook decomposition provided in the seed.
    # Seed code:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       U[i*m+j, col] = 1
    #       V[j*p+k, col] = 1
    #       W[k*n+i, col] = 1
    # This implies that for each (i,j,k), we have a rank-1 term.
    # The term is U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    # The product is A_{i,j} * B_{j,k} * C_{k,i}??
    # The output is W. W is (p*n, rank).
    # The index in W is k*n + i.
    # So the r-th term contributes to W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r].
    # If U,V,W are 1s for that column, the contribution to C[k*n+i] (which is C[i,k] in some ordering) is A[i,j] B[j,k].
    # So the sum over r of U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] must equal 1 for the specific (i,j,k) combination that generates the term, and 0 otherwise?
    # No, the bilinear form is:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # In the decomposition:
    # C_{i,k} = sum_{r} W_{k*n+i, r} (sum_{j} U_{i*m+j, r} A_{i,j}) (sum_{j'} V_{j'*p+k, r} B_{j',k})
    # This is not quite right.
    # The standard form is:
    # C_{i,k} = sum_{r} W_{i,k,r} (sum_{j} U_{i,j,r} A_{i,j}) (sum_{j'} V_{j',k,r} B_{j',k})
    # Here, U is (n*m, rank), V is (m*p, rank), W is (n*p, rank).
    # But the task says W is (p*n, rank) with index k*n+i.
    # So W index is (k,i).
    # U index is (i,j).
    # V index is (j,k).
    # The term is sum_r W_{k,i,r} * (sum_j U_{i,j,r} A_{i,j}) * (sum_j' V_{j',k,r} B_{j',k})
    # For this to equal sum_j A_{i,j} B_{j,k}, we need:
    # sum_r W_{k,i,r} U_{i,j,r} V_{j,k,r} = delta_{j,j} ? No.
    # We need sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r} to be 1 if j=j' and (i,k) matches, 0 otherwise.
    # Specifically, for fixed i,k, we want sum_r W_{k,i,r} U_{i,j,r} V_{j,k,r} = 1 for all j?
    # No, the sum over j is outside.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Decomposition: C_{i,k} = sum_r W_{k,i,r} (sum_j U_{i,j,r} A_{i,j}) (sum_j' V_{j',k,r} B_{j',k})
    # This expands to sum_r sum_j sum_j' W_{k,i,r} U_{i,j,r} V_{j',k,r} A_{i,j} B_{j',k}
    # We want this to equal sum_j A_{i,j} B_{j,k}.
    # So we need: sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r} = delta_{j,j'} for all i,k.
    # Wait, the RHS is sum_j A_{i,j} B_{j,k}. The coefficient of A_{i,j} B_{j',k} is 1 if j=j', 0 otherwise.
    # So yes, sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r} = delta_{j,j'}.
    
    # This is a system of linear equations if we fix one matrix.
    # If we fix W, we have a linear system for U and V? No, it's bilinear in U and V.
    # But if we fix V, it's linear in U?
    # sum_r W_{k,i,r} U_{i,j,r} V_{j',k,r} = delta_{j,j'}
    # Let M_{j,j'} = delta_{j,j'}.
    # This is equivalent to: for each (i,k), the matrix sum_r W_{k,i,r} u_{i,r} v_{k,r}^T = I_m.
    # Where u_{i,r} is the row i of U (length m), and v_{k,r} is the column k of V (length m)?
    # U is (n*m, rank). Row i*m+j is U[i*m+j, r].
    # So u_{i,r} is a vector of length m with components U[i*m+j, r] for j in 0..m-1.
    # V is (m*p, rank). Column k is V[j*p+k, r] for j in 0..m-1.
    # So v_{k,r} is a vector of length m with components V[j*p+k, r] for j in 0..m-1.
    
    # The condition is: sum_r W_{k,i,r} u_{i,r} v_{k,r}^T = I_m for all i,k.
    
    # If we fix W, this is a set of quadratic equations for U and V.
    # However, if we fix V to be an identity-like structure, it might simplify.
    # Inspiration 2 suggested fixing V to identity-like.
    
    # Let's try to fix V.
    # If we choose V such that v_{k,r} are orthogonal or simple, can we solve for U and W?
    
    # Consider the case where rank r = m*p = 21.
    # Can we set V to be the identity matrix of size 21?
    # V is (m*p, rank). If rank=21, V is 21x21.
    # Let V be the identity matrix I_21.
    # Then v_{k,r} is the column r of I_21?
    # V[j*p+k, r].
    # If V is I_21, then V[j*p+k, r] = 1 if j*p+k == r, else 0.
    # So for a fixed k, v_{k,r} is non-zero only if r = j*p+k for some j.
    # Specifically, v_{k,r} is a vector of length m.
    # v_{k,r}[j] = V[j*p+k, r] = delta_{j*p+k, r}.
    # So v_{k,r} is the unit vector e_j if r = j*p+k, else 0.
    
    # Substitute into the condition:
    # sum_r W_{k,i,r} u_{i,r} v_{k,r}^T = I_m
    # sum_{r: v_{k,r} != 0} W_{k,i,r} u_{i,r} v_{k,r}^T = I_m
    # The non-zero v_{k,r} are for r = j*p+k.
    # For such r, v_{k,r} = e_j.
    # So the sum is over j:
    # sum_{j=0}^{m-1} W_{k,i, j*p+k} u_{i, j*p+k} e_j e_j^T = I_m
    # This implies that for each j:
    # W_{k,i, j*p+k} u_{i, j*p+k} = 1
    # And for j != j', the terms are diagonal, so they don't interfere.
    # So we need W_{k,i, j*p+k} u_{i, j*p+k} = 1 for all i,k,j.
    
    # This can be satisfied by setting:
    # W_{k,i, j*p+k} = 1
    # u_{i, j*p+k} = 1
    # i.e., U[i*m+j, j*p+k] = 1?
    # Wait, u_{i,r} is the vector (U[i*m+0, r], ..., U[i*m+m-1, r]).
    # So u_{i, r}[j] = U[i*m+j, r].
    # We need u_{i, j*p+k}[j] = 1?
    # No, u_{i, r} is a vector. The term is u_{i,r} v_{k,r}^T.
    # If v_{k,r} = e_j, then u_{i,r} e_j^T is a matrix with column j equal to u_{i,r} and other columns 0.
    # The sum over r (which maps to j) gives:
    # sum_j W_{k,i, j*p+k} (u_{i, j*p+k} e_j^T)
    # This is a matrix where column j is W_{k,i, j*p+k} u_{i, j*p+k}.
    # We want this matrix to be I_m.
    # So column j of I_m is e_j.
    # So we need W_{k,i, j*p+k} u_{i, j*p+k} = e_j.
    # This means:
    # For all l:
    # W_{k,i, j*p+k} u_{i, j*p+k}[l] = delta_{j,l}
    # If l != j, u_{i, j*p+k}[l] must be 0 (or W is 0, but W is scalar).
    # If l == j, W_{k,i, j*p+k} u_{i, j*p+k}[j] = 1.
    
    # So we can choose:
    # U[i*m+l, j*p+k] = 0 if l != j
    # U[i*m+j, j*p+k] = 1
    # W[k*n+i, j*p+k] = 1
    
    # Let's check if this works.
    # U is (n*m, m*p).
    # V is (m*p, m*p) = I.
    # W is (p*n, m*p).
    
    # Rank is m*p = 21.
    # This is a valid decomposition if the indices are correct.
    
    # Let's construct this decomposition.
    
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Set V to identity
    for idx in range(m * p):
        V[idx][idx] = 1
    
    # Set U and W
    # U[i*m+l, j*p+k] = delta_{l,j}
    # W[k*n+i, j*p+k] = 1
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                col = j * p + k
                # U
                # l must be j
                U[i * m + j][col] = 1
                # W
                W[k * n + i][col] = 1
    
    # Verify this decomposition
    # We need sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j' V[j'*p+k, r] B[j',k]) = sum_j A[i,j] B[j,k]
    
    # LHS = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j' V[j'*p+k, r] B[j',k])
    
    # Since V is identity, V[j'*p+k, r] = 1 if j'*p+k == r, else 0.
    # So sum_j' V[j'*p+k, r] B[j',k] = B[j,k] if r = j*p+k, else 0.
    # So the term is non-zero only if r = j*p+k for some j.
    # Let r = j*p+k.
    # Then the second factor is B[j,k].
    # The first factor is sum_l U[i*m+l, j*p+k] A[i,l].
    # From our construction, U[i*m+l, j*p+k] = 1 if l=j, else 0.
    # So sum_l ... = A[i,j].
    # The third factor W[k*n+i, j*p+k] = 1.
    
    # So the sum over r becomes sum_j (1 * A[i,j] * B[j,k]) = sum_j A[i,j] B[j,k].
    # This matches the RHS.
    
    # So rank 21 is achievable!
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
