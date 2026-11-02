"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # The tensor is T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check by constructing the tensor from the factors and comparing to the standard MM tensor.
    # Standard tensor: T_std[i, j, k] = 1 if (i,j,k) is a valid index for the MM operation.
    # Actually, the MM tensor T is defined such that (A*B)[i,k] = sum_j A[i,j] B[j,k].
    # In tensor form, T_{i,j,k} = 1. The decomposition is sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c = T_{i,j,k}.
    # Wait, the standard MM tensor has T_{i,j,k} = 1 for all valid i,j,k? No.
    # The tensor for matrix multiplication of n x m and m x p is a 3-tensor T in R^{n x m x p} where
    # T[i, j, k] = 1. But that's not right either.
    # The bilinear map is (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # This is a trilinear form in the tensor decomposition context?
    # The task says: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r]"
    # The tensor being decomposed is the matrix multiplication tensor.
    # T_{i,j,k} = 1 for all i,j,k? No, that would be the all-ones tensor.
    # The MM tensor T is such that <T, A \otimes B \otimes C^T> = trace(A B C^T) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}.
    # So T_{i,j,k} = 1 for all i,j,k.
    # Wait, if T_{i,j,k} = 1, then the rank is 1? No.
    # The tensor T is in R^{(n*m) x (m*p) x (p*n)}.
    # The entries are T_{(i,j), (j',k), (k',i')} where the indices must match for the term to be non-zero?
    # No, the tensor T is defined on the full space.
    # T_{(i,j), (j',k), (k',i')} = 1 if j == j' and k == k' and i == i', else 0.
    # Yes! That's the identity structure.
    # So T[(i*m+j), (j'*p+k), (k'*n+i')] = 1 if (i==i' and j==j' and k==k') else 0.
    # Let's verify this interpretation.
    # sum_c U[i*m+j, c] V[j'*p+k, c] W[k'*n+i', c] = T[(i*m+j), (j'*p+k), (k'*n+i')]
    # If j==j', k==k', i==i', RHS = 1. Else 0.
    # This is the standard MM tensor.
    
    # Construct the standard tensor in flattened form? It's huge (60 * 30 * 20 = 36000 entries).
    # We can check the reconstruction by evaluating the trilinear form on random inputs.
    # Or better, just check the specific non-zero entries.
    # There are n*m*p = 60 non-zero entries in the tensor, all equal to 1.
    # They are at positions:
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    # For all i in [0,n), j in [0,m), k in [0,p).
    
    # Let's check all 60 non-zero entries and a few zero entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check some zero entries to ensure no spurious non-zeros.
    # Total entries: (n*m) * (m*p) * (p*n) = 60 * 30 * 20 = 36000.
    # Non-zero: 60.
    # We check a random sample of zero entries.
    all_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_indices.add((i * m + j, j * p + k, k * n + i))
    
    # Sample some zero entries
    random.seed(42)
    for _ in range(100):
        u_idx = random.randint(0, n * m - 1)
        v_idx = random.randint(0, m * p - 1)
        w_idx = random.randint(0, p * n - 1)
        if (u_idx, v_idx, w_idx) in all_indices:
            continue
        val = 0.0
        for c in range(rank):
            val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
        if abs(val) > 1e-9:
            return False
    return True

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_based_decomposition(n, m, p):
    """
    Construct a rank-55 decomposition using Strassen-like ideas.
    For 2x3x10, we can try to combine some terms.
    """
    # This is a placeholder for a specific construction if one is known.
    # For now, we fall back to schoolbook if no better construction is found.
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 10s margin
    
    # Start with schoolbook
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using MIQP or other methods.
    # The strategy is to formulate tensor rank minimization as a MIQP.
    # However, a full MIQP over all possible rank-r subsets is intractable for r < 60.
    # Instead, we can try a heuristic: 
    # 1. Assume a rank-r decomposition exists.
    # 2. We know the tensor has 60 non-zero entries.
    # 3. Each rank-1 term (u, v, w) can cover multiple non-zero entries if u, v, w are non-zero at those positions.
    # 4. But the tensor is sparse: T[u, v, w] = 1 iff (u,v,w) is in the set S of 60 indices.
    # 5. A rank-1 term u \otimes v \otimes w has non-zeros at all (u_i, v_j, w_k) where u_i, v_j, w_k are non-zero.
    # 6. For the decomposition to be exact, the sum of rank-1 terms must match T.
    # 7. This means for any (u,v,w) not in S, the sum must be 0.
    # 8. For (u,v,w) in S, the sum must be 1.
    
    # This is a system of polynomial equations.
    # We can try to find a rank-r decomposition by fixing the support of U, V, W.
    # But that's too complex.
    
    # Alternative: Use the fact that n=2.
    # The tensor is T[i, j, k] with i in {0,1}, j in {0,1,2}, k in {0,...,9}.
    # T is non-zero only when i,j,k are consistent with the MM indices.
    # Actually, T[(i*m+j), (j*p+k), (k*n+i)] = 1.
    # Let's denote the indices as:
    # U index: a = i*m + j
    # V index: b = j*p + k
    # W index: c = k*n + i
    # Note that a, b, c are not independent. They are linked by i, j, k.
    # For a given (i,j,k), we have a specific (a,b,c).
    # There are 60 such triples.
    
    # Can we find a rank-54 decomposition?
    # This would require combining 6 of the 60 terms.
    # A rank-1 term (u, v, w) contributes to T[a, b, c] = u_a v_b w_c.
    # If we choose u, v, w to be non-zero at specific positions, we can cover multiple (a,b,c) triples.
    # But the constraint is that for any (a,b,c) not in S, u_a v_b w_c must sum to 0 across all terms.
    
    # This is a hard problem. Let's try a simpler heuristic:
    # Try to find a rank-55 decomposition by merging two terms.
    # Suppose we merge two schoolbook terms (i1,j1,k1) and (i2,j2,k2).
    # We need to find u, v, w such that:
    # u_{a1} v_{b1} w_{c1} + u_{a2} v_{b2} w_{c2} = 1 (if (a1,b1,c1) and (a2,b2,c2) are in S)
    # And the sum is 0 for all other (a,b,c).
    # This is generally not possible with a single rank-1 term unless the terms are "compatible".
    
    # Given the complexity, and the 120s time limit, we will:
    # 1. Return the schoolbook decomposition (rank 60).
    # 2. If time permits, try a few random perturbations or structured merges.
    
    # For now, we return the schoolbook decomposition.
    # A future improvement could involve a more sophisticated search.
    
    return best_result
# EVOLVE-BLOCK-END
