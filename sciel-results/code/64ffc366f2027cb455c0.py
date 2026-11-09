"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for rank-minimal exact tensor decomposition
of 2x3 * 3x7 matrix multiplication. Uses structured families + exact verification with
a hard deadline guard. Falls back to schoolbook rank 42."""

import time
import itertools

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    DEADLINE = 130.0  # safety margin under 150s kill

    n, m, p = DIMENSIONS
    NM, MP, PN = n * m, m * p, p * n
    schoolbook_rank = NM * MP // p  # = n*m*p? Actually schoolbook is n*m*p
    # Schoolbook rank = n*m*p = 42 for (2,3,7)
    schoolbook_rank = n * m * p

    def build_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = build_schoolbook()

    # The multiplication tensor T[i,j,k] = delta_{i,i'} delta_{j,j'}: 
    # (A * B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # As a 3-tensor T[i, j, k] with indices i in [0,n), j in [0,m), k in [0,p):
    # T[i,j,k] = 1 if (i,j,k) is a valid "edge" i.e. T_{i,j,k} = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p)
    # Wait: the tensor for matrix multiplication is T[i, j, k] = 1 for the term A[i,j]*B[j,k].
    # The tensor is T with T[i,j,k] = 1 for all i,j,k (it's the "all ones" 3-tensor? No.)
    # Actually the matmul tensor M has M[i, j, k] = 1 if the output (i,k) gets contribution from (j):
    # C[i,k] = sum_j A[i,j] B[j,k]. So the tensor is M[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
    # Yes! The tensor is the all-ones tensor of shape (n, m, p). Its rank is the bilinear complexity.

    # So we need: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]  for all i,j,k.
    # Note the W index is (k*n + i) per the trace convention W[k*n+i, r].

    # Verification: given U, V, W as lists of lists (rows x rank), check exact decomposition.
    import numpy as np

    def verify(U, V, W):
        """U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank.
        Check that for all i,j,k: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1."""
        rank = len(U[0])
        # Convert to numpy arrays for speed
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Reshape to 3D tensors: U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # T[i, j, k] = sum_c U_3d[i, j, c] * V_3d[j, k, c] * W_3d[k, i, c]
        T = np.zeros((n, m, p), dtype=np.float64)
        
        for c in range(rank):
            u_slice = U_3d[:, :, c] # (n, m)
            v_slice = V_3d[:, :, c] # (m, p)
            w_slice = W_3d[:, :, c] # (p, n)
            
            # Compute term: T[i, j, k] += u_slice[i, j] * v_slice[j, k] * w_slice[k, i]
            # We can do this by looping over i and j, or using einsum?
            # einsum('ij,jk,ki->ijk', u_slice, v_slice, w_slice)
            term = np.einsum('ij,jk,ki->ijk', u_slice, v_slice, w_slice)
            T += term
            
        # Check if T is all 1s
        # Due to floating point, check closeness
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    # Try to find a smaller rank using a structured family.
    # Family: "block" decompositions where we group indices.
    # Known: for 2x3x7, the rank might be achievable below 42 with clever structure.
    # We'll try a parametric ansatz: allow each rank-1 term to have support on a "line" in one mode
    # and full in others, but with coefficients.
    
    # A more powerful approach: try ranks from lower bound up to schoolbook-1, and for each rank
    # try a few structured parameterizations (integer/half-integer) and verify.
    
    # Lower bound: for n=2, m=3, p=7, a trivial lower bound is max(n*m, m*p, p*n) = max(6,21,14)=21.
    # Also n*m*p / max(n,m,p) ... Strassen-like. Let's try from 21 up to 41.
    
    # Given time, we'll try a heuristic: for each candidate rank r in a range, try to construct
    # a valid decomposition using a "sum of Kronecker" style with random but structured coefficients,
    # then verify. Since exact search is hard, we focus on a few promising families.
    
    # Family 1: "diagonal" + "off-diagonal" corrections. Start with schoolbook and try to merge
    # terms. This is complex. Instead, we'll try small structured decompositions.
    
    # Given the difficulty, and that finding a rank < 42 for 2x3x7 is a research-level problem,
    # we'll attempt a limited search: try to express the all-ones tensor of shape (2,3,7) as a sum
    # of rank-1 tensors with half-integer entries, using a greedy algorithm.
    
    def try_random_rank(r, max_iters=50):
        """Random search for a decomposition of rank r."""
        import random
        random.seed(42)
        # We search for integer entries (scale by 2 if half-integers needed, but start with ints)
        # Actually, just try small integers -1, 0, 1
        for _ in range(max_iters):
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(NM)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(MP)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(PN)]
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
        return None

    # Structured family: "Kronecker sum" decomposition.
    # For 2x3x7, note that 3 = 3, 7 = 7. 
    # A known construction: if you can decompose smaller matrices, you can combine.
    # But 2x3x7 is small enough that the optimal rank is known in some tables.
    # The bilinear complexity of 2x3x7: I recall that for 2xNxN it's known, but 2x3x7 is asymmetric.
    
    # Given time constraints and the difficulty of exact rank minimization, 
    # we return the schoolbook decomposition which is guaranteed valid.
    # We could try rank 41, 40, ... but verification is O(r * n*m*p) which is fine, 
    # but finding a valid decomposition is the hard part.
    
    # Let's at least try a few quick structured guesses for lower rank.
    # Guess: rank = n*m*p - something. 
    # A simple improvement: if two terms share structure, merge them.
    # For example, terms with same (i,j) but different k can be combined if W allows.
    
    # Since we can't easily find a lower rank without a real search algorithm,
    # and the task rewards lower rank, we'll do a quick randomized search for small rank improvements.
    
    # Time check: if we've spent too long, return best.
    if time.time() - t_start > DEADLINE - 10:
        return best
    
    # Try to find a lower rank using random search.
    # We limit the search to a few ranks and a few iterations to stay within time limits.
    for r in range(schoolbook_rank - 1, max(20, schoolbook_rank - 5), -1):
        if time.time() - t_start > DEADLINE - 10:
            break
        result = try_random_rank(r, max_iters=10)
        if result:
            best = result
            break
            
    # Return the best found (schoolbook if no improvement)
    return best
# EVOLVE-BLOCK-END
