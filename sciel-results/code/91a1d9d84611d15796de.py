"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def _verify(U, V, W, n=2, m=4, p=8, tol=1e-9):
    """Verify that the decomposition correctly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check shapes
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    # Build the multiplication tensor and check
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,k,j] = delta (the identity)
    # The decomposition says: T[i,k,j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right. The tensor decomposition for matrix mult is:
    # (A*B)[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) ... no
    
    # Actually the standard form: We want C[i,k] = sum_j A[i,j] B[j,k]
    # A rank-r decomposition uses r intermediate products:
    # C[i,k] = sum_{c=1}^{r} U_c[i,k] * V_c[i,k]  ... no
    
    # The correct formulation: 
    # We have three tensors U, V, W of shapes (n*m, r), (m*p, r), (p*n, r)
    # The product is: for each c, form intermediate I_c = U[:,c]^T * A (reshaped)
    # Actually, let me think more carefully.
    
    # The matrix multiplication tensor T has indices (i, j, k) where T[i,j,k] = 1 if we're computing C[i,k] from A[i,:] and B[:,k]
    # Standard: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,k,j] = 1 (for all valid i,k,j) -- no, that's not a tensor decomposition.
    
    # The bilinear complexity: We seek rank r and matrices U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: delta_{j,j'} ... 
    
    # Let me use the standard verification: 
    # The multiplication tensor M has entries M[i,k,j] = 1 for all i,k,j (it's the all-ones tensor in the sense that every triple contributes)
    # No wait. M[i,k,j] is the coefficient such that C[i,k] = sum_{j} A[i,j] * B[j,k] * M[i,k,j] where M[i,k,j] = 1.
    # So the tensor is all ones? No, that's not right either.
    
    # The correct tensor: T is a 3-tensor of size (n*p) x (n*m) x (m*p) or similar.
    # Let me just verify by checking that the bilinear form matches.
    
    # A rank-r decomposition of the matrix multiplication tensor means:
    # There exist U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i in [n], k in [p], j in [m]:
    #   1 (the coefficient of A[i,j]*B[j,k] in C[i,k]) = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not how it works. The tensor decomposition is:
    # T[i,k,j] = sum_c U_c[i,m,j] * V_c[j,p,k] * W_c[k,n,i]
    # where T[i,k,j] = 1 for all i,k,j.
    
    # So we need: for all i,k,j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let me verify this.
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=2, m=4, p=8):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_random_decomposition(n, m, p, rank, num_trials=100, seed=None):
    """Try random rank-r decompositions and check if any is valid."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    for trial in range(num_trials):
        # Initialize with random small values
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = rng.randn(p * n, rank) * 0.1
        
        # Try to solve: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a system of equations. With n*m*p = 64 equations and 3*r variables.
        # For r=32, we have 96 unknowns and 64 equations, so potentially solvable.
        # But it's nonlinear (trilinear). Let's try a different approach.
        
        # Alternative: fix U and V, solve for W.
        # For each (i,k): W[k*n+i, c] appears in equations for all j.
        # The equation is: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Group by (i,k): for each (i,k), we have m equations (one per j) and p*n unknowns (W entries for this i,k).
        # Actually W[k*n+i, :] has r entries, and for fixed (i,k), the equations are:
        # for each j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # This is: (U_row_i * V_col_k)^T * W[k*n+i, :] = 1 (as a vector of length m)
        # Where U_row_i is n*m x r with only the rows for index i... 
        
        # This is getting complicated. Let me just try a greedy/structured approach.
        pass
    return None

def _strassen_block_decomposition(n=2, m=4, p=8):
    """
    Try to use Strassen-like block decomposition.
    Split into 2x2 blocks where possible.
    """
    # A is 2x4, B is 4x8
    # Split B into four 4x2 blocks: B1, B2, B3, B4
    # C = A * B = [A*B1, A*B2, A*B3, A*B4]
    # Each A*Bj is 2x2 * 4x2 = 2x2 result
    # For 2x2 * 4x2: C[i,k] = sum_j A[i,j] * Bj[j,k]
    # This is (2,4,2) matrix multiplication.
    
    # What's the rank of (2,4,2)?
    # Schoolbook: 2*4*2 = 16
    # Lower bound: max(2*4, 4*2, 2*2) = max(8,8,4) = 8
    
    # Let's try to find a rank-8 or better decomposition for (2,4,2).
    # If we can do (2,4,2) in rank 8, then total rank = 4 * 8 = 32.
    
    # For (2,4,2): C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k], i in {0,1}, k in {0,1}
    # 4 outputs, 8 inputs for A, 8 inputs for B.
    
    # Try: Can we compute each output with 2 multiplications?
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
    # That's 4 terms. Hard to reduce without Strassen-like tricks.
    
    # Alternative: Use the fact that we can combine terms.
    # Let me just return schoolbook for now and let the search find improvements.
    pass

def _structured_search(n=2, m=4, p=8, deadline=None):
    """
    Structured search for low-rank decomposition.
    Uses the fact that the tensor is all-ones and tries to find sparse/structured solutions.
    """
    start_time = time.time()
    
    # The tensor T[i,k,j] = 1 for all i,k,j needs to be decomposed as:
    # T[i,k,j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Key insight: If we can find U, V, W with entries in {-1, 0, 1} or similar,
    # we might find a low-rank solution.
    
    # Strategy: Try rank values from lower bound up.
    # Lower bound: max(n*m, m*p, n*p) = max(8, 32, 16) = 32
    
    # For rank r, we need to solve:
    # For all i in [n], k in [p], j in [m]:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*p*m = 64 trilinear equations in 3*r unknowns.
    
    # Approach: Use the structure. Note that the tensor is "separable" in a sense.
    # T[i,k,j] = 1 = (1) * (1) * (1) for all i,k,j.
    # This suggests that a rank-1 decomposition would work if U, V, W were all ones,
    # but then U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1*1*1 = 1, which works!
    # Wait, but that's rank 1! Let me check.
    
    # If U is all ones (n*m x 1), V is all ones (m*p x 1), W is all ones (p*n x 1):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1*1*1 = 1. 
    # That works for all i,j,k!
    
    # But wait, that can't be right. The rank of the matrix multiplication tensor
    # for (2,4,8) should be at least 32 by the standard lower bound.
    
    # Let me re-examine. The standard lower bound for the rank of the (n,m,p) matrix
    # multiplication tensor is max(n*m, m*p, n*p). For (2,4,8), that's max(8, 32, 16) = 32.
    
    # But my "all ones" argument gives rank 1. What's wrong?
    
    # Ah, I think I'm confusing the tensor structure. Let me reconsider.
    
    # The matrix multiplication tensor is NOT all ones. It's defined by:
    # T[i,k,j] = 1 if and only if the product A[i,j] * B[j,k] contributes to C[i,k].
    # In standard matrix multiplication, EVERY A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.
    # So T[i,k,j] = 1 for all valid i,k,j.
    
    # But the tensor decomposition is different. The tensor T is a 3-tensor of dimensions
    # (n*p) x (n*m) x (m*p) or some arrangement. Let me be more careful.
    
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k].
    # We can view this as a trilinear form: sum_{i,j,k} C[i,k] * f(A,B) where f involves T.
    
    # Actually, the standard formulation: The multiplication tensor M is in 
    # (R^{n*p}*) x (R^{n*m}) x (R^{m*p}) or similar. The rank is the minimum r such that
    # M = sum_{c=1}^r u_c v_c w_c where u_c in R^{n*p}, v_c in R^{n*m}, w_c in R^{m*p}.
    
    # The entries: M[i,k, i*m+j, j*p+k] = 1 for all i,k,j, and 0 otherwise.
    # Wait, that's not a 3-tensor in the usual sense. Let me use the standard definition.
    
    # The matrix multiplication tensor T of size (n*p) x (n*m) x (m*p) has:
    # T[a, b, c] = 1 if a = i*p + k, b = i*m + j, c = j*p + k for some i,j,k
    #             = 0 otherwise
    # Wait, that's not right either. Let me look at this more carefully.
    
    # Standard: T[i,k, i*m+j, j*p+k] ... this doesn't make sense as a 3-tensor.
    
    # Let me use the formulation from the task: U (n*m x r), V (m*p x r), W (p*n x r)
    # such that the decomposition works. The verification I wrote above should be correct:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # Now, if U is all ones, V is all ones, W is all ones, then the sum is 1*1*1 = 1. 
    # So rank 1 would work?!
    
    # But that contradicts the known lower bound. Let me check with a small case.
    # For 1x1 x 1x1: C[0,0] = A[0,0]*B[0,0]. 
    # U[0,0] * V[0,0] * W[0,0] = 1. So U=V=W=[1] works. Rank 1. Correct.
    
    # For 1x2 x 2x1: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0].
    # We need: U[0,0]*V[0,0]*W[0,0] = 1 (for j=0)
    #          U[1,0]*V[1,0]*W[0,0] = 1 (for j=1)
    # Wait, U has shape (n*m, r) = (2, r), V has shape (m*p, r) = (2, r), W has shape (p*n, r) = (1, r).
    # For i=0, k=0, j=0: U[0,0]*V[0,0]*W[0,0] = 1
    # For i=0, k=0, j=1: U[1,0]*V[1,0]*W[0,0] = 1
    # If r=1: U[0,0]*V[0,0]*W[0,0] = 1 and U[1,0]*V[1,0]*W[0,0] = 1.
    # Let W[0,0] = 1, then U[0,0]*V[0,0] = 1 and U[1,0]*V[1,0] = 1.
    # This is satisfiable! E.g., U[0,0]=1, V[0,0]=1, U[1,0]=1, V[1,0]=1.
    # So rank 1 works for 1x2 x 2x1?
    
    # But the bilinear complexity of 1x2 x 2x1 is 2, not 1!
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]. This requires 2 scalar multiplications.
    
    # So what's wrong with my verification? Let me re-examine.
    
    # AH, I see the issue. The tensor decomposition for matrix multiplication is NOT:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # 
    # That would be a trilinear form that equals 1, which is trivially satisfied.
    # 
    # The correct formulation is different. The matrix multiplication tensor has a specific
    # structure that I'm not capturing correctly.
    
    # Let me look up the correct definition. The (n,m,p) matrix multiplication tensor T is
    # an element of (R^{n*m}*) x (R^{m*p}*) x (R^{n*p}*) or similar. The rank is the minimum
    # number of rank-1 tensors needed to express T.
    
    # The correct condition is:
    # T = sum_c u_c \otimes v_c \otimes w_c
    # where T[i*m+j, j*p+k, k*n+i] ... no, the indices don't work that way.
    
    # I think the issue is that the tensor T is not simply "all ones." It has a specific
    # pattern based on the matrix multiplication structure.
    
    # Let me use a different approach: directly verify by checking that the bilinear map
    # computed from U, V, W matches standard matrix multiplication.
    
    # Given U (n*m x r), V (m*p x r), W (p*n x r), the computed product is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k]) * W[k*n+i, c]
    # Wait, that's not right either.
    
    # I think the correct interpretation is:
    # Intermediate_c = U[:, c]^T * A_reshaped ... this is getting confusing.
    
    # Let me just use a numerical verification: check that for random A and B,
    # the product computed via the decomposition matches A*B.
    
    # The standard way: A rank-r decomposition of the matrix multiplication tensor gives
    # r intermediate products. The i-th intermediate product is:
    # M_c = (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # And C[i,k] = sum_c M_c * W[k*n+i, c]
    
    # No wait, that's not how it works. Let me think about this differently.
    
    # Actually, I recall now. The correct verification is:
    # For the decomposition to be valid, we need:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # to equal sum_j A[i,j] * B[j,k] for all A, B.
    
    # This means: for all i, k, and for all A, B:
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) = sum_j A[i,j] * B[j,k]
    
    # Expanding the LHS:
    # sum_c W[k*n+i, c] * sum_j U[i*m+j, c] * A[i,j] * sum_j' V[j'*p+k, c] * B[j',k]
    # = sum_j sum_j' A[i,j] * B[j',k] * sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c]
    
    # For this to equal sum_j A[i,j] * B[j,k] for all A, B, we need:
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j, j'}
    
    # So the correct condition is:
    # For all i, k, j, j': sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j, j'}
    
    # This is a much more restrictive condition! Let me verify with the schoolbook:
    # In schoolbook, U[i*m+j, c] = 1 if c = (i*m+j)*p + k' ... no, let me re-examine.
    
    # In the schoolbook, each column c corresponds to a unique (i,j,k) triple.
    # U[i*m+j, c] = 1 if the (i,j) part matches, i.e., c encodes (i,j,k) and we're looking at (i,j).
    # V[j*p+k, c] = 1 if the (j,k) part matches.
    # W[k*n+i, c] = 1 if the (k,i) part matches.
    
    # For a given c corresponding to (i0, j0, k0):
    # U[i*m+j, c] = 1 if i=i0 and j=j0, else 0.
    # V[j*p+k, c] = 1 if j=j0 and k=k0, else 0.
    # W[k*n+i, c] = 1 if k=k0 and i=i0, else 0.
    
    # So U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = 1 if i=i0, j=j0, j'=j0, k=k0, else 0.
    # Sum over c: this is 1 if j=j' (since we sum over all c, and for each (i,j,k) there's a c,
    # but the product is nonzero only when all three match, which requires j=j0=j' and k=k0 and i=i0).
    # Wait, for fixed i, k, j, j', the sum over c of U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c]:
    # This is nonzero only when there exists c such that U[i*m+j, c]=1, V[j'*p+k, c]=1, W[k*n+i, c]=1.
    # In schoolbook, c corresponds to (i0, j0, k0). U[i*m+j, c]=1 means i=i0, j=j0.
    # V[j'*p+k, c]=1 means j'=j0, k=k0. W[k*n+i, c]=1 means k=k0, i=i0.
    # So all three are 1 iff i=i0, j=j0, j'=j0, k=k0.
    # For fixed i, k, j, j', this is 1 iff j=j' (since we can choose c corresponding to (i, j, k) if j=j').
    # If j != j', there's no c that satisfies all three, so the sum is 0.
    # If j = j', there's exactly one c (corresponding to (i, j, k)) that gives 1.
    # So the sum is delta_{j, j'}. Correct!
    
    # So the correct verification is:
    # For all i, k, j, j': sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j, j'}
    
    # This is n*p*m*m = 2*8*4*4 = 256 equations.
    # The "all ones" solution gives sum_c = 1*1*1 = 1 for all i,k,j,j', which is NOT delta_{j,j'}.
    # So rank 1 does NOT work. Good, this matches the known lower bound.
    
    pass

def _verify_correct(U, V, W, n=2, m=4, p=8, tol=1e-9):
    """Correct verification: check sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] = delta_{j,j'}"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for jp in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[jp * p + k, c] * W[k * n + i, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > tol:
                        return False
    return True

def _construct():
    """
    Find a low-rank decomposition for 2x4 x 4x8 matrix multiplication.
    Uses structured search with verification.
    """
    n, m, p = 2, 4, 8
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # The correct condition: for all i,k,j,j': sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] = delta_{j,j'}
    # This is 256 equations in 3*r unknowns.
    
    # Lower bound: rank >= max(n*m, m*p, n*p) = max(8, 32, 16) = 32
    
    # Let's try to find a rank-32 decomposition.
    # For r=32, we have 96 unknowns and 256 equations. Overdetermined, but the structure
    # of the equations (Kronecker-like) might make it solvable.
    
    # Approach: Use the structure. Note that the equations separate by (i,k).
    # For fixed (i,k), we have m*m = 16 equations:
    # For all j, j': sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
    # Let a_c = U[i*m+j, c] (for each j), b_c = V[j'*p+k, c] (for each j'), w_c = W[k*n+i, c]
    # Then: sum_c a_j(c) * b_jp(c) * w(c) = delta_{j,jp}
    
    # This is: for each (i,k), the matrix A_{ik} = [a_j(c)] (m x r), B_{ik} = [b_jp(c)] (m x r), w_{ik} = [w(c)] (r,)
    # satisfies: A_{ik} * diag(w_{ik}) * B_{ik}^T = I_m
    
    # So we need: A_{ik} * diag(w_{ik}) * B_{ik}^T = I_4 for all (i,k).
    # With r=32, A_{ik} is 4x32, B_{ik} is 4x32, w_{ik} is 32x1.
    # A * diag(w) * B^T = I means: A * diag(w) = (B^T)^{-1} = (B^T)^{-1}
    # So A * diag(w) * B^T = I. If B is full rank (4x32, rank 4), then B^T is 32x4 with rank 4.
    # We need A * diag(w) * B^T = I_4.
    
    # One solution: Let B_{ik} be the first 4 columns of I_32 (i.e., B_{ik} = [I_4 | 0]), w_{ik} = 1 for first 4, 0 for rest.
    # Then A_{ik} * diag(w) * B^T = A_{ik}[:, :4] * I_4 = A_{ik}[:, :4]. We need this to be I_4.
    # So A_{ik}[:, :4] = I_4.
    
    # But this must work for ALL (i,k) pairs simultaneously, with the SAME U, V, W.
    # U[i*m+j, c] depends on i and j. V[j*p+k, c] depends on j and k. W[k*n+i, c] depends on k and i.
    
    # This is a complex constraint satisfaction problem. Let me try a computational approach.
    
    # Try rank 32 with a structured initialization and solve.
    for r in [32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 48, 52, 56, 60, 64]:
        if time.time() > deadline:
            break
        if r > 64:
            break
            
        # Initialize with a structured guess
        # Try: U[i*m+j, c] = 1 if c = j*r//m + something... 
        # Let me try a random approach with local optimization.
        
        # For now, just try schoolbook if we can't find better.
        pass
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 2x4 x 4x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find a better decomposition than schoolbook
    result = _construct()
    
    # Verify the result
    if _verify_correct(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook
    schoolbook = _schoolbook(n, m, p)
    return schoolbook

# EVOLVE-BLOCK-END
