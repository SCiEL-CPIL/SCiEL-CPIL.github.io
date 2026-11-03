"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 2, 3, 6
    schoolbook_rank = n * m * p  # 36

    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k])
                    # Wait, that's not right. The tensor equation is:
                    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                    # This is NOT a simple dot product of U and V rows.
                    # Let me redo this properly.
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = schoolbook_rank
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return U, V, W, rank

    # Key insight: For matrix multiplication A*B where A is n x m and B is m x p,
    # the tensor is (i,j,k) -> delta(i,j,k) in the sense that C[i,k] = sum_j A[i,j]*B[j,k].
    # The tensor decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all (i,j,k), and 0 otherwise... wait, actually for matrix multiplication,
    # every triple (i,j,k) contributes exactly A[i,j]*B[j,k] to C[i,k].
    # So the tensor has T[i,j,k] = 1 for ALL i,j,k (since we need the coefficient to be 1).
    # Wait no. Let me re-read the task.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor M has M[i,j,k] = 1 if and only if... 
    # Actually, for the bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k],
    # the tensor is: T[i,k,j] = 1 for all i,j,k.
    # In the convention W[k*n+i, r], the tensor entry is T[i,j,k] and we need:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 for all valid (i,j,k).
    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

    # This is a 3-way tensor of all ones (size n x m x p = 2 x 3 x 6 = 36 entries, all equal to 1).
    # We need to decompose the all-ones tensor of size 2x3x6 into a sum of rank-1 tensors.
    # The rank of the all-ones tensor of size a x b x c is at most... 
    # For the all-ones tensor, one can always use rank = a*b*c (schoolbook).
    # But can we do better?
    
    # For an all-ones tensor of size a x b x c, the rank is known to be at most a*b (or similar)
    # if we allow certain structures. Let me think...
    
    # Actually, the all-ones tensor T[i,j,k] = 1 for all i,j,k can be written as:
    # T[i,j,k] = (1)(1)(1) -- that's rank 1! But wait, that would mean U[i,j] = 1, V[j,k] = 1, W[k,i] = 1.
    # Let me check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # If rank = 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then the sum = 1*1*1 = 1. Yes! That works!
    
    # Wait, that can't be right. Let me re-examine.
    # If U[i*m+j, 0] = 1 for all i,j, V[j*p+k, 0] = 1 for all j,k, W[k*n+i, 0] = 1 for all k,i,
    # then for any (i,j,k): sum = 1*1*1 = 1. Yes, it works!
    
    # But wait, this would mean the rank of any matrix multiplication is 1, which is absurd.
    # I must be misunderstanding the tensor structure.
    
    # Let me reconsider. The matrix multiplication tensor is NOT the all-ones tensor.
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is such that C = T(A,B), and T is a 3-tensor with entries:
    # T[i, j, k] represents the coefficient of A[i,j] * B[?, k] in C[i, k].
    # Actually, the standard way: T is a tensor of shape (n, m, p) where
    # C[i,k] = sum_{j} sum_{j'} T[i, j, j'] A[i,j] B[j',k]... no that's not right either.
    
    # The standard tensor for matrix multiplication: The bilinear map (A,B) -> C can be written as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a 3-tensor T with T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k].
    # Since A[i,j]*B[j,k] appears exactly once in C[i,k], T[i,j,k] = 1 for all i,j,k.
    # So the tensor IS the all-ones tensor!
    
    # But then rank 1 would work, which contradicts the known Strassen result for 2x2*2x2 being 7.
    # I must be confusing the tensor decomposition convention.
    
    # Let me look at this more carefully. In tensor decomposition for bilinear maps:
    # A bilinear map f: U x V -> W can be represented as a tensor T in U* x V* x W*.
    # f(u,v) = sum_w T(u,v,w) w
    # For matrix mult: f(A,B) = C, where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T_{i,k,j} such that C[i,k] = sum_j T_{i,k,j} A[i,j] B[j,k]
    # Wait, that's not a tensor decomposition in the standard sense.
    
    # The standard tensor decomposition for a bilinear map f: U x V -> W is:
    # f(u,v) = sum_r u^T(U_r) * v^T(V_r) * W_r
    # where U_r in U*, V_r in V*, W_r in W.
    
    # For matrix mult, U = R^{n x m}, V = R^{m x p}, W = R^{n x p}
    # f(A,B) = A*B
    # A basis for U*: {E_{i,j}} for i in [n], j in [m]
    # A basis for V*: {F_{j,k}} for j in [m], k in [p]
    # A basis for W: {G_{i,k}} for i in [n], k in [p]
    
    # f(A,B) = sum_{i,j,k} E_{i,j}(A) * F_{j,k}(B) * G_{i,k}
    # = sum_{i,j,k} A[i,j] * B[j,k] * G_{i,k}
    
    # So the tensor T has entries T[(i,j), (j,k), (i,k)] = 1 for all i,j,k.
    # In the decomposition: f(A,B) = sum_r U_r(A) * V_r(B) * W_r
    # where U_r, V_r are linear forms on A and B, and W_r is a matrix.
    
    # The tensor decomposition rank is the minimum r such that:
    # sum_r U_r(A) * V_r(B) * W_r = A*B
    # where U_r(A) is a scalar linear form on A, V_r(B) is a scalar linear form on B,
    # and W_r is a matrix.
    
    # Equivalently, if we flatten: let u be the vectorization of A (length n*m),
    # v be the vectorization of B (length m*p), and the output is vec(C) (length n*p).
    # The bilinear map is: vec(C) = sum_{i,j,k} u[i*m+j] * v[j*p+k] * e_{i*p+k}
    # (where e_{i*p+k} is the standard basis vector for C[i,k])
    
    # The tensor is T in (R^{n*m})* x (R^{m*p})* x R^{n*p} with entries:
    # T[i*m+j, j*p+k, i*p+k] = 1 for all i,j,k, and 0 otherwise.
    
    # Now, the decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r in R^{n*m}, V_r in R^{m*p}, W_r in R^{n*p}.
    # The condition is: for all (i,j,k):
    # sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[i*p+k] = 1
    # And for all other triples (a,b,c) not of the form (i*m+j, j*p+k, i*p+k),
    # sum_r U_r[a] * V_r[b] * W_r[c] = 0
    
    # Wait, but in the task description, the output factor is W[k*n+i, r], not W[i*p+k, r].
    # So the output index is k*n + i, meaning the output matrix is indexed as W[k*n + i] which
    # corresponds to... let me parse: W has shape (p*n, rank). The index is k*n + i.
    # If we think of the output C as having entries C[i,k] (i = row, k = column),
    # then k*n + i means: for each column k, we have n rows i. So W[k*n+i] corresponds to C[i,k].
    # This is column-major ordering of the output matrix C.
    
    # So the condition is: for all i in [n], j in [m], k in [p]:
    # sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = 1
    
    # And there are NO other constraints! The tensor only has 1's at positions (i*m+j, j*p+k, k*n+i)
    # and we don't care about other positions because the output only has n*p entries, and each
    # output entry C[i,k] = sum_j A[i,j] B[j,k] is fully determined by the n*m*p terms.
    
    # Wait, but in a proper tensor decomposition, we need the tensor to match at ALL positions,
    # not just the non-zero ones. Let me re-examine.
    
    # The bilinear map f: R^{n*m} x R^{m*p} -> R^{n*p} is defined by:
    # f(u, v) = sum_{i,j,k} u[i*m+j] * v[j*p+k] * e_{i,k} (in column-major: e_{k*n+i})
    
    # As a tensor in R^{n*m} * R^{m*p} * R^{n*p}:
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # T[a, b, c] = 0 otherwise
    
    # The tensor decomposition rank is the minimum r such that:
    # T[a,b,c] = sum_{r=1}^r U_r[a] * V_r[b] * W_r[c] for ALL a, b, c
    
    # So we DO need the decomposition to be zero at all other positions. This is much more constrained!
    
    # For the all-ones tensor (where T[a,b,c] = 1 for all a,b,c), rank 1 works.
    # But for the matrix multiplication tensor, which has a specific sparse pattern,
    # the rank is much higher.
    
    # So the problem is: find the minimum rank decomposition of the specific tensor
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i, else 0.
    
    # This is a well-studied problem. For n=m=p=2, the rank is 7 (Strassen).
    # For n=2, m=3, p=6, we need to find the rank.
    
    # Let me try a different approach: use known bounds and try to construct a solution.
    # The schoolbook rank is n*m*p = 36.
    # We want to find if we can do better.
    
    # One approach: use the fact that we can group columns of B.
    # For each pair of columns (k1, k2) of B, and for each row i of A, we need to compute:
    # C[i, k1] = sum_j A[i,j] * B[j, k1]
    # C[i, k2] = sum_j A[i,j] * B[j, k2]
    
    # We can use a Strassen-like trick for the 2-column case.
    # For fixed i and a pair (k1, k2), define:
    # X_j = A[i,j] (for j = 0, 1, 2)
    # Y_{j,k1} = B[j,k1], Y_{j,k2} = B[j,k2]
    
    # We need: sum_j X_j * Y_{j,k1} and sum_j X_j * Y_{j,k2}
    
    # This is a 3x2 matrix (X as row vector, Y as 3x2) product, giving a 1x2 result.
    # The rank of this specific bilinear map (R^3 x R^6 -> R^2) where the 6-dim is 3x2...
    
    # Actually, let me just try to do a numerical search with a better strategy.
    # I'll use the fact that the tensor has a specific structure and try to find
    # a low-rank decomposition using optimization.
    
    # For a given rank r, we have variables:
    # U: (n*m) x r = 6 x r
    # V: (m*p) x r = 18 x r  
    # W: (p*n) x r = 12 x r
    # Total: 36r variables
    
    # Constraints: for each of the 36 triples (i,j,k):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for all other (a,b,c) not in the set: the sum = 0
    
    # The number of "other" positions is (6*18*12) - 36 = 1296 - 36 = 1260
    # So we have 36 + 1260 = 1296 constraints and 36r variables.
    # For r = 36: 1296 variables, 1296 constraints (square system)
    # For r = 35: 1260 variables, 1296 constraints (overdetermined)
    
    # This is a hard nonlinear problem. Let me try a different strategy.
    
    # Strategy: Exploit the structure. The tensor T is determined by the matrix multiplication.
    # We can write T = sum over (i,j,k) of E_{i*m+j} ⊗ E_{j*p+k} ⊗ E_{k*n+i}
    # This is the schoolbook decomposition with rank 36.
    
    # To reduce the rank, we need to find linear combinations that cancel in the "wrong" positions.
    
    # Let me try a structured approach: 
    # For each i (row of A), consider the sub-problem of computing row i of C.
    # C[i,k] = sum_j A[i,j] * B[j,k] for k = 0,...,5
    # This is a linear map from A[i,:] (length 3) and B (3x6) to C[i,:] (length 6).
    
    # For fixed i, this is: C[i,:] = A[i,:] * B
    # Which is a row vector (length 3) times a 3x6 matrix, giving a 1x6 row vector.
    
    # The tensor for this sub-problem: T_i[j, k, c] where j is the column index of A[i,:],
    # k is the column index of B (and C[i,:]), and c is... wait, this doesn't quite work
    # because the output index is (i,k) not just k.
    
    # Let me just try a computational approach. I'll set up the problem as a system of
    # polynomial equations and try to solve it for decreasing ranks.
    
    # Actually, let me try a completely different approach: use the fact that we can
    # represent the matrix multiplication using block structures.
    
    # For 2x3 * 3x6, we can think of it as:
    # [A1; A2] * B where A1, A2 are 1x3 rows, B is 3x6
    # C = [A1*B; A2*B]
    
    # Each row computation is a (1x3) * (3x6) = 1x6 multiplication.
    # The tensor for (1x3)*(3x6) -> (1x6) has rank at most 3*6 = 18 (schoolbook).
    # Can we do better for (1x3)*(3x6)?
    
    # For (1xm)*(mxp), the tensor is T[j, k, c] = delta_{j,k}... no.
    # C[k] = sum_j A[j] * B[j,k]
    # T[j, k, c] where j indexes A, k indexes B columns, c indexes C.
    # T[j, k, c] = 1 if c = k (since C[k] gets contribution from A[j]*B[j,k])
    # Wait: C[c] = sum_j A[j] * B[j,c]
    # So T[j, (j,c), c]... this is getting confused with the flattening.
    
    # Let me just use the flattened indices:
    # A has indices 0..2 (length 3), B has indices 0..17 (3x6 flattened), C has indices 0..5 (length 6)
    # T[a, b, c] = 1 if a = j, b = j*6 + c, c = c for some j, c
    # I.e., T[j, j*6+c, c] = 1, else 0.
    
    # Number of non-zero entries: 3*6 = 18
    # Total entries: 3 * 18 * 6 = 324
    
    # The rank of this tensor... for (1xm)*(mxp), I believe the rank is m*p = 18 in general,
    # but maybe we can do better with structure.
    
    # Actually, I recall that for the tensor of (1xm)*(mxp), the rank is exactly m*p
    # for most values of m,p. So the row-wise approach gives rank 2*18 = 36, same as schoolbook.
    
    # Let me try yet another approach: use known results from algebraic complexity theory.
    # The tensor rank of matrix multiplication M_{n,m,p} is denoted R(M_{n,m,p}).
    # For M_{2,3,6}, I don't have a specific reference, but we can try to construct
    # a solution with rank less than 36.
    
    # Let me try a computational search with a smarter initialization.
    # I'll use the schoolbook solution as a starting point and try to reduce the rank
    # by merging terms.
    
    # Approach: Start with rank 36 (schoolbook). Try to express one term as a linear
    # combination of others, reducing the rank by 1. Repeat.
    
    # Actually, this is equivalent to finding the tensor rank, which is NP-hard in general.
    # For small cases, we can use numerical methods.
    
    # Let me try a different tactic: use the fact that half-integer entries are allowed,
    # and try a structured ansatz with small ranks.
    
    # For M_{2,2,2}, Strassen's algorithm gives rank 7.
    # For M_{2,3,6}, we can try to embed Strassen-like tricks.
    
    # Let me try: decompose the 3x6 matrix B into three 3x2 blocks: B = [B1, B2, B3]
    # Then C = [A*B1, A*B2, A*B3] where each A*Bi is 2x2.
    # Each 2x2 * 3x2... wait, A is 2x3, Bi is 3x2, so A*Bi is 2x2.
    # The tensor for (2x3)*(3x2) -> (2x2) has schoolbook rank 2*3*2 = 12.
    # If we can do (2x3)*(3x2) in rank < 12, we can reduce the total.
    
    # For (2x3)*(3x2), the tensor M_{2,3,2}... I'm not sure of the exact rank.
    # Let me try to compute it numerically.
    
    # Actually, let me just try a direct numerical optimization for the full problem.
    # I'll use a least-squares approach with random restarts.
    
    def solve_rank(r):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        np.random.seed(12345 + r)
        
        # Initialize with small random values
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Try to optimize using a simple gradient-free method
            # The objective is: sum over all (a,b,c) of (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # But T is sparse, so only 36 positions have T=1, rest are 0.
            
            # Build the list of non-zero positions
            nz = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        nz.append((i*m+j, j*p+k, k*n+i, 1.0))
            
            # Zero positions: all (a,b,c) not in nz
            # But checking all 1296 positions is expensive. Let me only check the nz positions
            # and a sample of zero positions.
            
            # For now, just check the 36 nz positions
            best_obj = float('inf')
            best_U, best_V, best_W = U, V, W
            
            for step in range(2000):
                if time.time() > deadline:
                    return None
                
                # Compute objective at nz positions
                obj = 0.0
                for (a, b, c, val) in nz:
                    s = 0.0
                    for rr in range(r):
                        s += U[a, rr] * V[b, rr] * W[c, rr]
                    obj += (s - val) ** 2
                
                if obj < best_obj:
                    best_obj = obj
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_obj < 1e-10:
                    break
                
                # Random perturbation
                scale = 0.1
                U += np.random.randn(n*m, r) * scale
                V += np.random.randn(m*p, r) * scale
                W += np.random.randn(p*n, r) * scale
            
            if best_obj < 1e-8:
                # Check all zero positions
                valid = True
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            # Check if (a,b,c) is in nz
                            is_nz = False
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        if a == i*m+j and b == j*p+k and c == k*n+i:
                                            is_nz = True
                            if not is_nz:
                                s = 0.0
                                for rr in range(r):
                                    s += best_U[a, rr] * best_V[b, rr] * best_W[c, rr]
                                if abs(s) > 1e-8:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        return None

    # Try ranks from 20 to 35
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    
    for rank in range(20, 36):
        if time.time() > deadline:
            break
        result = solve_rank(rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END

