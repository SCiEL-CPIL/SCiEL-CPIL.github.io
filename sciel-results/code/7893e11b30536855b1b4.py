"""Matrix multiplication 2,4,5 rank-minimal tensor decomposition search."""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [2, 4, 5]
# n=2 (rows of A), m=4 (cols of A / rows of B), p=5 (cols of B)

# Known lower bound for rank(MM(n,m,p)) for n,m,p >= 2 is n*m*p - max(n,m,p) + 1? No.
# For 2x4x5, schoolbook is 40.
# Using block decomposition:
# A is 2x4, B is 4x5.
# We can view A as [A1 | A2] where A1, A2 are 2x2.
# B as [B1; B2] where B1, B2 are 2x5.
# C = A1 B1 + A2 B2.
# Each 2x2 * 2x5 multiplication.
# Rank of MM(2,2,5)?
# If we use Strassen for 2x2 part, it's 7 multiplications for the 2x2x2 core, but we have 5 cols.
# Actually, rank(MM(n,m,p)) is at least n*m, n*p, m*p.
# Rank(MM(2,4,5)) >= max(8, 10, 20) = 20.
# Upper bound: 40 (schoolbook).
# Can we do better?
# Let's try to construct a decomposition of rank 30, 25, 20?
# The tensor is T_{i,j,k} = delta_{i,i} delta_{j,j} ... no, T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
# T is a 3-tensor of size 2x4x5.
# We want to write T = sum_r u^{(r)} v^{(r)} w^{(r)} where u is vector of length 8 (i,j), v is length 20 (j,k), w is length 10 (k,i).
# Wait, W index is k*n + i. So w is a vector of length p*n = 10.
# The decomposition is:
# T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# Note: W is indexed by k and i.
# This is exactly the matrix multiplication tensor.

def construct():
    n, m, p = DIMENSIONS
    # We will search for a rank r decomposition.
    # We know rank 40 works.
    # We will try to find a rank < 40.
    # Since the search space is huge, we will use a heuristic:
    # Try to fix some entries and solve for others.
    # Or, use the known result: rank(MM(2,4,5)) = ?
    # Let's check known values.
    # rank(MM(2,2,2)) = 7 (Strassen)
    # rank(MM(2,2,3)) = ?
    # Let's try a simple block decomposition with Strassen.
    # A = [A11 A12] (2x2 each)
    # B = [B11 B12] (2x5 each) -- wait, B is 4x5.
    # A (2x4) = [A_L (2x2) | A_R (2x2)]
    # B (4x5) = [B_T (2x5); B_B (2x5)]
    # C = A_L B_T + A_R B_B
    # We need to compute 2x2 * 2x5.
    # Let X be 2x2, Y be 2x5. Z = X Y.
    # Z has 2x5 = 10 entries.
    # Can we do this with < 10 multiplications?
    # Rank of MM(2,2,5).
    # It is known that rank(MM(2,2,p)) = 2p for p even? No.
    # For 2x2x2, rank is 7.
    # For 2x2x3, rank is ?
    # Let's try to find a rank 30 decomposition by using Strassen on the 2x2 blocks.
    # If we can do 2x2*2x5 with rank R, then total rank is 2R.
    # If R=15, total is 30.
    # If R=10, total is 20.
    # Let's assume we can find a decomposition of rank 30.
    
    # Let's try to construct one.
    # This is a hard problem. Let's use a solver.
    # We will set up the equations and use a numerical solver to find a rank-r decomposition.
    # We will try ranks from 40 down to 20.
    
    # Since we have limited time, we will try to find a known construction.
    # One idea: Use the identity for 2x2x2 (Strassen) and extend it.
    # Strassen for 2x2x2:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # We need to do 2x2 * 2x5.
    # Let X = [[x11, x12], [x21, x22]]
    # Let Y = [[y11, y12, y13, y14, y15], [y21, y22, y23, y24, y25]]
    # Z = X Y = [[z11..z15], [z21..z25]]
    
    # We can view Y as two 2x2 blocks and one 2x1 block.
    # Y = [Y1 (2x2) | Y2 (2x2) | Y3 (2x1)]
    # Z = X Y1 + X Y2 + X Y3
    # This doesn't help directly unless we can reuse Strassen multiplications.
    
    # Alternative:
    # Maybe the rank is exactly 40? No, 7 < 8 suggests reduction is possible.
    # For 2x4x5, the rank is likely less than 40.
    # Let's try to find a rank 30 decomposition.
    
    # We will use a random search with constraints.
    # But first, let's try a specific known construction.
    # I will construct a rank 30 decomposition using the "split" method.
    
    # Actually, let's just return the schoolbook if we can't find better quickly.
    # But the task asks for rank-minimal.
    # Let's try to solve for rank 30.
    
    # We will use scipy if available, else a simple solver.
    # Since we have 150s, we can try a few iterations.
    
    # Let's try to construct a rank 30 decomposition by solving a system of linear equations
    # where we fix some variables and solve for the rest.
    
    # This is too complex to implement correctly in a short time without bugs.
    # Let's try a simpler heuristic:
    # If we can reduce the rank of 2x2x5 to 15, then 2 * 15 = 30.
    
    # Let's try to find a 2x2x5 decomposition of rank 15.
    # T[i,j,k] = sum_{r=1}^{15} U[i,j,r] V[j,k,r] W[k,i,r]
    # U is 2x2x15, V is 2x5x15, W is 5x2x15.
    # This is a tensor of size 2x2x5.
    
    # I will use a random initial guess and gradient descent on the "rank constraint" (sum of squares of singular values)?
    # No, we need exact integer/half-integer.
    
    # Given the time limit and complexity, I will return the schoolbook rank 40.
    # However, I suspect the rank is 30 or lower.
    # Let's try to implement a solver for rank 30.
    
    # We will use the fact that the tensor is bilinear.
    # We can set up the equations for C_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    # Let's try to use the Strassen-like construction for 2x2x5.
    # Reference: "On the rank of the matrix multiplication tensor"
    # For 2x2xp, the rank is known to be 2p for p <= 4? No, 2x2x2 is 7.
    # 2x2x3 is 9? 2x2x4 is 12? 2x2x5 is 15?
    # If rank(2x2x5) = 15, then rank(2x4x5) <= 30.
    
    # Let's assume rank 30 and try to construct it.
    # I will write a function to verify a candidate decomposition.
    
    def verify(U, V, W):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We check if this equals delta_{i,i} delta_{j,j} ... no, it equals the matrix product tensor.
        # The tensor T has T[i,j,k] = 1 if the indices correspond to a valid product term?
        # No, the tensor T is defined such that <T, A \otimes B> = C.
        # T[i,j,k] is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
        # In the standard basis, T[i,j,k] = 1 if the indices match the pattern?
        # Actually, T is the 3-tensor with T[i,j,k] = 1 if i,j,k are "compatible"?
        # No, for matrix multiplication, the tensor is T_{i,j,k} = \delta_{i, i} \delta_{j, j} ... 
        # Let's be precise.
        # C_{i,k} = \sum_j A_{i,j} B_{j,k}.
        # So the tensor T has non-zero entries only when the indices form a "chain" i-j-k?
        # No, T is a 3-tensor of size n x m x p.
        # T[i,j,k] is the coefficient of A_{i,j} B_{j,k} in the expansion of C.
        # But C is a matrix of size n x p.
        # So T is not a 3-tensor in the sense of a single entry.
        # The decomposition is:
        # C_{i,k} = \sum_r (\sum_j U[i,j,r] A_{i,j}) (\sum_l V[j,l,r] B_{j,l}) W[k,i,r] ?
        # No, the standard decomposition is:
        # C_{i,k} = \sum_r U[i,j,r] V[j,k,r] W[k,i,r] is NOT correct because j is summed out?
        # Let's look at the interface:
        # U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # The product is:
        # \sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # This sum is over r. The indices i,j,k are free.
        # So for fixed i,j,k, we get a value.
        # We want this value to be 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}, and 0 otherwise.
        # But A_{i,j} B_{j,k} contributes to C_{i,k} for all i,j,k?
        # No, in the matrix product C = AB, C_{i,k} = \sum_j A_{i,j} B_{j,k}.
        # So the term A_{i,j} B_{j,k} appears in C_{i,k}.
        # So the tensor T[i,j,k] should be 1 for all i,j,k?
        # No, that would mean C_{i,k} = \sum_{j,r} U V W.
        # If T[i,j,k] = 1 for all i,j,k, then the decomposition is:
        # 1 = \sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # This is only true if the product of the matrices forms an identity?
        # No, the tensor T for matrix multiplication is defined as:
        # T[i,j,k] = 1 if the indices are such that A_{i,j} and B_{j,k} multiply to form C_{i,k}.
        # Since A is n x m, B is m x p, C is n x p.
        # The term A_{i,j} B_{j,k} always contributes to C_{i,k}.
        # So T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # Is that right?
        # Let's check.
        # If T[i,j,k] = 1, then \sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
        # This must hold for all i,j,k.
        # So we need to find U,V,W such that the trilinear form is 1 everywhere.
        
        # Let's test with schoolbook.
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = i*m*p + j*p + k.
        # Sum over r.
        # For a fixed i,j,k, only one r (the one corresponding to i,j,k) has non-zero entries.
        # For that r, U=1, V=1, W=1. Product is 1.
        # For other r, at least one is 0.
        # So the sum is 1.
        # So yes, T[i,j,k] = 1 for all i,j,k.
        
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    # We will try to find a rank 30 decomposition.
    # We will use a random search.
    # Initialize U, V, W with random integers.
    # Then, try to adjust them to satisfy the equations.
    # This is a system of n*m*p = 40 equations with 3*rank*(n*m + m*p + p*n) variables.
    # For rank 30, variables = 30 * (8 + 20 + 10) = 30 * 38 = 1140.
    # Equations = 40.
    # Underdetermined.
    
    # We can set up the system as:
    # \sum_r U_i r V_j r W_k r = 1
    # This is not linear in U,V,W.
    # We can fix W and solve for U,V?
    # Or fix U and V and solve for W?
    # If we fix U and V, then W[k*n+i, r] is linear in the equations?
    # \sum_r W[k*n+i, r] (U[i*m+j, r] V[j*p+k, r]) = 1
    # Let A_{i,j,k,r} = U[i*m+j, r] V[j*p+k, r].
    # Then \sum_r W[k*n+i, r] A_{i,j,k,r} = 1.
    # This is a linear system for W.
    # The number of W variables is p*n*rank = 10*30 = 300.
    # The number of equations is n*m*p = 40.
    # We can solve for W if the matrix is full rank.
    # But we also need W to be integers/half-integers.
    
    # Strategy:
    # 1. Randomly initialize U and V with integers.
    # 2. Form the linear system for W.
    # 3. Solve for W using least squares (or exact if possible).
    # 4. Check if W is close to integers/half-integers.
    # 5. If yes, round and verify.
    
    # Let's try this.
    
    import numpy as np
    
    best_rank = 40
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook
    rank = 40
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    if verify(U.tolist(), V.tolist(), W.tolist()):
        best_U = U
        best_V = V
        best_W = W
        best_rank = rank
    
    # Try to find lower rank
    for r in range(30, 40):
        # Try a few random initializations
        for _ in range(10):
            U = np.random.randint(-2, 3, (n*m, r))
            V = np.random.randint(-2, 3, (m*p, r))
            # Form system for W
            # Equations: for each i,j,k: sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r] = 1
            # Let M be a matrix of size (n*m*p, p*n*r)
            # M[(i*m+j)*p+k, (k*n+i)*r + r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
            # But W is indexed by (k*n+i, r).
            # Let's vectorize W into a vector of length p*n*r.
            # W_vec[idx] = W[idx // r, idx % r]
            # The equation is: sum_r W_vec[(k*n+i)*r + r] * U[i*m+j, r] * V[j*p+k, r] = 1
            # Let's build the matrix A where A[eq_idx, w_idx] = coefficient.
            A = np.zeros((n*m*p, p*n*r))
            b = np.ones(n*m*p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i*m + j)*p + k
                        for r_idx in range(r):
                            w_idx = (k*n + i)*r + r_idx
                            A[eq_idx, w_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
            
            # Solve A W_vec = b
            try:
                W_vec, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                # Check if solution is close to integers/half-integers
                # W_vec should be close to multiples of 0.5
                W_test = np.round(W_vec * 2) / 2
                if np.allclose(A @ W_test, b, atol=1e-9):
                    # Check if W_test is valid
                    W_mat = W_test.reshape((p*n, r))
                    if verify(U.tolist(), V.tolist(), W_mat.tolist()):
                        best_U = U
                        best_V = V
                        best_W = W_mat
                        best_rank = r
                        break # Found a decomposition for this rank
            except:
                pass
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# Keep the original construct name and signature
def construct():
    return _construct_impl()

def _construct_impl():
    return construct_impl()

# Actually, just use the name construct
# Let's rename the inner one to avoid confusion

def construct():
    n, m, p = DIMENSIONS
    best_rank = 40
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook
    rank = 40
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify function
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    if verify(U.tolist(), V.tolist(), W.tolist()):
        best_U = U
        best_V = V
        best_W = W
        best_rank = rank
    
    # Try to find lower rank
    # We will try ranks 30, 35
    for r in [30, 35]:
        for _ in range(20):
            U = np.random.randint(-2, 3, (n*m, r))
            V = np.random.randint(-2, 3, (m*p, r))
            A = np.zeros((n*m*p, p*n*r))
            b = np.ones(n*m*p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i*m + j)*p + k
                        for r_idx in range(r):
                            w_idx = (k*n + i)*r + r_idx
                            A[eq_idx, w_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
            
            try:
                W_vec, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                W_test = np.round(W_vec * 2) / 2
                if np.allclose(A @ W_test, b, atol=1e-9):
                    W_mat = W_test.reshape((p*n, r))
                    if verify(U.tolist(), V.tolist(), W_mat.tolist()):
                        best_U = U
                        best_V = V
                        best_W = W_mat
                        best_rank = r
                        break
            except:
                pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
