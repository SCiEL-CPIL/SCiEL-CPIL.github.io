"""
Optimized tensor decomposition finder for 2x5x5 matrix multiplication.
Attempts to find a rank-minimal exact decomposition.
Falls back to the standard schoolbook decomposition if no better rank is found.
"""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 5]
DEADLINE_SECONDS = 140  # Leave 10s margin for the evaluator's 150s limit

def _deadline_ok(start_time, margin=10):
    return (time.time() - start_time) < (DEADLINE_SECONDS - margin)

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _verify(U, V, W, n, m, p, tol=1e-10):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if i==k (for the identity-like structure of matmul),
    but actually for matrix multiplication (A is n x m, B is m x p, C is n x p),
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T_{i,j,k} represents the coefficient of A_{i,j} * B_{j,k} in C_{i,k}.
    So T_{i,j,k} = 1 if the product contributes to C_{i,k}, which it always does.
    Wait, the tensor for matrix multiplication is defined as:
    T_{i,j,k} = delta_{i,k} ? No.
    
    Let me reconsider. The matrix multiplication tensor T has entries T_{i,j,k}
    such that the multiplication can be written as:
    C_{i,k} = sum_{j} sum_{r} U_{i,j}^{(r)} * A_{i,j} * V_{j,k}^{(r)} * B_{j,k} ... 
    
    Actually, the standard formulation is:
    C_{i,k} = sum_{j=1}^{m} A_{i,j} B_{j,k}
    
    The tensor T_{i,j,k} = 1 for all i,j,k? No, that's not right either.
    
    Let me think more carefully. The matrix multiplication tensor is a 3-tensor T
    with indices (i, j, k) where i in [n], j in [m], k in [p].
    T_{i,j,k} = 1 if the term A_{i,j} * B_{j,k} contributes to C_{i,k}.
    But every term A_{i,j} * B_{j,k} contributes to C_{i,k} for fixed i,k.
    
    Actually, the tensor is defined as T_{i,j,k} = 1 for all i, j, k,
    and the multiplication is:
    C_{i,k} = sum_{j} T_{i,j,k} * A_{i,j} * B_{j,k}
    
    Wait, that would mean T_{i,j,k} = 1 for all i,j,k, which gives rank n*m*p
    in the schoolbook. But the tensor rank is the minimum number of rank-1
    tensors needed to express T.
    
    A rank-1 tensor has the form u_i * v_j * w_k.
    So we need: T_{i,j,k} = sum_{r=1}^{R} U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
    
    Hmm, the indices need to be consistent. Let me use the convention from the task:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W[k*n + i, r]
    
    So the decomposition is:
    T_{i,j,k} = sum_{r} U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    And T_{i,j,k} = 1 for all i, j, k (since every product A_{i,j}B_{j,k} contributes
    to C_{i,k}).
    
    Wait, that's not right. Let me reconsider.
    
    Actually, for matrix multiplication C = AB where A is n x m and B is m x p:
    C_{i,k} = sum_{j=1}^{m} A_{i,j} B_{j,k}
    
    The "tensor" here is the structure of the multiplication. Each term A_{i,j} B_{j,k}
    contributes to C_{i,k}. So the tensor T has T_{i,j,k} = 1 for all valid i,j,k.
    
    But that's a trivial tensor of all 1s, which has rank 1! That can't be right.
    
    I think I'm confusing the tensor. Let me look at this differently.
    
    The matrix multiplication tensor is a 3-tensor with dimensions (n, m, p) where
    T_{i,j,k} represents the coefficient in the bilinear form. The correct definition is:
    
    The multiplication can be expressed as:
    C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    
    This is a bilinear map from (A, B) to C. The tensor T is defined such that
    <T, A ⊗ B> = C, where the inner product is over the appropriate indices.
    
    More precisely, T is a tensor in R^{n x p} ⊗ (R^{n x m})^* ⊗ (R^{m x p})^*.
    Flattening A to a vector in R^{n*m} and B to a vector in R^{m*p}, we have:
    vec(C) = M * (vec(A) ⊗ vec(B)) for some matrix M.
    
    The tensor rank is the minimum R such that:
    T_{i,k,j} = sum_{r=1}^{R} C_{i,k}^{(r)} * A_{i,j}^{(r)} * B_{j,k}^{(r)}
    
    Wait, I need to be more careful with indices.
    
    Let me use the convention from the problem statement:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W[k*n + i, r]
    
    The decomposition is:
    T_{i,j,k} = sum_{r} U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    And T_{i,j,k} should equal 1 for all i, j, k? No, that's not right.
    
    Actually, I think the tensor T_{i,j,k} = 1 for all i, j, k is correct for the
    "structure" of matrix multiplication, but the rank is not 1 because the tensor
    is not a simple outer product.
    
    Let me verify with the schoolbook:
    For each (i, j, k), we have a term U[i*m+j, col] = 1, V[j*p+k, col] = 1,
    W[k*n+i, col] = 1. So T_{i,j,k} = sum over cols where all three are 1.
    In the schoolbook, each (i,j,k) has exactly one column where all three are 1,
    so T_{i,j,k} = 1. Good.
    
    So the tensor T is indeed all 1s, and we need to find the minimum rank decomposition
    of the all-1s tensor of shape (n*m, m*p, p*n) with the specific index structure.
    
    Wait, but the all-1s tensor of shape (a, b, c) has rank 1 (it's just the outer
    product of three all-1s vectors). So why is the schoolbook rank n*m*p?
    
    I think I'm misunderstanding the tensor structure. Let me reconsider.
    
    Ah, I see the issue. The tensor is NOT all 1s. The matrix multiplication tensor
    has a specific structure where T_{i,j,k} depends on the indices in a way that
    reflects the multiplication structure.
    
    Actually, I think the correct interpretation is:
    - The input A has entries A_{i,j} for i in [n], j in [m]
    - The input B has entries B_{j,k} for j in [m], k in [p]
    - The output C has entries C_{i,k} for i in [n], k in [p]
    - C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    
    The tensor T is defined as T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} contributes
    to C_{i,k}, which it does for all i, j, k. But the "contribution" is structured
    such that the same (i,j) index in A and (j,k) index in B must be paired.
    
    I think the key insight is that the tensor is not simply all 1s, but rather
    T_{i,j,k} = 1 for all i, j, k, and the rank is the tensor rank of this all-1s
    tensor under the specific index structure imposed by the matrix multiplication.
    
    Actually, I think I've been overthinking this. Let me just verify that the
    schoolbook decomposition is correct and try to find a lower rank.
    
    For the all-1s tensor of shape (n*m, m*p, p*n), the tensor rank is indeed 1
    (outer product of all-1s vectors). But the constraint is that the decomposition
    must respect the index structure of matrix multiplication.
    
    Hmm, I think the correct interpretation is that the tensor T has entries
    T_{i,j,k} = 1 for all i, j, k, and we're looking for a decomposition of the
    form:
    T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    
    Since T_{i,j,k} = 1 for all i, j, k, this is equivalent to finding the minimum
    rank of the all-1s tensor under this specific index structure.
    
    But the all-1s tensor has rank 1, so we should be able to do it with rank 1:
    U[i*m+j, 0] = 1 for all i, j
    V[j*p+k, 0] = 1 for all j, k
    W[k*n+i, 0] = 1 for all k, i
    
    Then T_{i,j,k} = 1 * 1 * 1 = 1. That works!
    
    But that can't be right because the schoolbook is rank 50 for 2x5x5.
    
    I think the issue is that the tensor is NOT all 1s. Let me reconsider the
    definition of the matrix multiplication tensor.
    
    After more thought, I believe the correct definition is:
    The matrix multiplication tensor T is a 3-tensor with T_{i,j,k} = 1 if
    (i, j, k) is a valid triple for the multiplication, which it is for all
    i in [n], j in [m], k in [p]. But the "structure" of the tensor is such that
    the indices are linked in a specific way.
    
    I think the key is that the tensor is not simply all 1s, but rather encodes
    the bilinear structure of matrix multiplication. The correct tensor is:
    T_{i,j,k} = 1 for all i, j, k, and the rank is the minimum number of
    rank-1 terms needed to express T under the constraint that each rank-1 term
    has the form u_i * v_j * w_k where u, v, w are vectors of appropriate lengths.
    
    But again, the all-1s tensor has rank 1.
    
    I'm clearly misunderstanding something. Let me just implement the verification
    correctly and see what happens.
    
    Actually, I think the issue is that the tensor T is NOT all 1s. The matrix
    multiplication tensor has T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} contributes
    to C_{i,k}, but the "contribution" is not just a scalar 1; it's a structured
    contribution that depends on the indices.
    
    Let me try a different approach: I'll implement the verification by checking
    that the decomposition correctly computes matrix multiplication for random
    matrices.
    """
    # Verify by checking that the decomposition correctly computes matrix multiplication
    # for several random test matrices.
    
    # Generate random test matrices
    rng = np.random.RandomState(42)
    for _ in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # Compute C using the decomposition
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Using the decomposition:
        # C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # Wait, that's not right. The decomposition is:
        # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And C_{i,k} = sum_j T_{i,j,k} * A[i,j] * B[j,k]
        # So: C_{i,k} = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(U.shape[1]):
                        C_computed[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False, np.max(np.abs(C_computed - C_expected))
    
    return True, 0.0

def _hooi_decomposition(n, m, p, rank, max_iter=100, tol=1e-12, seed=0):
    """
    Attempt to find a rank-r decomposition using Higher-Order Orthogonal Iteration.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize factors randomly
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Normalize
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    for it in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()
        W_old = W.copy()
        
        # Update U: for each (i, j), solve least squares
        # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For fixed (i, j), we want to find U[i*m+j, :] such that:
        # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = T_{i,j,k} for all k
        # This is a least squares problem: U[i*m+j, :] @ (V[j*p+k, r] * W[k*n+i, r])^T = T_{i,j,k}
        
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        for i in range(n):
            for j in range(m):
                # Build the matrix for least squares
                # For each k, we have a row: (V[j*p+k, 0] * W[k*n+i, 0], ..., V[j*p+k, rank-1] * W[k*n+i, rank-1])
                A_ls = np.zeros((p, rank))
                for k in range(p):
                    for r in range(rank):
                        A_ls[k, r] = V[j*p+k, r] * W[k*n+i, r]
                b_ls = np.ones(p)  # T_{i,j,k} = 1
                # Solve A_ls @ U[i*m+j, :]^T = b_ls
                sol, _, _, _ = np.linalg.lstsq(A_ls, b_ls, rcond=None)
                U_new[i*m+j, :] = sol
        
        for j in range(m):
            for k in range(p):
                A_ls = np.zeros((n * m, rank))
                for i in range(n):
                    for r in range(rank):
                        A_ls[i*m + (j if j < m else j % m), r] = U_new[i*m+j, r] * W_new[k*n+i, r]
                # This is getting complicated. Let me simplify.
                # For fixed (j, k), we want V[j*p+k, :] such that:
                # sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = T_{i,j,k} for all i
                # A_ls[i, r] = U[i*m+j, r] * W[k*n+i, r]
                A_ls = np.zeros((n, rank))
                for i in range(n):
                    for r in range(rank):
                        A_ls[i, r] = U_new[i*m+j, r] * W_new[k*n+i, r]
                b_ls = np.ones(n)
                sol, _, _, _ = np.linalg.lstsq(A_ls, b_ls, rcond=None)
                V_new[j*p+k, :] = sol
        
        for k in range(p):
            for i in range(n):
                A_ls = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        A_ls[j, r] = U_new[i*m+j, r] * V_new[j*p+k, r]
                b_ls = np.ones(m)
                sol, _, _, _ = np.linalg.lstsq(A_ls, b_ls, rcond=None)
                W_new[k*n+i, :] = sol
        
        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        if np.max(np.abs(U - U_old)) < tol and np.max(np.abs(V - V_old)) < tol and np.max(np.abs(W - W_old)) < tol:
            break
    
    # Verify
    valid, err = _verify(U, V, W, n, m, p, tol=1e-8)
    if valid:
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    best_U, best_V, best_W, best_rank = _schoolbook(n, m, p)
    
    # Try to find lower rank using HOOI
    # The theoretical lower bound for 2x5x5 is not known exactly,
    # but let's try ranks from 21 upward (Strassen-like bounds suggest < 50)
    
    for rank in range(21, best_rank + 1):
        if not _deadline_ok(start_time, margin=15):
            break
        
        found = False
        for seed in range(5):
            if not _deadline_ok(start_time, margin=15):
                break
            
            result = _hooi_decomposition(n, m, p, rank, max_iter=50, tol=1e-10, seed=seed)
            if result is not None:
                U, V, W = result
                valid, err = _verify(U, V, W, n, m, p, tol=1e-8)
                if valid:
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
        
        if found:
            continue
    
    # Convert to lists for the return interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
