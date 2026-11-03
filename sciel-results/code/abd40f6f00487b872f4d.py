"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Strassen terms M1..M7
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # A00 is U[0], A11 is U[3]
    # B00 is V[0], B11 is V[3]
    # C00 is W[0], C11 is W[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    # A10 is U[2], A11 is U[3]
    # B00 is V[0]
    # C10 is W[1], C11 is W[3]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    # A00 is U[0]
    # B01 is V[1], B11 is V[3]
    # C01 is W[2], C11 is W[3]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    # A11 is U[3]
    # B10 is V[2], B00 is V[0]
    # C00 is W[0], C10 is W[1]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    # A00 is U[0], A01 is U[1]
    # B11 is V[3]
    # C00 is W[0], C01 is W[2]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # A10 is U[2], A00 is U[0]
    # B00 is V[0], B01 is V[1]
    # C11 is W[3]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    # A01 is U[1], A11 is U[3]
    # B10 is V[2], B11 is V[3]
    # C00 is W[0]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _get_school_221():
    """
    Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4).
    U shape: (4, 4), V shape: (2, 4), W shape: (2, 4)
    Indices: U[i*2+j], V[j*1+k], W[k*2+i]
    """
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W satisfy the tensor identity exactly.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Identity: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = delta_{j,k} is WRONG.
    Correct Identity for Matrix Mult (A n x m, B m x p -> C n x p):
    C[i, k] = sum_j A[i, j] B[j, k]
    Tensor T[i, j, k] = delta_{j, j'} is not right.
    
    The tensor for matrix multiplication M_{n,m,p} has entries T[i, j, k] = 1 if j is the index connecting A and B.
    Wait, the standard tensor definition for matrix multiplication:
    T is a tensor of order 3 with dimensions (n, m, p).
    T[i, j, k] = 1 if j is the "middle" index? No.
    
    Let's look at the constraint in the prompt:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    This implies:
    For fixed i, k, the sum over c of W_{ik} U_{ij} V_{j'k} equals delta_{j,j'}.
    This is exactly the condition that the matrices U, V, W define a decomposition of the identity tensor in the middle dimension?
    No, this is the condition for the *multiplication* tensor.
    Let A be n x m, B be m x p. C = A B.
    C_{ik} = sum_j A_{ij} B_{jk}.
    If we write A_{ij} U_{ij,c} ... no.
    
    The decomposition is:
    C_{ik} = sum_c W_{ik,c} (sum_j U_{ij,c} A_{ij}) (sum_j' V_{j'k,c} B_{j'k}) ? No.
    
    The standard bilinear complexity formulation:
    C_{ik} = sum_c M_c
    M_c = (sum_j U_{ij,c} A_{ij}) * (sum_j V_{j'k,c} B_{j'k})
    
    Expanding M_c:
    M_c = sum_j sum_j' U_{ij,c} V_{j'k,c} A_{ij} B_{j'k}
    
    So C_{ik} = sum_c sum_j sum_j' U_{ij,c} V_{j'k,c} A_{ij} B_{j'k}
    
    We require C_{ik} = sum_j A_{ij} B_{jk}.
    This means for all i, j, j', k:
    sum_c U_{ij,c} V_{j'k,c} W_{ik,c} = delta_{j, j'}
    
    Yes, this matches the prompt's constraint:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    """
    r = U.shape[1]
    # Optimization: use numpy broadcasting or loops?
    # n=2, m=4, p=7. Small enough for loops.
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j, c] * V[jp * p + k, c]
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _build_rank50():
    """Build the rank-50 block decomposition."""
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [
        (0, 2, 'strassen'),
        (2, 4, 'strassen'),
        (4, 6, 'strassen'),
        (6, 7, 'school')
    ]
    
    U_s, V_s, W_s = _get_strassen_222()
    U_sc, V_sc, W_sc = _get_school_221()
    
    for j_start, j_end in m_blocks:
        for k_start, k_end, p_type in p_blocks:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            
            for i in range(n):
                for j_sub in range(j_end - j_start):
                    j = j_start + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            p_sub = k_end - k_start
            for j_sub in range(j_end - j_start):
                j = j_start + j_sub
                for k_sub in range(p_sub):
                    k = k_start + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_sub + k_sub, :]
            
            for k_sub in range(p_sub):
                k = k_start + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy."""
    r = U.shape[1]
    # For each (i, k), check that sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    # This is equivalent to: for each c, the outer product U_c ⊗ V_c ⊗ W_c should sum to the identity tensor
    # Let's build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and check T[i,j,k] = 1 if j==k else 0... wait, no.
    # Actually T[i,j,j',k] where j and j' are both middle indices.
    # The condition is: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
    
    # Let's construct it properly.
    # For fixed i, k: the matrix M_{j,j'} = sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c]
    # This should equal I (identity matrix of size m x m).
    
    for i in range(n):
        for k in range(p):
            # Build m x m matrix
            M = np.zeros((m, m))
            for c in range(r):
                w_val = W[k * n + i, c]
                if w_val != 0:
                    u_col = U[i * m:(i+1) * m, c]  # length m
                    v_col = V[:, k].reshape(-1)  # this is wrong, V[j*p+k, c] for all j
                    # V[j*p+k, c] for j in range(m)
                    v_vals = np.array([V[j * p + k, c] for j in range(m)])
                    M += w_val * np.outer(u_col, v_vals)
            expected = np.eye(m)
            if not np.allclose(M, expected, atol=1e-9):
                return False
    return True

def _try_reduce_rank_by_one(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by 1 by finding a linear dependency.
    The idea: if there exist coefficients alpha_c (not all zero) such that
    sum_c alpha_c * (U_c ⊗ V_c ⊗ W_c) = 0 (as a tensor),
    then we can eliminate one term.
    
    In practice, we look for a column c0 such that the rank-1 tensor for c0
    is in the span of the others.
    
    A simpler approach: randomly perturb and see if we can find a lower rank solution.
    But that's hard. Instead, let's try a different strategy:
    
    We know the block structure. The rank-50 decomposition has 2 m-blocks, each with
    3 Strassen (rank 7) and 1 schoolbook (rank 4) p-blocks.
    
    One idea: maybe we can merge some of the schoolbook terms across m-blocks.
    Or maybe there's a non-block decomposition with rank 49.
    
    For now, let's try a randomized search: start from rank-50, randomly modify
    some entries, and check if we can maintain validity with rank 49.
    """
    r = U.shape[1]  # 50
    
    # Strategy: try to find if any column is linearly dependent on others
    # in a way that allows rank reduction.
    
    # Build the tensor as a matrix: T is (n*m*p, n*m*p) or similar.
    # Actually, the multiplication tensor M_{n,m,p} can be viewed as a matrix
    # of size (n*p) x (n*m) * (m*p) or flattened.
    
    # Let's use the "slice" approach: for each c, we have a rank-1 tensor.
    # The sum of all rank-1 tensors should equal the multiplication tensor.
    
    # One approach: check if the 50 rank-1 tensors are linearly independent.
    # If not, we can reduce rank.
    
    # Represent each rank-1 term as a vector in R^{n*m*p} (or R^{(n*m)*(m*p)*(p*n)}).
    # Actually, each term c contributes to T[i,j,k] via U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # But the tensor T is 3D with indices (i, j, k) where j is the "middle" index.
    # Wait, the tensor for matrix mult has indices (i, j, k) but the constraint is
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}.
    # This is a 4-index tensor, not 3-index.
    
    # Let me reconsider. The bilinear complexity formulation:
    # C[i,k] = sum_c M_c, where M_c = (sum_j U[i*m+j, c] * A[i,j]) * (sum_j' V[j'*p+k, c] * B[j',k])
    # Expanding: M_c = sum_j sum_j' U[i*m+j, c] * V[j'*p+k, c] * A[i,j] * B[j',k]
    # So C[i,k] = sum_c sum_j sum_j' U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] * A[i,j] * B[j',k]
    # We need this to equal sum_j A[i,j] * B[j,k].
    # So for all i, j, j', k: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
    
    # This is a system of equations. The number of equations is n * m * m * p = 2*4*4*7 = 224.
    # The number of variables is r * (n*m + m*p + p*n) = 50 * (8 + 28 + 14) = 50 * 50 = 2500.
    # So there are many degrees of freedom.
    
    # To reduce rank, we need to find a solution with r=49.
    # This is a hard problem in general, but for small cases, we can try.
    
    # Approach: use the rank-50 solution as a starting point, and try to find
    # a linear combination that allows us to eliminate one column.
    
    # Let's try a simple heuristic: for each pair of columns (c1, c2), check if
    # they are "similar" in a way that allows merging.
    
    # Actually, let's try a different approach: use scipy to solve for rank 49.
    # Set up the linear system and try to find a solution with rank 49.
    
    # The equations are: for each (i, j, j', k):
    # sum_{c=0}^{r-1} U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] = delta_{j,j'}
    
    # This is nonlinear (product of three variables). But we can linearize by
    # fixing two of them and solving for the third.
    
    # Alternative: use the fact that the rank-50 solution is known, and try to
    # perturb it to find a rank-49 solution.
    
    # For now, let's just return the rank-50 solution.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Build rank-50 solution
    U, V, W = _build_rank50()
    
    # Verify
    if _verify_fast(U, V, W, n, m, p):
        # Try to reduce rank
        if time.time() < deadline - 10:
            reduced = _try_reduce_rank_by_one(U, V, W, n, m, p, deadline)
            if reduced is not None:
                return reduced
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        r = n * m * p
        U_fb = [[0] * r for _ in range(n * m)]
        V_fb = [[0] * r for _ in range(m * p)]
        W_fb = [[0] * r for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
