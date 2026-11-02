import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS

    # Schoolbook solution (rank 56) as fallback
    def schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > tol:
                            return False
        return True

    def try_mps_decomposition():
        """
        Matrix Product State (MPS) / Tensor Network approach.
        
        The tensor T is 2x4x7. We want to decompose it into a sum of rank-1 tensors.
        We can view this as finding a low-rank approximation in a specific tensor network structure.
        
        Specifically, we can arrange the indices as:
        A (2x4) and B (4x7) -> C (2x7).
        
        Let's try to represent the multiplication tensor T[i, j, k] (where i is row of A/C, j is col of A/row of B, k is col of B/C)
        as a contraction of MPS-like cores.
        
        However, for exact decomposition, we are looking for the tensor rank.
        The "MPS" idea here is to use the structure of the indices to find a decomposition.
        
        Let's try a different perspective: 
        The tensor is T = sum_{j} A[:,j] * B[j,:] * E_{j,j}? No.
        T[i,j,k] = 1 if we are at the "right" position.
        Actually, the multiplication tensor M satisfies M[i,j,k] = 1 for all i,j,k? No.
        The tensor for matrix multiplication C = A B is defined such that C_{ik} = sum_j A_{ij} B_{jk}.
        So the tensor T has T_{i,j,k} = 1.
        
        We want T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}.
        
        Let's try to use the small dimension n=2 to our advantage.
        We can write T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}.
        
        For fixed i, T_{i,j,k} is a 4x7 matrix (j,k).
        T_{0,j,k} = 1.
        T_{1,j,k} = 1.
        
        Wait, the tensor entries are all 1?
        Let's check the verify function:
        s = sum_c U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
        We want s = 1 if j==jp else 0.
        
        So the tensor we are decomposing is NOT the all-ones tensor.
        It is the tensor T where T[i, j, k, jp] = delta_{j,jp}.
        But the decomposition form is U[i,j] * V[jp,k] * W[k,i] ?
        Let's re-read the verify function carefully.
        
        s = sum_c U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
        
        The indices are:
        U: (i*m+j) -> (i, j)
        V: (jp*p+k) -> (jp, k)
        W: (k*n+i) -> (k, i)
        
        So we are decomposing a 4-index tensor T[i, j, jp, k] such that:
        T[i, j, jp, k] = sum_c U[i, j, c] V[jp, k, c] W[k, i, c]
        
        And we require T[i, j, jp, k] = 1 if j==jp else 0.
        Note that T does not depend on i or k in the value (it's just delta_{j,jp}).
        But the decomposition must hold for all i, k.
        
        This means:
        For all i, k, j, jp:
        sum_c U[i, j, c] V[jp, k, c] W[k, i, c] = delta_{j,jp}
        
        This is a very strong condition. It implies that the "interaction" between the (i,k) indices and the (j,jp) indices is factorized in a specific way.
        
        Let's fix j, jp.
        If j != jp, then for all i, k: sum_c U[i, j, c] V[jp, k, c] W[k, i, c] = 0.
        If j == jp, then for all i, k: sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1.
        
        This looks like we can separate the i,k part.
        Let X_{i,k,c} = W[k, i, c].
        Then sum_c U[i, j, c] V[j, k, c] X_{i,k,c} = 1.
        
        This is essentially asking for a decomposition of the identity-like structure.
        
        Let's try to construct such a decomposition using the Strassen block structure, which we know works (rank 50).
        The question is: can we do better?
        
        The MPS strategy suggests looking for a low bond dimension representation.
        In the context of tensor rank, this doesn't directly apply unless we view the tensor as a matrix product state.
        
        However, we can try to optimize the rank 50 solution by checking if any columns are linearly dependent or if we can merge columns.
        Since the entries are integers (from Strassen), we can check for linear dependence over Q.
        
        Let's try to reduce the rank by checking if the 50 vectors (columns of U, V, W) are linearly independent.
        Actually, we can check if the tensor decomposition is minimal.
        
        A simpler check: can we remove a column and adjust others?
        This is equivalent to checking if the rank is 49.
        
        Let's try a random search for rank 49, starting from the rank 50 solution and perturbing it.
        Or, better, use the known lower bounds.
        The lower bound for 2x4x7 is not well known, but it's at least 2*4 = 8? No, that's for 2xn x nx2.
        For 2x4x7, the trivial lower bound is 2*4*7 / (2*4*7) = 1? No.
        The lower bound is max(2, 4, 7) = 7? No.
        
        Actually, the rank of the 2x4x7 multiplication tensor is at least the rank of the 2x4 matrix (which is 2) and the 4x7 matrix (which is 4)? No.
        
        Let's stick with the rank 50 solution and try to verify if it can be improved by a small search.
        Given the time constraints, we will just return the rank 50 solution, as it is known to be valid.
        
        If we have time, we can try to see if rank 49 is possible by solving a system of equations.
        """
        if time.time() > 140:
            return None
        
        # Try to see if we can reduce rank by 1 by checking linear dependence
        # This is computationally expensive, so we'll skip it for now and return the known solution.
        return None

    # Build the rank 50 solution
    rank = 50
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)

        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)

        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M2 = (c+d)e
        U[c_idx, col] += 1; U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1; W[c11_idx, col] -= 1
        col += 1

        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1; V[h_idx, col] -= 1
        W[c01_idx, col] += 1; W[c11_idx, col] += 1
        col += 1

        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1; V[e_idx, col] -= 1
        W[c00_idx, col] += 1; W[c10_idx, col] += 1
        col += 1

        # M5 = (a+b)h
        U[a_idx, col] += 1; U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1; W[c01_idx, col] += 1
        col += 1

        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1; U[a_idx, col] -= 1
        V[e_idx, col] += 1; V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1

        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1; U[d_idx, col] -= 1
        V[g_idx, col] += 1; V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)

        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0

        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)

        # P1 = a*e
        U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P2 = b*g
        U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
        # P3 = c*e
        U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
        # P4 = d*g
        U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

    # Build the decomposition
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)

    assert col == rank, f"Expected rank {rank}, got {col}"

    if verify(U, V, W):
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
