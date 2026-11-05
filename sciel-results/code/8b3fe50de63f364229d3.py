import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            if val != 0:
                U[idx][col] += val
        for idx, val in v_coeffs.items():
            if val != 0:
                V[idx][col] += val
        for idx, val in w_coeffs.items():
            if val != 0:
                W[idx][col] += val
        col += 1

    # Strassen for 2x2x2:
    # m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e)
    # m5=(a+b)h, m6=(c-d)(f+h), m7=(b-c)(e+h)
    # C00=m1+m4-m5+m7, C01=m3+m5, C10=m2+m4, C11=m1-m2+m3+m6
    strassen_local = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]

    # Schoolbook 2x2x1
    schoolbook_221 = [
        ({0:1}, {0:1}, {0:1}),
        ({1:1}, {1:1}, {0:1}),
        ({2:1}, {0:1}, {1:1}),
        ({3:1}, {1:1}, {1:1}),
    ]

    for q in range(2):
        for s in range(3):
            if s < 2:
                for u_t, v_t, w_t in strassen_local:
                    u_g, v_g, w_g = {}, {}, {}
                    for idx, val in u_t.items():
                        li, lj = idx // 2, idx % 2
                        u_g[li * m + (2*q + lj)] = val
                    for idx, val in v_t.items():
                        lj, lk = idx // 2, idx % 2
                        v_g[(2*q + lj) * p + (2*s + lk)] = val
                    for idx, val in w_t.items():
                        li, lk = idx // 2, idx % 2
                        w_g[(2*s + lk) * n + li] = val
                    add_term(u_g, v_g, w_g)
            else:
                for u_t, v_t, w_t in schoolbook_221:
                    u_g, v_g, w_g = {}, {}, {}
                    for idx, val in u_t.items():
                        li, lj = idx // 2, idx % 2
                        u_g[li * m + (2*q + lj)] = val
                    for idx, val in v_t.items():
                        v_g[(2*q + idx) * p + 4] = val
                    for idx, val in w_t.items():
                        w_g[4 * n + idx] = val
                    add_term(u_g, v_g, w_g)

    # Verify: check that the decomposition is correct
    if time.time() < deadline - 5:
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            # Build tensor from decomposition
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
            # C[k*n+i] should equal sum over j of A[i*m+j] * B[j*p+k]
            # So T[i,j,k] should be 1 if k*n+i corresponds to the product, 0 otherwise
            # Actually: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # For matrix mult: C_{i,k} = sum_j A_{i,j} B_{j,k}
            # So the tensor element for (i,j,k) where i is row of A, j is inner, k is col of B
            # should be 1 if this is a valid triple, 0 otherwise.
            # More precisely: T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
            # contribute to C[i,k]. So T[i,j,k] = 1 for all valid (i,j,k) triples.
            # Wait, that's the schoolbook tensor. The tensor for matrix mult M_{n,m,p} has
            # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
            # Because C[i,k] = sum_j A[i,j] * B[j,k].
            # So the tensor is all ones!
            # Let's verify: sum_r U[i,r]*V[j,r]*W[k*n+i... 
            # Actually W is indexed by k_global = k_col * n + i_row.
            # So W[k_global, r] where k_global = gk * n + gi.
            # The tensor element T[i_global, j_global, k_global] = sum_r U[i_global,r]*V[j_global,r]*W[k_global,r]
            # This should equal 1 if i_global = i*m+j, j_global = j*p+k, k_global = k*n+i
            # for some valid i,j,k, and 0 otherwise.
            # Since every triple (i,j,k) with i<n, j<m, k<p is valid, T should be all 1s
            # for the "right" indexing.
            # Let's just check a few random cases.
            np.random.seed(42)
            ok = True
            for trial in range(20):
                i = np.random.randint(0, n)
                j = np.random.randint(0, m)
                k = np.random.randint(0, p)
                i_g = i * m + j
                j_g = j * p + k
                k_g = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U_np[i_g, r] * V_np[j_g, r] * W_np[k_g, r]
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                # Fallback: return schoolbook rank-40
                rank40 = n * m * p
                U40 = [[0] * rank40 for _ in range(n * m)]
                V40 = [[0] * rank40 for _ in range(m * p)]
                W40 = [[0] * rank40 for _ in range(p * n)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = i * m * p + j * p + k
                            U40[i * m + j][r] = 1
                            V40[j * p + k][r] = 1
                            W40[k * n + i][r] = 1
                return {"U": U40, "V": V40, "W": W40}
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
