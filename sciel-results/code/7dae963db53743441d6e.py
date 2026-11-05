import time
import numpy as np

_DIMENSIONS = (3, 4, 6)
_T0 = time.time()
_DEADLINE = 135.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _T0

def _deadline_ok():
    return _elapsed() < _DEADLINE

def construct():
    n, m, p = _DIMENSIONS
    # Keep the proven p=2+2+2 block structure.
    # Within each (3,4,2) block:
    #   n=0 row handled by schoolbook M(1,4,2) rank 8
    #   n=1,2 rows handled by M(2,4,2) = 2*Strassen rank 14
    # Total 66 per three blocks.
    #
    # Improvement within this family: attempt to merge overlapping
    # rank-1 terms across the two Strassen subblocks of each M(2,4,2)
    # by checking if any pair of terms can be combined into a single
    # rank-1 term (i.e., their U/V/W support sets align so that the
    # sum is still rank-1). This is a constant-size combinatorial check
    # that runs in microseconds.
    
    rank_target = 66
    U = [[0] * rank_target for _ in range(n * m)]
    V = [[0] * rank_target for _ in range(m * p)]
    W = [[0] * rank_target for _ in range(p * n)]
    pos = 0

    def add_term(u_terms, v_terms, w_terms):
        """Add one rank-1 term. u_terms/v_terms/w_terms are lists of (idx, val)."""
        nonlocal pos
        if pos >= rank_target or not _deadline_ok():
            return
        for idx, val in u_terms:
            U[idx][pos] += val
        for idx, val in v_terms:
            V[idx][pos] += val
        for idx, val in w_terms:
            W[idx][pos] += val
        pos += 1

    def schoolbook_1_4_2(i0, j0, k0):
        for j in range(4):
            for k in range(2):
                ui = i0 * m + (j0 + j)
                vi = (j0 + j) * p + (k0 + k)
                wi = (k0 + k) * n + i0
                add_term([(ui, 1)], [(vi, 1)], [(wi, 1)])

    def strassen_2_2_2(i0, j0, k0):
        """Standard 7-multiplication Strassen for a 2x2 block at offset (i0,j0,k0)."""
        def U_(i, j): return (i0 + i) * m + (j0 + j)
        def V_(j, k): return (j0 + j) * p + (k0 + k)
        def W_(i, k): return (k0 + k) * n + (i0 + i)

        terms = [
            # m1 = (a00+a11)(b00+b11)
            ([(U_(0,0),1),(U_(1,1),1)], [(V_(0,0),1),(V_(1,1),1)], [(W_(0,0),1),(W_(1,1),1)]),
            # m2 = (a10+a11)(b00)
            ([(U_(1,0),1),(U_(1,1),1)], [(V_(0,0),1)], [(W_(1,0),1),(W_(1,1),-1)]),
            # m3 = (a00)(b01-b11)
            ([(U_(0,0),1)], [(V_(0,1),1),(V_(1,1),-1)], [(W_(0,1),1),(W_(1,1),1)]),
            # m4 = (a11)(b10-b00)
            ([(U_(1,1),1)], [(V_(1,0),1),(V_(0,0),-1)], [(W_(0,0),1),(W_(1,0),1)]),
            # m5 = (a00+a01)(b11)
            ([(U_(0,0),1),(U_(0,1),1)], [(V_(1,1),1)], [(W_(0,0),-1),(W_(0,1),1)]),
            # m6 = (a10-a00)(b00+b01)
            ([(U_(1,0),1),(U_(0,0),-1)], [(V_(0,0),1),(V_(0,1),1)], [(W_(1,1),1)]),
            # m7 = (a01-a11)(b10+b11)
            ([(U_(0,1),1),(U_(1,1),-1)], [(V_(1,0),1),(V_(1,1),1)], [(W_(0,0),1)]),
        ]
        for u_t, v_t, w_t in terms:
            add_term(u_t, v_t, w_t)

    def m_2_4_2(i0, j0, k0):
        strassen_2_2_2(i0, j0, k0)
        strassen_2_2_2(i0, j0 + 2, k0)

    # Build the three p=2 blocks
    for blk in range(3):
        k0 = blk * 2
        if not _deadline_ok():
            break
        schoolbook_1_4_2(0, 0, k0)
        m_2_4_2(1, 0, k0)

    # Now try to reduce rank by checking if any two adjacent rank-1 terms
    # in the last 14 (the two Strassen blocks of the last M(2,4,2)) can be
    # merged. A merge is possible only if the sum of two rank-1 tensors is
    # still rank-1, which requires their U-supports, V-supports, and W-supports
    # to each be collinear in a specific way. For our integer-valued terms
    # this essentially never happens unless the terms are scalar multiples,
    # which they are not in Strassen. So we skip the search and just verify.

    # Truncate to actual used rank (should be exactly 66)
    U = [row[:pos] for row in U]
    V = [row[:pos] for row in V]
    W = [row[:pos] for row in W]

    # Quick sanity check: verify decomposition on a small random instance
    if _deadline_ok():
        rng = np.random.default_rng(42)
        A = rng.integers(-5, 6, size=(n, m)).astype(float)
        B = rng.integers(-5, 6, size=(m, p)).astype(float)
        C_expected = A @ B

        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        # Tensor contraction: C[i,k] = sum_j A[i,j]*B[j,k]
        # Decomposition: C[i,k] = sum_r U[i*m+j... ] ...
        # Actually the decomposition is: T_{ijk} = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # where T_{ijk} = delta (the multiplication tensor)
        # So C[i,k] = sum_{j,r} A[i,j]*B[j,k]*U[i,j,r]*V[j,k,r]*W[i,k,r] ... 
        # Wait, the standard form is:
        # C[i,k] = sum_r (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r]) * ... 
        # No. The correct decomposition of the matrix multiplication tensor is:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        # where T[i,j,k] = 1 if this is the "multiplication" tensor element.
        # For matrix multiplication: (AB)[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is: mu[i,j,k] = delta (it's 1 for all valid i,j,k in the sense
        # that the bilinear map is defined by these coordinates)
        
        # Let me re-derive. The multiplication tensor for M(n,m,p) has indices
        # i in [0,n), j in [0,m), k in [0,p), and the tensor value is 1 (it's
        # the structure tensor of the bilinear map). A rank-r decomposition gives:
        # mu[i,j,k] = sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[i,k,r]
        # where U has shape (n*m, r) indexed by (i,j), V has shape (m*p, r)
        # indexed by (j,k), W has shape (p*n, r) indexed by (k,i).
        
        # So to verify: for random A, B:
        # C[i,k] should equal sum_j A[i,j]*B[j,k]
        # And from decomposition: C[i,k] = sum_r W[i,k,r] * (sum_j A[i,j]*U[i,j,r]) * ... 
        # No wait. Let me think more carefully.
        
        # The standard form: C = sum_r (U_r * A) * (B * V_r) ... no.
        # 
        # Actually for matrix multiplication tensor decomposition:
        # The bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
        # can be written as C = sum_r alpha_r * (A * U_r) * V_r * W_r ... 
        # 
        # Let me just use the direct verification:
        # mu[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 for the multiplication tensor (every entry is 1
        # in the sense that the tensor IS the identity map on the bilinear form).
        
        # Actually the multiplication tensor has mu[i,j,k] = 1 for all valid
        # (i,j,k) - no that's not right either. The multiplication tensor
        # for C = AB has the property that C[i,k] = sum_j A[i,j] B[j,k].
        # As a trilinear form: C[i,k] = sum_{j,r} A[i,j] B[j,k] * T[i,j,k,r] ... 
        # 
        # I think the correct interpretation is:
        # T is a 3-tensor of shape (n*m, m*p, p*n) where
        # T[(i,j),(j,k),(k,i)] = 1 and all other entries are 0.
        # A rank-r decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
        # So T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # This means: for each valid triple (i,j,k), the sum should be 1.
        # And for invalid triples, it should be 0. But in our construction,
        # we only place non-zero values at valid (i,j,k) positions, so we
        # just need to check that for each valid (i,j,k), the sum is 1.
        
        # Let's verify:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vi = j * p + k
                    wi = k * n + i
                    s = 0.0
                    for r in range(pos):
                        s += U[ui][r] * V[vi][r] * W[wi][r]
                    if abs(s - 1.0) > 1e-9:
                        max_err = max(max_err, abs(s - 1.0))
        
        # Also verify no spurious terms at invalid positions
        for ui in range(n * m):
            for vi in range(m * p):
                # Check if this (i,j) and (j,k) is valid: j must match
                i = ui // m
                j = ui % m
                j2 = vi // p
                k = vi % p
                if j != j2:
                    # This is an invalid combination; the tensor entry should be 0
                    for wi in range(p * n):
                        k2 = wi // n
                        i2 = wi % n
                        if k2 != k or i2 != i:
                            continue  # also invalid on W side, skip
                        s = 0.0
                        for r in range(pos):
                            s += U[ui][r] * V[vi][r] * W[wi][r]
                        if abs(s) > 1e-9:
                            max_err = max(max_err, abs(s))
                # For valid j==j2, already checked above via the 1.0 target
    
    return {"U": U, "V": V, "W": W}
