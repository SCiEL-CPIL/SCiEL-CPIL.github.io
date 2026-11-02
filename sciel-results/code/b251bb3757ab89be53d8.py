import numpy as np
import time

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return 7 Strassen multiplication terms as (u, v, w) index sets.
    
    Each term t is (u_set, v_set, w_set) where each set contains (row, col) tuples
    with their coefficients. The convention is:
      A[i,k] -> u indices (i,k)
      B[k,j] -> v indices (k,j)  
      C[i,j] -> w indices (i,j)
    And the product sums: C[i,j] = sum_t w_t[i,j] * (sum_ik u_t[i,k]*A[i,k]) * (sum_kj v_t[k,j]*B[k,j])
    """
    return [
        # P1=(a+d)(e+h)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2=(c+d)e
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # P3=a(f-h)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P4=d(g-e)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P5=(a+b)h
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P6=(c-a)(e+f)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7=(b-d)(g+h)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

def _build_rank49_factors():
    """Build U,V,W matrices (as numpy int arrays) for rank-49 4x4 matmul.
    
    Convention: U has shape (16, 49) = (4*4, 7*7), indexed U[i*4+k, r]
    V has shape (16, 49) = (4*4, 7*7), indexed V[k*4+j, r]
    W has shape (16, 49) = (4*4, 7*7), indexed W[j*4+i, r]
    where r = t*7 + s for outer Strassen term t and inner term s.
    
    C[i,j] = sum_r W[j*4+i, r] * U[i*4+k, r] * A[i,k] * V[k*4+j, r] * B[k,j]
    Simplified: C[i,j] = sum_r W[j*4+i,r] * (sum_ik U[i*4+k,r]*A[i,k]) * (sum_kj V[k*4+j,r]*B[k,j])
    """
    n = 4
    outer_terms = _strassen_2x2_terms()
    inner_terms = _strassen_2x2_terms()
    outer_r = 7
    inner_r = 7
    total_r = outer_r * inner_r
    
    U = np.zeros((16, total_r), dtype=np.int64)
    V = np.zeros((16, total_r), dtype=np.int64)
    W = np.zeros((16, total_r), dtype=np.int64)
    
    for t_idx, (u_out, v_out, w_out) in enumerate(outer_terms):
        for s_idx, (u_in, v_in, w_in) in enumerate(inner_terms):
            r = t_idx * inner_r + s_idx
            
            # Fill U: for each outer block (br,bc) in u_out and inner (ir,ic) in u_in
            for (br, bc), coeff_o in u_out.items():
                for (ir, ic), coeff_i in u_in.items():
                    row_global = br * 2 + ir
                    col_global = bc * 2 + ic
                    U[row_global * 4 + col_global, r] += coeff_o * coeff_i
            
            # Fill V: for each outer block (br,bc) in v_out and inner (ir,ic) in v_in
            for (br, bc), coeff_o in v_out.items():
                for (ir, ic), coeff_i in v_in.items():
                    row_global = br * 2 + ir
                    col_global = bc * 2 + ic
                    V[row_global * 4 + col_global, r] += coeff_o * coeff_i
            
            # Fill W: for each outer block (br,bc) in w_out and inner (ir,ic) in w_in
            for (br, bc), coeff_o in w_out.items():
                for (ir, ic), coeff_i in w_in.items():
                    row_global = br * 2 + ir
                    col_global = bc * 2 + ic
                    W[col_global * 4 + row_global, r] += coeff_o * coeff_i
    
    return U, V, W

def _verify_decomposition(U, V, W, n=4, seed=42):
    """Verify the decomposition with random test matrices."""
    total_r = U.shape[1]
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    B = rng.randn(n, n)
    C_expected = A @ B
    
    U4 = U.reshape(n, n, total_r)
    V4 = V.reshape(n, n, total_r)
    W4 = W.reshape(n, n, total_r)
    
    u_lin = np.einsum('ikr,ik->ir', U4, A)
    v_lin = np.einsum('kjr,kj->jr', V4, B)
    C_comp = np.einsum('jir,ir,jr->ij', W4, u_lin, v_lin)
    
    return float(np.max(np.abs(C_comp - C_expected)))

def _build_fallback_rank64():
    """Trivial rank-64 (schoolbook) decomposition as numpy arrays."""
    n = 4
    r = 64
    U = np.zeros((n*n, r), dtype=np.int64)
    V = np.zeros((n*n, r), dtype=np.int64)
    W = np.zeros((n*n, r), dtype=np.int64)
    
    idx = 0
    for i in range(n):
        for k in range(n):
            for j in range(n):
                U[i*n + k, idx] = 1
                V[k*n + j, idx] = 1
                W[j*n + i, idx] = 1
                idx += 1
    
    return U, V, W

def construct():
    t0 = time.time()
    deadline = 145
    
    U, V, W = _build_rank49_factors()
    
    # Quick verification
    err = _verify_decomposition(U, V, W)
    
    if time.time() - t0 > deadline:
        raise RuntimeError("Deadline exceeded before verification")
    
    if err > 1e-9:
        U, V, W = _build_fallback_rank64()
        err = _verify_decomposition(U, V, W)
        if err > 1e-9:
            raise RuntimeError("Verification failed even for fallback")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
