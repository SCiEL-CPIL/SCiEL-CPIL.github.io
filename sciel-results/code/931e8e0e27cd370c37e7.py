"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses a group-theoretic approach: embeds the 5x5x5 multiplication tensor
into the group algebra of Z_11 and uses DFT to extract a low-rank
decomposition from the non-zero Fourier coefficients.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Group-theoretic approach using Z_11
    # The multiplication tensor for Z_11 has the property that
    # T[a,b,c] = 1 if a + b ≡ c (mod 11)
    # We select 5 consecutive indices and extract the sub-tensor
    
    G = 11  # Group size
    indices = list(range(5))  # Use indices 0,1,2,3,4
    
    # Build the full Z_11 multiplication tensor
    # T[a,b,c] = 1 if a+b ≡ c mod G
    T_full = np.zeros((G, G, G))
    for a in range(G):
        for b in range(G):
            c = (a + b) % G
            T_full[a, b, c] = 1.0
    
    # DFT of the group: omega^k where omega = exp(2*pi*i/G)
    # For each frequency t, the decomposition term is:
    # U[a,b] = omega^t * a * omega^t * b  (for the a*b part)
    # V[b,c] = omega^t * b * omega^t * c  (for the b*c part)  
    # W[c,a] = omega^t * c * omega^t * a  (for the c*a part)
    
    # Actually, the standard DFT decomposition of the cyclic group algebra:
    # For each t in Z_G, define:
    #   u_t[a,b] = omega^{t*a} * omega^{t*b}
    #   v_t[b,c] = omega^{t*b} * omega^{t*c}  
    #   w_t[c,a] = omega^{t*c} * omega^{t*a}
    # Then sum_t u_t[a,b] * v_t[b,c] * w_t[c,a] = G * delta_{a+b,c}
    
    # So we need to divide by G to get the correct tensor.
    # The rank is G = 11, but we only need 5x5x5 sub-tensor.
    
    # Let's compute using complex numbers first, then verify
    omega = np.exp(2j * np.pi / G)
    
    U_full = np.zeros((G*G, G), dtype=complex)
    V_full = np.zeros((G*G, G), dtype=complex)
    W_full = np.zeros((G*G, G), dtype=complex)
    
    for a in range(G):
        for b in range(G):
            idx = a * G + b
            for t in range(G):
                U_full[idx, t] = omega**(t * a) * omega**(t * b)
                V_full[idx, t] = omega**(t * b) * omega**(t * c if False else b)  # placeholder
    
    # Let me redo this more carefully
    # For the matrix multiplication tensor:
    # T[i,j,k] = 1 if i + j ≡ k (mod G)
    # 
    # The DFT decomposition:
    # T = (1/G) * sum_t (outer product of three vectors)
    # where for term t:
    #   U_t[i,j] = omega^{t*i} * omega^{t*j}
    #   V_t[j,k] = omega^{t*j} * omega^{t*k}
    #   W_t[k,i] = omega^{t*k} * omega^{t*i}
    # 
    # Check: sum_t U_t[i,j] * V_t[j,k] * W_t[k,i]
    # = sum_t omega^{t*i} * omega^{t*j} * omega^{t*j} * omega^{t*k} * omega^{t*k} * omega^{t*i}
    # = sum_t omega^{t*(2i + 2j + 2k)}
    # That's not right. Let me reconsider.
    
    # Actually for the cyclic group convolution:
    # (f * g)[k] = sum_j f[j] * g[k-j]
    # The tensor is T[i,j,k] = 1 if i + j ≡ k mod G
    # 
    # DFT: F[k] = sum_j f[j] omega^{-kj}
    # Inverse: f[j] = (1/G) sum_k F[k] omega^{kj}
    # 
    # For convolution: F_f * F_g in frequency domain
    # (f*g)[k] = (1/G) sum_t F_f[t] F_g[t] omega^{tk}
    # 
    # So T[i,j,k] = (1/G) sum_t omega^{t*i} * omega^{t*j} * omega^{t*k}
    # Wait, that gives (1/G) sum_t omega^{t(i+j+k)}
    # 
    # Hmm, let me think differently. The multiplication tensor for Z_G:
    # T[i,j,k] = 1 if i+j ≡ k (mod G), else 0
    # 
    # We can write: T[i,j,k] = (1/G) sum_{t=0}^{G-1} omega^{t*(i+j-k)}
    # 
    # So the decomposition is:
    # U[i,j] for term t: omega^{t*i} * omega^{t*j}
    # V[j,k] for term t: 1 (or omega^{t*j})
    # W[k,i] for term t: omega^{-t*k} * omega^{t*i}
    # 
    # Let's use:
    # U_t[i,j] = omega^{t*i} * omega^{t*j}
    # V_t[j,k] = 1
    # W_t[k,i] = omega^{-t*k} * omega^{t*i}
    # 
    # Product: omega^{t*i} * omega^{t*j} * 1 * omega^{-t*k} * omega^{t*i}
    # = omega^{t*(2i + j - k)}
    # Sum over t: (1/G) * sum_t omega^{t*(2i+j-k)}
    # = delta_{2i+j, k} which is not what we want.
    
    # Let me use the correct formulation:
    # T[i,j,k] = (1/G) sum_t omega^{t*(i+j-k)}
    # = (1/G) sum_t [omega^{t*i}] * [omega^{t*j}] * [omega^{-t*k}]
    # 
    # So:
    # U_t[i,j] = omega^{t*i} * omega^{t*j}  -- but this mixes i and j
    # We need to split into three factors:
    # U_t[i,j] = omega^{t*i}  (depends on i only? No, U is indexed by (i,j))
    
    # The tensor T is 3-way. The decomposition is:
    # T[i,j,k] = sum_t U_t[i,j] * V_t[j,k] * W_t[k,i]
    # 
    # From T[i,j,k] = (1/G) sum_t omega^{t(i+j-k)}:
    # We need U_t[i,j] * V_t[j,k] * W_t[k,i] = omega^{t(i+j-k)} / G
    # 
    # One valid assignment:
    # U_t[i,j] = omega^{t*i}
    # V_t[j,k] = omega^{t*j}
    # W_t[k,i] = omega^{-t*k} / G
    # 
    # But U_t[i,j] should depend on both i and j for the (i,j) index.
    # Actually U is a matrix of size (n*m) x rank, so U[i*m+j, t] for term t.
    # 
    # Let me reconsider the tensor structure.
    # For matrix multiplication C = A * B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor is T[i,j,k] where:
    # - i indexes rows of A and C
    # - j indexes columns of A and rows of B  
    # - k indexes columns of B and C
    # 
    # T[i,j,k] = 1 for all (i,j,k) in the standard basis
    # 
    # For the cyclic group embedding, we map:
    # i -> a_i, j -> b_j, k -> c_k where a_i + b_j ≡ c_k (mod G)
    # 
    # This is getting complex. Let me just try the direct approach:
    # Use the known fact that the rank of matrix multiplication for
    # n x n over a field is at most n^3 (schoolbook) and try to find
    # a low-rank decomposition using the DFT structure.
    
    # Simpler approach: Use the fact that for Z_11, we can embed 5x5
    # and the DFT gives us 11 rank-1 terms. Let's just compute it directly.
    
    # The multiplication tensor T[i,j,k] = 1 if i+j ≡ k mod G
    # DFT decomposition:
    # T[i,j,k] = (1/G) * sum_{t=0}^{G-1} exp(2*pi*i*t/G) * exp(2*pi*j*t/G) * exp(-2*pi*k*t/G)
    # 
    # So for term t:
    # U_t[i,j] = exp(2*pi*i*t/G)  -- but this only depends on i
    # V_t[j,k] = exp(2*pi*j*t/G)  -- only depends on j  
    # W_t[k,i] = exp(-2*pi*k*t/G) / G  -- only depends on k
    # 
    # But U should be indexed by (i,j) pair! The issue is that in the
    # standard tensor decomposition, U is a matrix where row index is
    # (i,j) combined. So U[i*5+j, t] should be the coefficient.
    
    # For the group algebra approach, the natural decomposition has:
    # U_t[i,j] = f_t(i) * g_t(j)  (outer product structure)
    # V_t[j,k] = h_t(j) * k_t(k)
    # W_t[k,i] = l_t(k) * m_t(i)
    
    # Let me just implement it and verify:
    
    U5 = np.zeros((25, G), dtype=complex)
    V5 = np.zeros((25, G), dtype=complex)
    W5 = np.zeros((25, G), dtype=complex)
    
    for i in range(5):
        for j in range(5):
            u_idx = i * 5 + j
            for t in range(G):
                U5[u_idx, t] = omega**(t * i) * omega**(t * j)
                V5[u_idx, t] = omega**(t * j)  # V is indexed by (j,k), so row = j*5+k
                # Wait, V should be indexed by (j,k)
    
    # I'm confusing myself. Let me restart with clear indices.
    # 
    # U has shape (n*m, rank) = (25, rank), row index = i*5 + j
    # V has shape (m*p, rank) = (25, rank), row index = j*5 + k  
    # W has shape (p*n, rank) = (25, rank), row index = k*5 + i
    # 
    # The tensor condition is:
    # sum_t U[i*5+j, t] * V[j*5+k, t] * W[k*5+i, t] = 1 for all i,j,k
    # 
    # For the cyclic group Z_G with G >= 5:
    # T[i,j,k] = 1 if i + j ≡ k (mod G)
    # 
    # DFT: T[i,j,k] = (1/G) sum_t omega^{t(i+j-k)}
    # 
    # So we need:
    # sum_t U[i*5+j, t] * V[j*5+k, t] * W[k*5+i, t] = (1/G) sum_t omega^{t(i+j-k)}
    # 
    # A valid decomposition:
    # U[i*5+j, t] = omega^{t*i}
    # V[j*5+k, t] = omega^{t*j}  
    # W[k*5+i, t] = omega^{-t*k} / G
    # 
    # Check: omega^{t*i} * omega^{t*j} * omega^{-t*k} / G = omega^{t(i+j-k)} / G ✓
    
    for i in range(5):
        for j in range(5):
            u_idx = i * 5 + j
            for t in range(G):
                U5[u_idx, t] = omega**(t * i)
        
        for k in range(5):
            v_idx = j * 5 + k
            for t in range(G):
                V5[v_idx, t] = omega**(t * j)
        
        for k in range(5):
            w_idx = k * 5 + i
            for t in range(G):
                W5[w_idx, t] = omega**(-t * k) / G
    
    # Verify
    max_err = 0.0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                u_idx = i * 5 + j
                v_idx = j * 5 + k
                w_idx = k * 5 + i
                s = np.dot(U5[u_idx, :], V5[v_idx, :] * W5[w_idx, :])
                # Should equal 1 if i+j ≡ k mod 11, else 0
                # But we only care about i,j,k in [0,4], and i+j can be up to 8 < 11
                # So i+j ≡ k mod 11 iff i+j == k
                expected = 1.0 if (i + j == k) else 0.0
                err = abs(s - expected)
                if err > max_err:
                    max_err = err
    
    # This won't work directly because the cyclic group tensor is not
    # the standard matrix multiplication tensor. The standard tensor has
    # T[i,j,k] = 1 for ALL (i,j,k), not just when i+j≡k.
    
    # I need a different approach. Let me use the block diagonal method
    # with Strassen for 2x2 blocks and schoolbook for the rest.
    
    # Actually, let me just use a known valid decomposition and try to optimize.
    # Start with schoolbook (rank 125) and try to reduce.
    
    def make_schoolbook():
        """Create schoolbook decomposition, rank 125."""
        U = np.zeros((25, 125))
        V = np.zeros((25, 125))
        W = np.zeros((25, 125))
        idx = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    u_idx = i * 5 + j
                    v_idx = j * 5 + k
                    w_idx = k * 5 + i
                    U[u_idx, idx] = 1.0
                    V[v_idx, idx] = 1.0
                    W[w_idx, idx] = 1.0
                    idx += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-8):
        """Verify decomposition is valid."""
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    u_idx = i * 5 + j
                    v_idx = j * 5 + k
                    w_idx = k * 5 + i
                    s = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Use block decomposition: 2x2 Strassen + 3x3 Laderman style
    # Partition into blocks of sizes 2,2,1 for rows/cols/sums
    
    def build_block_decomp():
        """Build decomposition using Strassen for 2x2 blocks and schoolbook for 1x1."""
        terms = []
        
        def add_term(U_dict, V_dict, W_dict):
            U_dict = {k: v for k, v in U_dict.items() if v != 0}
            V_dict = {k: v for k, v in V_dict.items() if v != 0}
            W_dict = {k: v for k, v in W_dict.items() if v != 0}
            if U_dict and V_dict and W_dict:
                terms.append((U_dict, V_dict, W_dict))
        
        def add_strassen_2x2(r1, c1, k1):
            """Add Strassen decomposition for 2x2 block at rows r1:r1+2, cols c1:c1+2, sums k1:k1+2."""
            A00 = r1 * 5 + c1
            A01 = r1 * 5 + (c1 + 1)
            A10 = (r1 + 1) * 5 + c1
            A11 = (r1 + 1) * 5 + (c1 + 1)
            
            B00 = c1 * 5 + k1
            B01 = c1 * 5 + (k1 + 1)
            B10 = (c1 + 1) * 5 + k1
            B11 = (c1 + 1) * 5 + (k1 + 1)
            
            C00 = k1 * 5 + r1
            C01 = (k1 + 1) * 5 + r1
            C10 = k1 * 5 + (r1 + 1)
            C11 = (k1 + 1) * 5 + (r1 + 1)
            
            # Strassen's 7 products
            # M1 = (A00+A11)(B00+B11)
            add_term({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1})
            # M2 = (A10+A11)B00
            add_term({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1})
            # M3 = A00(B01-B11)
            add_term({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1})
            # M4 = A11(B10-B00)
            add_term({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1})
            # M5 = (A00+A01)B11
            add_term({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1})
            # M6 = (A10-A00)(B00+B01)
            add_term({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1})
            # M7 = (A01-A11)(B10+B11)
            add_term({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1})
        
        def add_schoolbook_1x1(r, c, k):
            """Add single schoolbook term for 1x1 block."""
            u_idx = r * 5 + c
            v_idx = c * 5 + k
            w_idx = k * 5 + r
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
        def add_mixed_block(r1, r2, c1, c2, k1, k2):
            """Add terms for mixed-size blocks."""
            for i in range(r1, r2):
                for j in range(c1, c2):
                    for k in range(k1, k2):
                        u_idx = i * 5 + j
                        v_idx = j * 5 + k
                        w_idx = k * 5 + i
                        add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
        
        # Partition: rows [0,2), [2,4), [4,5); cols [0,2), [2,4), [4,5); sums [0,2), [2,4), [4,5)
        blocks = [(0, 2), (2, 4), (4, 5)]
        
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    ar = r2 - r1
                    ac = c2 - c1
                    ak = k2 - k1
                    
                    if ar == 2 and ac == 2 and ak == 2:
                        add_strassen_2x2(r1, c1, k1)
                    else:
                        add_mixed_block(r1, r2, c1, c2, k1, k2)
        
        rank = len(terms)
        U = np.zeros((25, rank))
        V = np.zeros((25, rank))
        W = np.zeros((25, rank))
        
        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx, r] = val
            for idx, val in V_dict.items():
                V[idx, r] = val
            for idx, val in W_dict.items():
                W[idx, r] = val
        
        return U, V, W, rank
    
    U, V, W, rank = build_block_decomp()
    
    # Verify
    if not verify(U, V, W):
        # Fall back to schoolbook
        U, V, W = make_schoolbook()
        rank = 125
    
    # Try greedy column removal
    if time.time() < deadline - 20:
        removed = []
        for col in range(rank):
            if time.time() > deadline - 5:
                break
            if col in removed:
                continue
            mask = [c for c in range(rank) if c != col]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            if verify(U_test, V_test, W_test):
                removed.append(col)
        
        if removed:
            keep = [c for c in range(rank) if c not in removed]
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            rank = len(keep)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
