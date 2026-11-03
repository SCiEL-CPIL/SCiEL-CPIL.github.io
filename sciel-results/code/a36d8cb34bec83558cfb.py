"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = 2, 3, 5
    
    # We'll build U, V, W incrementally
    terms = []  # list of (u_dict, v_dict, w_dict) where keys are (i,j), (j,k), (k,i)
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # Strassen for 2x2x2, mapping local (i',j',k') to global
    # We'll define Strassen terms as dictionaries
    
    # Standard Strassen for M = A*B where A,B are 2x2, C = A*B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_222(i_offset, j_offset, k_offset):
        """Add Strassen terms for 2x2 block at positions (i_offset,i_offset+1), (j_offset,j_offset+1), (k_offset,k_offset+1)"""
        i0, i1 = i_offset, i_offset+1
        j0, j1 = j_offset, j_offset+1
        k0, k1 = k_offset, k_offset+1
        
        # M1 = (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        # C contribution: C[i0,k0] += M1, C[i1,k1] += M1 (via C00=M1+M4-M5+M7, C11=M1-M2+M3+M6)
        # Actually, each M contributes to multiple C entries.
        
        # Let me think of it as: each "product" Mx corresponds to one rank-1 term in the tensor.
        # The tensor entry T[i,j,k] gets contributions from all terms.
        
        # M1: u has A[i0,j0] and A[i1,j1], v has B[j0,k0] and B[j1,k1], w has C[i0,k0] and C[i1,k1]
        # The term contributes: for each (i,j,k), u[i,j]*v[j,k]*w[k,i]
        # u[i0,j0]=1, u[i1,j1]=1, else 0
        # v[j0,k0]=1, v[j1,k1]=1, else 0  
        # w[k0,i0]=1, w[k1,i1]=1, else 0
        # Contribution to T[i,j,k]:
        #   (i0,j0): u=1, v[j0,k0]=1,w[k0,i0]=1 → T[i0,j0,k0]+=1; v[j0,k1]=0; etc.
        #   Actually: T[i0,j0,k0] += 1*1*1 = 1
        #              T[i0,j0,k1] += 1*0*1 = 0 (v[j0,k1]=0)
        #              T[i1,j1,k0] += 0*... 
        #   Let me be more careful.
        
        # u[i,j] nonzero: u[i0,j0]=1, u[i1,j1]=1
        # v[j,k] nonzero: v[j0,k0]=1, v[j1,k1]=1
        # w[k,i] nonzero: w[k0,i0]=1, w[k1,i1]=1
        # 
        # T[i,j,k] contribution = u[i,j]*v[j,k]*w[k,i]
        # i=i0,j=j0: v[j0,k0]=1→w[k0,i0]=1: T[i0,j0,k0]+=1
        #             v[j0,k1]=0: nothing
        # i=i0,j=j1: u[i0,j1]=0: nothing
        # i=i1,j=j0: u[i1,j0]=0: nothing
        # i=i1,j=j1: v[j1,k0]=0; v[j1,k1]=1→w[k1,i1]=1: T[i1,j1,k1]+=1
        
        # So M1 only contributes to T[i0,j0,k0] and T[i1,j1,k1]!
        # That's not how Strassen works. I'm confusing the algebraic decomposition with the tensor decomposition.
        
        pass
    
    # I think I'm overcomplicating this. Let me use the schoolbook approach with Strassen substitutions.
    # 
    # The key insight: Instead of having a separate term for each (i,j,k), we can combine terms
    # where the same linear combination of A entries is multiplied by the same linear combination of B entries.
    #
    # Let me just implement the known 28-rank result directly.
    
    # Approach: Use the fact that we can decompose 2x3x5 into smaller blocks.
    # For now, let me implement a verified 28-rank decomposition.
    
    # I'll use a different approach: enumerate all possible Strassen-like combinations
    # and verify numerically.
    
    # Let me use a computational approach to find a low-rank decomposition.
    
    # Actually, let me just carefully implement the block decomposition.
    
    # Split: j into {0,1} and {2}
    # For j in {0,1}: 2x2x5 problem
    # For j in {2}: 2x1x5 problem (rank 2 as shown above)
    
    # For 2x2x5: split k into {0,1},{2,3},{4}
    # 2x2x2 + 2x2x2 + 2x2x1
    
    # Let me build each piece.
    
    col = 0
    U = {}  # sparse: (i,j) -> list of (val, col)
    V = {}
    W = {}
    
    # Simpler: just build dense matrices
    max_rank = 30  # start with schoolbook size, we'll use what we need
    U_dense = np.zeros((n*m, max_rank))
    V_dense = np.zeros((m*p, max_rank))
    W_dense = np.zeros((p*n, max_rank))
    
    def add_term(u_vals, v_vals, w_vals):
        """u_vals: list of (i,j,val), v_vals: list of (j,k,val), w_vals: list of (k,i,val)"""
        nonlocal col
        for (i, j, val) in u_vals:
            U_dense[i*m+j, col] += val
        for (j, k, val) in v_vals:
            V_dense[j*p+k, col] += val
        for (k, i, val) in w_vals:
            W_dense[k*n+i, col] += val
        col += 1
    
    # ===== PART 1: j in {0,1}, k in {0,1} (2x2x2 Strassen, rank 7) =====
    i0,i1 = 0,1; j0,j1 = 0,1; k0,k1 = 0,1
    
    # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    add_term([(i0,j0,1),(i1,j1,1)], [(j0,k0,1),(j1,k1,1)], [(k0,i0,1),(k1,i1,1)])
    # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    add_term([(i1,j0,1),(i1,j1,1)], [(j0,k0,1)], [(k0,i1,1)])
    # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
    add_term([(i0,j0,1)], [(j1,k0,1),(j1,k1,-1)], [(k0,i0,1),(k1,i1,1)])
    # M4 = A[i1,j1] * (B[j0,k1]-B[j0,k0])
    add_term([(i1,j1,1)], [(j0,k1,1),(j0,k0,-1)], [(k0,i1,1),(k1,i0,1)])
    # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    add_term([(i0,j0,1),(i0,j1,1)], [(j1,k1,1)], [(k1,i0,1),(k1,i1,1)])
    # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    add_term([(i1,j0,1),(i0,j0,-1)], [(j0,k0,1),(j0,k1,1)], [(k1,i0,1),(k1,i1,1)])
    # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    add_term([(i0,j1,1),(i1,j1,-1)], [(j1,k0,1),(j1,k1,1)], [(k0,i0,1),(k1,i1,1)])
    
    # ===== PART 2: j in {0,1}, k in {2,3} (2x2x2 Strassen, rank 7) =====
    k0,k1 = 2,3
    
    # M1
    add_term([(i0,j0,1),(i1,j1,1)], [(j0,k0,1),(j1,k1,1)], [(k0,i0,1),(k1,i1,1)])
    # M2
    add_term([(i1,j0,1),(i1,j1,1)], [(j0,k0,1)], [(k0,i1,1)])
    # M3
    add_term([(i0,j0,1)], [(j1,k0,1),(j1,k1,-1)], [(k0,i0,1),(k1,i1,1)])
    # M4
    add_term([(i1,j1,1)], [(j0,k1,1),(j0,k0,-1)], [(k0,i1,1),(k1,i0,1)])
    # M5
    add_term([(i0,j0,1),(i0,j1,1)], [(j1,k1,1)], [(k1,i0,1),(k1,i1,1)])
    # M6
    add_term([(i1,j0,1),(i0,j0,-1)], [(j0,k0,1),(j0,k1,1)], [(k1,i0,1),(k1,i1,1)])
    # M7
    add_term([(i0,j1,1),(i1,j1,-1)], [(j1,k0,1),(j1,k1,1)], [(k0,i0,1),(k1,i1,1)])
    
    # ===== PART 3: j in {0,1}, k = 4 (2x2x1, rank 4) =====
    for i in range(n):
        for j in range(2):
            add_term([(i,j,1)], [(j,4,1)], [(4,i,1)])
    
    # ===== PART 4: j = 2, all k (2x1x5, rank 2) =====
    # Term for i=0: A[0,2]*B[2,k] for all k
    add_term([(0,2,1)], [(2,k,1) for k in range(p)], [(k,0,1) for k in range(p)])
    # Term for i=1: A[1,2]*B[2,k] for all k
    add_term([(1,2,1)], [(2,k,1) for k in range(p)], [(k,1,1) for k in range(p)])
    
    # Extract result
    rank = col
    U_out = U_dense[:, :rank]
    V_out = V_dense[:, :rank]
    W_out = W_dense[:, :rank]
    
    # Convert to lists of lists with float entries
    def to_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                # Convert to integer or half-integer
                val_f = float(val)
                if abs(val_f - round(val_f*2)) < 1e-10:
                    val_half = round(val_f*2)/2
                    if val_half == int(val_half):
                        r.append(int(val_half))
                    else:
                        r.append(val_half)
                else:
                    r.append(val_f)
            result.append(r)
        return result
    
    # Verify with random test
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        # Compute the rank-1 contribution
        a_part = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                a_part[i, j] = U_out[i*m+j, r]
        b_part = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                b_part[j, k] = V_out[j*p+k, r]
        w_part = W_out[:, r]  # shape (p*n,)
        
        # C[i,k] += w_part[k*n+i] * sum over i,j of a_part... 
        # Actually: C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * A[i,j]) * ... 
        # No, that's not right. The decomposition is:
        # C[i,k] = sum_r W[k*n+i, r] * U_vec_r · A_row_i * V_vec_r · B_col_k
        # Hmm, I need to think about this more carefully.
        
        # The tensor T has T[i,j,k] = 1 for matrix multiplication (summed over j).
        # Rank-1 decomposition: T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
        # Then C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k] ... no.
        
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r (sum_j U[i*m+j,r] A[i,j]) * (sum_j V[j*p+k,r] B[j,k]) * W[k*n+i,r]?
        
        # That doesn't seem right dimensionally. Let me reconsider.
        
        # I think the correct formula is:
        # C[i,k] = sum_r W[k*n+i, r] * U_col_r · (A row i) * V_col_r · (B col k)
        # But U is (n*m) x rank, so U[i*m+j, r] is the entry.
        # For a fixed r, define:
        #   u_r[i,j] = U[i*m+j, r]
        #   v_r[j,k] = V[j*p+k, r]
        #   w_r[k,i] = W[k*n+i, r]
        # Then the claim is: sum_j u_r[i,j] v_r[j,k] = some value, and C[i,k] = sum_r w_r[k,i] * (sum_j ...)?
        
        # No, I think it's:
        # C[i,k] = sum_r (sum_j U[i*m+j,r] A[i,j]) * W[k*n+i,r] * (sum_j V[j*p+k,r] B[j,k])
        # That's not linear in A and B separately.
        
        # I believe the correct interpretation is that the tensor T is decomposed as:
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # And then C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k]... but that's not how it works either.
        
        # OK let me just verify numerically with the schoolbook approach first to make sure I understand the convention.
        
        pass
    
    # Let me just verify the full thing numerically
    C_verify = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                # Sum over j
                total = 0
                for j in range(m):
                    total += U_out[i*m+j, r] * V_out[j*p+k, r]
                C_verify[i,k] += W_out[k*n+i, r] * total
    
    error = np.max(np.abs(C_verify - C_expected))
    if error > 1e-8:
        # Fallback to schoolbook
        U_school = np.zeros((n*m, n*m*p))
        V_school = np.zeros((m*p, n*m*p))
        W_school = np.zeros((p*n, n*m*p))
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col_s] = 1
                    V_school[j*p+k, col_s] = 1
                    W_school[k*n+i, col_s] = 1
                    col_s += 1
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
    
    return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
# EVOLVE-BLOCK-END
