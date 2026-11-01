import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 105
    
    n, m, p = DIMENSIONS
    
    # We'll build the decomposition using a structured approach.
    # The tensor T_{ijk} = e_{ik} for i in [0,n), j in [0,m), k in [0,p).
    # We want to express T as sum_{r} U_r ⊗ V_r ⊗ W_r where
    # U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}.
    # T_{ijk} = sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
    
    # Strategy: Use the fact that we're multiplying (2,5) x (5,5).
    # Think of the 5x5 matrix B as having 5 columns. Each column b_k gives
    # A @ b_k = c_k (a 2-vector).
    # 
    # Key insight: We can decompose the 5 columns of B into groups and use
    # rank-deficient decompositions within groups.
    #
    # Alternative: Use the approach of decomposing the tensor directly.
    # The tensor has entries T[i,j,k] = 1 if i==k, else 0 (for i in 0..1, j in 0..4, k in 0..4).
    # Wait, let me re-check. A is (n,m) = (2,5), B is (m,p) = (5,5).
    # (AB)[i,k] = sum_j A[i,j] B[j,k].
    # The multiplication tensor has entries T[i,j,k] such that the product is
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = (AB)[i,k]... 
    # Actually the standard tensor decomposition: 
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor T has T[i,j,k] = delta_{ik}? No.
    # T[i,j,k] is the coefficient of A[i,j]*B[j,k] in the product.
    # (AB)_{ik} = sum_j A_{ij} B_{jk}, so T[i,j,k] = 1 if the (i,k) entry gets contribution from (j), 
    # but the tensor is indexed differently.
    
    # Let me use the convention from the task:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: for all i,j,k: 
    #   delta_{i==k} (for the (i,k) output) ... 
    # Actually, the tensor for matrix multiplication M_{n,m,p} has:
    #   M[i,j,k] = 1 if the (i,k) component of the product uses A[i,j]*B[j,k]
    # In the standard formulation: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor entries are: T[i, j, k] where the output index is (i,k)
    # and input indices are (i,j) for A and (j,k) for B.
    # 
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k,i) -> row = k*n+i.
    # U is indexed by (i,j) -> row = i*m+j (shape n*m)
    # V is indexed by (j,k) -> row = j*p+k (shape m*p)
    
    # The decomposition requires:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{ik} * 1  ... wait
    # Actually it should equal the tensor entry which is 1 when we're computing (i,k) from j.
    # The product (AB)_{ik} = sum_j A_{ij} B_{jk}.
    # So for a fixed (i,k), the contribution from index j is A_{ij} * B_{jk} * 1.
    # The tensor T has T[(i,j), (j,k), (k,i)] = 1 for the appropriate terms.
    # More precisely: T[i,j,k] = 1 if we're at position (i,j) in A and (j,k) in B 
    # contributing to output (i,k). This is always 1 for valid indices.
    # Wait no - the tensor is: for each triple (i,j,k) with i in [0,n), j in [0,m), k in [0,p),
    # T[i,j,k] = 1 (it's the structure tensor of matrix multiplication).
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where W[k,i,r] means W at row (k*n+i), column r.
    
    # So we need: for all i in [0,2), j in [0,5), k in [0,5):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait, that's not right either. The tensor for 2x5*5x5 multiplication is:
    # T[i,j,k] = delta_{i,k}? No.
    
    # Let me think again. Matrix multiplication C = A*B where A is (n,m), B is (m,p), C is (n,p).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The structure tensor T in R^{(n,m) x (m,p) x (n,p)} has:
    # T[i,j, k, i_out] ... this is getting confusing with 4 indices.
    
    # In the standard bilinear form: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is in R^{nm x mp x np} (or R^{nm x mp x pn} with trace convention)
    # T[(i,j), (j',k), (k',i')] = 1 if i==i', j==j', k==k' ... no.
    
    # Actually: T is a 3-tensor. T[a, b, c] where a indexes A's entries (i,j), 
    # b indexes B's entries (j,k), c indexes C's entries (i,k) or (k,i).
    # T[(i,j), (j,k), (i,k)] = 1 (using standard convention)
    # With trace convention W[k*n+i]: T[(i,j), (j,k), (k,i)] = 1
    
    # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let me verify this makes sense. If we pick a specific (i,j,k), the sum over r 
    # of the product should equal 1. For a different combination of (i',j',k') that doesn't 
    # form a valid "path" in the multiplication, it should also work out because 
    # the tensor is defined as 1 for ALL valid (i,j,k) triples (since every A[i,j]*B[j,k] 
    # contributes to exactly one C[i,k]).
    
    # Wait, I need to be more careful. The tensor T for matrix multiplication satisfies:
    # T[i,j,k] = 1 for all valid i,j,k (where i in [0,n), j in [0,m), k in [0,p)).
    # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need rank-minimal U, V, W such that for ALL i in [0,2), j in [0,5), k in [0,5):
    #   sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*2+i, r] = 1
    
    # This is a system of 2*5*5 = 50 equations in the unknowns (rank * 10 + rank * 25 + rank * 10 = 45*rank variables).
    # But it's nonlinear (trilinear). The rank is the minimum r such that this system is solvable.
    
    # Current best: rank 46. Let me try to find a better decomposition.
    
    # Approach: Try to use a structured decomposition.
    # Note that n=2 is small. Let's think of it as two rows of the output.
    # For output row 0 (i=0): C[0,k] = sum_j A[0,j] B[j,k] for k=0..4
    # For output row 1 (i=1): C[1,k] = sum_j A[1,j] B[j,k] for k=0..4
    
    # The tensor entries are all 1, so we need the bilinear form to evaluate to 1 
    # for every (i,j,k) triple.
    
    # Let me try a different approach: use known small-rank decompositions.
    # For 2x2x2, Strassen gives rank 7.
    # For 1x5x5 (vector-matrix), the rank is 5 (schoolbook is optimal).
    # 
    # We can partition the j-index (size 5) into groups and handle each group.
    # If we split j into {0,1}, {2,3}, {4}, we get three sub-problems:
    # - 2x2x2 (j=0,1; k=0,1): rank 7 (Strassen)
    # - 2x2x2 (j=0,1; k=2,3): rank 7 (Strassen)  
    # - 2x1x1 (j=0; k=4): rank 2 (schoolbook, since n=2,p=1)
    # - 2x2x2 (j=2,3; k=0,1): rank 7 (Strassen)
    # - 2x2x2 (j=2,3; k=2,3): rank 7 (Strassen)
    # - 2x1x1 (j=2; k=4): rank 2
    # - 2x2x1 (j=4; k=0,1): rank 4 (schoolbook: n*m*p = 2*2*1=4)
    # - 2x2x1 (j=4; k=2,3): rank 4
    # - 2x1x1 (j=4; k=4): rank 2
    # Total: 7+7+2+7+7+2+4+4+2 = 42
    
    # Wait, that's better than 46! Let me double-check.
    # The partition of j: {0,1}, {2,3}, {4}
    # The partition of k: {0,1}, {2,3}, {4}
    # For each (J, K) block:
    # (j in {0,1}, k in {0,1}): 2x2x2 -> Strassen rank 7
    # (j in {0,1}, k in {2,3}): 2x2x2 -> Strassen rank 7
    # (j in {0,1}, k in {4}): 2x1x1 -> schoolbook rank 2
    # (j in {2,3}, k in {0,1}): 2x2x2 -> Strassen rank 7
    # (j in {2,3}, k in {2,3}): 2x2x2 -> Strassen rank 7
    # (j in {2,3}, k in {4}): 2x1x1 -> schoolbook rank 2
    # (j in {4}, k in {0,1}): 2x1x2 -> schoolbook rank 2*1*2=4
    # (j in {4}, k in {2,3}): 2x1x2 -> schoolbook rank 2*1*2=4
    # (j in {4}, k in {4}): 2x1x1 -> schoolbook rank 2
    # Total: 7+7+2+7+7+2+4+4+2 = 42
    
    # That's rank 42, better than 46! Let me implement this.
    
    terms = []
    
    def add_strassen_block(i_off, j_off, k_off, n_block, m_block, p_block):
        """Add Strassen decomposition for n_block x m_block x p_block = 2x2x2."""
        # Strassen 7 terms for 2x2x2 tensor decomposition.
        # We need: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k in {0,1}
        # U indexed by (i,j), V by (j,k), W by (k,i)
        #
        # M1 = (A00+A11)(B00+B11): C00+=M1, C11+=M1
        #   U: (0,0)=1, (1,1)=1
        #   V: (0,0)=1, (1,1)=1
        #   W: (0,0)=1, (1,1)=1  [W[k,i]: k=0,i=0 and k=1,i=1]
        #   Check (0,0,0): 1*1*1=1, (1,1,1): 1*1*1=1, others 0. Good.
        #
        # M2 = (A10+A11)B00: C10+=M2, C11-=M2
        #   U: (1,0)=1, (1,1)=1
        #   V: (0,0)=1
        #   W: (0,1)=1, (1,1)=-1  [W[k,i]: k=0,i=1 and k=1,i=1]
        #   Check (1,0,0): U[1,0]*V[0,0]*W[0,1]=1*1*1=1
        #   Check (1,1,0): U[1,1]*V[0,0]*W[0,1]=1*1*1=1
        #   Check (1,0,1): U[1,0]*V[0,1]*W[1,1]=1*0*(-1)=0
        #   Check (1,1,1): U[1,1]*V[0,1]*W[1,1]=1*0*(-1)=0
        #   So M2 gives +1 to (1,0,0) and (1,1,0). Good.
        #
        # M3 = A00(B10-B00): C00-=M3, C01+=M3
        #   U: (0,0)=1
        #   V: (0,0)=-1, (1,0)=1
        #   W: (0,0)=-1, (1,0)=1  [W[k,i]: k=0,i=0 and k=1,i=0]
        #   Check (0,0,0): U[0,0]*V[0,0]*W[0,0]=1*(-1)*(-1)=1
        #   Check (0,1,0): U[0,1]*V[1,0]*W[0,0]=0*1*(-1)=0
        #   Check (0,0,1): U[0,0]*V[0,1]*W[1,0]=1*0*1=0
        #   Check (0,1,1): U[0,1]*V[1,1]*W[1,0]=0*0*1=0
        #   So M3 gives +1 to (0,0,0). Good (it's -M3 in C00, so contributes -1 there,
        #   but in tensor form it's +1 to the (0,0,0) entry since the tensor is all 1s).
        #   Wait, I need to reconsider. The tensor T[i,j,k]=1 for all valid (i,j,k).
        #   The decomposition sums to 1. So each term's contribution to (i,j,k) adds up to 1.
        #   M3 contributes +1 to (0,0,0). That's fine as long as the total sums to 1.
        #
        # M4 = A11(B01-B10): C00+=M4, C10+=M4
        #   U: (1,1)=1
        #   V: (0,1)=1, (1,0)=-1
        #   W: (0,0)=1, (0,1)=1  [W[k,i]: k=0,i=0 and k=0,i=1]
        #   Check (1,1,1): U[1,1]*V[1,1]*W[1,1]=1*0*0=0
        #   Check (1,0,1): U[1,0]*V[0,1]*W[1,1]=0*1*0=0
        #   Check (1,1,0): U[1,1]*V[1,0]*W[0,1]=1*(-1)*1=-1
        #   Check (1,0,0): U[1,0]*V[0,0]*W[0,1]=0*(-1)*1=0
        #   Hmm, M4 gives -1 to (1,1,0). Let me reconsider.
        #
        # Actually, let me just verify numerically. I'll compute the sum of all 7 terms
        # for each (i,j,k) and check it equals 1.

        strassen_terms = [
            # M1: U=(00,11), V=(00,11), W=(00,11)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2: U=(10,11), V=(00), W=(01,11) with W[0,1]=1, W[1,1]=-1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
            # M3: U=(00), V=(00,10) with V[0,0]=-1,V[1,0]=1, W=(00,10) with W[0,0]=-1,W[1,0]=1
            ({(0,0): 1}, {(0,0): -1, (1,0): 1}, {(0,0): -1, (1,0): 1}),
            # M4: U=(11), V=(01,10) with V[0,1]=1,V[1,0]=-1, W=(00,01) with W[0,0]=1,W[0,1]=1
            ({(1,1): 1}, {(0,1): 1, (1,0): -1}, {(0,0): 1, (0,1): 1}),
            # M5: U=(00,01), V=(11), W=(10,11) with W[1,0]=1,W[1,1]=1
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(1,0): 1, (1,1): 1}),
            # M6: U=(10,00) with U[1,0]=1,U[0,0]=-1, V=(00,01), W=(01,10) with W[0,1]=1,W[1,0]=-1
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,1): 1, (1,0): -1}),
            # M7: U=(01,11) with U[0,1]=1,U[1,1]=-1, V=(10,11), W=(00,11) with W[0,0]=1,W[1,1]=-1
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
        ]
        
        # Verify this Strassen decomposition locally before adding
        local_U = np.zeros((4, 7))  # (i*2+j, term)
        local_V = np.zeros((4, 7))  # (j*2+k, term)
        local_W = np.zeros((4, 7))  # (k*2+i, term)
        for r, (u_d, v_d, w_d) in enumerate(strassen_terms):
            for (di, dj), val in u_d.items():
                local_U[di*2+dj, r] = val
            for (dj, dk), val in v_d.items():
                local_V[dj*2+dk, r] = val
            for (dk, di), val in w_d.items():
                local_W[dk*2+di, r] = val
        local_ok = True
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    s = sum(local_U[di*2+dj, r] * local_V[dj*2+dk, r] * local_W[dk*2+di, r] for r in range(7))
                    if abs(s - 1) > 1e-9:
                        local_ok = False
                        break
                if not local_ok:
                    break
            if not local_ok:
                break
        
        if not local_ok:
            # Fallback: use schoolbook for this 2x2x2 block (rank 8)
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        u_abs = {(i_off + di, j_off + dj): 1}
                        v_abs = {(j_off + dj, k_off + dk): 1}
                        w_abs = {(k_off + dk, i_off + di): 1}
                        terms.append((u_abs, v_abs, w_abs))
            return
        
        for u_dict, v_dict, w_dict in strassen_terms:
            u_abs = {}
            for (di, dj), val in u_dict.items():
                u_abs[(i_off + di, j_off + dj)] = val
            v_abs = {}
            for (dj, dk), val in v_dict.items():
                v_abs[(j_off + dj, k_off + dk)] = val
            w_abs = {}
            for (dk, di), val in w_dict.items():
                w_abs[(k_off + dk, i_off + di)] = val
            terms.append((u_abs, v_abs, w_abs))
    
    def add_schoolbook_block(i_off, j_off, k_off, n_block, m_block, p_block):
        """Add schoolbook decomposition for n_block x m_block x p_block."""
        for di in range(n_block):
            for dj in range(m_block):
                for dk in range(p_block):
                    u = {(i_off + di, j_off + dj): 1}
                    v = {(j_off + dj, k_off + dk): 1}
                    w = {(k_off + dk, i_off + di): 1}
                    terms.append((u, v, w))
    
    # Partition: j = {0,1}, {2,3}, {4}; k = {0,1}, {2,3}, {4}
    j_groups = [(0, 2), (2, 4), (4, 5)]
    k_groups = [(0, 2), (2, 4), (4, 5)]
    
    for (j_start, j_end) in j_groups:
        for (k_start, k_end) in k_groups:
            m_block = j_end - j_start
            p_block = k_end - k_start
            n_block = n  # always 2
            
            if m_block == 2 and p_block == 2:
                # Use Strassen
                add_strassen_block(0, j_start, k_start, n_block, m_block, p_block)
            else:
                # Use schoolbook
                add_schoolbook_block(0, j_start, k_start, n_block, m_block, p_block)
    
    rank = len(terms)
    
    # Verify the decomposition
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        u_dict, v_dict, w_dict = terms[r]
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    # Verify: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    max_error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                error = abs(val - 1)
                if error > max_error:
                    max_error = error
    
    if max_error > 1e-9:
        # Fallback to the known-good rank 46 decomposition
        terms = []
        
        def add_strassen(i_start, j_start, k_start):
            u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
            v = {(j_start, k_start): 1}
            w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start): 1}
            v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
            w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start+1): 1}
            v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
            w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
            v = {(j_start+1, k_start+1): 1}
            w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
            terms.append((u, v, w))
            
            u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
            v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
            w = {(k_start+1, i_start+1): 1}
            terms.append((u, v, w))
            
            u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
            v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
            w = {(k_start, i_start): 1}
            terms.append((u, v, w))
        
        def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        u = {(i, j): 1}
                        v = {(j, k): 1}
                        w = {(k, i): 1}
                        terms.append((u, v, w))
        
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_schoolbook(0, 2, 0, 2, 4, 5)
        
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)
        add_schoolbook(0, 2, 2, 4, 4, 5)
        
        add_schoolbook(0, 2, 4, 5, 0, 2)
        add_schoolbook(0, 2, 4, 5, 2, 4)
        add_schoolbook(0, 2, 4, 5, 4, 5)
        
        rank = len(terms)
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for c in range(rank):
            u_dict, v_dict, w_dict = terms[c]
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = val
        
        return {"U": U, "V": V, "W": W}
    
    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Convert numpy floats to int where possible
    def clean_list(lst):
        result = []
        for row in lst:
            cleaned_row = []
            for val in row:
                if isinstance(val, (int, np.integer)):
                    cleaned_row.append(int(val))
                else:
                    fv = float(val)
                    if abs(fv - round(fv)) < 1e-9:
                        cleaned_row.append(int(round(fv)))
                    elif abs(fv*2 - round(fv*2)) < 1e-9:
                        cleaned_row.append(fv)
                    else:
                        cleaned_row.append(fv)
            result.append(cleaned_row)
        return result
    
    return {"U": clean_list(U_list), "V": clean_list(V_list), "W": clean_list(W_list)}
