import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    """
    Constructs the tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    
    # For 3x5x7 matrix multiplication, we'll use a direct construction
    # The tensor T_{i,a,b,k} = 1 if a==b else 0
    
    # We'll try to find a rank decomposition using the following strategy:
    # 1. Start with schoolbook (rank = n*m*p = 105)
    # 2. Try to reduce rank by finding linear dependencies
    
    # Actually, let me think about this more carefully.
    # The matrix multiplication tensor for A(n x m) * B(m x p) = C(n x p)
    # has entries T[i, j, k, l] where i in [0,n), j in [0,m), k in [0,m), l in [0,p)
    # T[i,j,k,l] = 1 if j==k else 0
    
    # This tensor can be decomposed as sum over c of U[i,j,c] * V[k,l,c] * W[i,l,c]
    # where the condition is sum_c U[i,j,c]*V[k,l,c]*W[i,l,c] = delta_{j,k}
    
    # For schoolbook: each term c corresponds to (i,j,k,l) with j==k
    # U[i,j,c] = 1 if c corresponds to (i,j,j,i) else 0
    # V[k,l,c] = 1 if c corresponds to (k,l) with j=k else 0  
    # W[i,l,c] = 1 if c corresponds to (i,l) else 0
    
    # Wait, let me re-examine the convention.
    # The task says: W[k*n+i, r]
    # So W is indexed by (k, i) where k is column index of C, i is row index of C
    # C has shape (n, p), so row index i in [0,n), col index k in [0,p)
    # W[k*n+i, r] means W[row=k*n+i, col=r] where row = k*n + i
    
    # Constraint: for all i in [0,n), a in [0,m), b in [0,m), k in [0,p):
    #   sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c] = (1 if a==b else 0)
    
    # So:
    # U has shape (n*m, rank), indexed by [i*m+a, c]
    # V has shape (m*p, rank), indexed by [b*p+k, c]  
    # W has shape (p*n, rank), indexed by [k*n+i, c]
    
    # The equation is: sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c] = delta_{a,b}
    
    # This is the standard matrix multiplication tensor decomposition.
    
    # For 3x5x7, schoolbook rank = 3*5*7 = 105
    # Can we do better?
    
    # Let me try a block approach:
    # Split the 5x5 middle dimension into blocks
    # If we can find a 5x5x5 block with rank < 125, we can use it
    
    # Actually, for square matrix multiplication n x n x n:
    # - n=1: rank 1
    # - n=2: rank 7 (Strassen)
    # - n=3: rank 23 (or better)
    # - n=5: unknown but likely < 125
    
    # For rectangular:
    # 3x5x7: we can try various block decompositions
    
    # Let me use a simple approach: enumerate all possible 2x2x2 Strassen blocks
    # and combine with schoolbook for the rest
    
    # First, let me try to construct the schoolbook decomposition as a baseline
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    # Term for (i, a, a, k)
                    U[i * m + a, c] = 1
                    V[a * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank
    
    # Now let me try to use Strassen for 2x2x2 subblocks
    # Strassen for 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the tensor decomposition, each Strassen term contributes to multiple entries of C
    
    def strassen_2x2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen.
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        Indexing: U[i*2+a, c], V[b*2+k, c], W[k*2+i, c]
        where i in [0,2), a in [0,2), b in [0,2), k in [0,2)
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (A00+A11)(B00+B11)
        # A00: i=0,a=0; A11: i=1,a=1
        # B00: b=0,k=0; B11: b=1,k=1
        # Contributes to C00, C11 with +1
        U[0*2+0, 0] += 1  # A00
        U[1*2+1, 0] += 1  # A11
        V[0*2+0, 0] += 1  # B00
        V[1*2+1, 0] += 1  # B11
        W[0*2+0, 0] += 1  # C00 (k=0, i=0)
        W[1*2+1, 0] += 1  # C11 (k=1, i=1)
        
        # M2 = (A10+A11)B00
        # A10: i=1,a=0; A11: i=1,a=1
        # B00: b=0,k=0
        # Contributes to C10 with +1, C11 with -1
        U[1*2+0, 1] += 1  # A10
        U[1*2+1, 1] += 1  # A11
        V[0*2+0, 1] += 1  # B00
        W[0*2+1, 1] += 1  # C10 (k=0, i=1)
        W[1*2+1, 1] += -1 # C11 (k=1, i=1)
        
        # M3 = A00(B01-B11)
        # A00: i=0,a=0
        # B01: b=0,k=1; B11: b=1,k=1
        # Contributes to C01 with +1, C11 with +1
        U[0*2+0, 2] += 1  # A00
        V[0*2+1, 2] += 1  # B01
        V[1*2+1, 2] += -1 # B11
        W[1*2+0, 2] += 1  # C01 (k=1, i=0)
        W[1*2+1, 2] += 1  # C11 (k=1, i=1)
        
        # M4 = A11(B10-B00)
        # A11: i=1,a=1
        # B10: b=1,k=0; B00: b=0,k=0
        # Contributes to C00 with +1, C10 with +1
        U[1*2+1, 3] += 1  # A11
        V[1*2+0, 3] += 1  # B10
        V[0*2+0, 3] += -1 # B00
        W[0*2+0, 3] += 1  # C00 (k=0, i=0)
        W[0*2+1, 3] += 1  # C10 (k=0, i=1)
        
        # M5 = (A00+A01)B11
        # A00: i=0,a=0; A01: i=0,a=1
        # B11: b=1,k=1
        # Contributes to C00 with -1, C01 with +1
        U[0*2+0, 4] += 1  # A00
        U[0*2+1, 4] += 1  # A01
        V[1*2+1, 4] += 1  # B11
        W[0*2+0, 4] += -1 # C00 (k=0, i=0)
        W[1*2+0, 4] += 1  # C01 (k=1, i=0)
        
        # M6 = (A10-A00)(B00+B01)
        # A10: i=1,a=0; A00: i=0,a=0
        # B00: b=0,k=0; B01: b=0,k=1
        # Contributes to C11 with +1
        U[1*2+0, 5] += 1  # A10
        U[0*2+0, 5] += -1 # A00
        V[0*2+0, 5] += 1  # B00
        V[0*2+1, 5] += 1  # B01
        W[1*2+1, 5] += 1  # C11 (k=1, i=1)
        
        # M7 = (A01-A11)(B10+B11)
        # A01: i=0,a=1; A11: i=1,a=1
        # B10: b=1,k=0; B11: b=1,k=1
        # Contributes to C00 with +1
        U[0*2+1, 6] += 1  # A01
        U[1*2+1, 6] += -1 # A11
        V[1*2+0, 6] += 1  # B10
        V[1*2+1, 6] += 1  # B11
        W[0*2+0, 6] += 1  # C00 (k=0, i=0)
        
        return U.tolist(), V.tolist(), W.tolist(), rank
    
    # Now let's build a block decomposition
    # We'll try different ways to partition the dimensions
    
    def all_partitions(size, max_block=2):
        """Generate all partitions of 'size' into blocks of size 1 or 2."""
        if size == 0:
            return [[]]
        results = []
        for rest in all_partitions(size - 1, max_block):
            results.append([1] + rest)
        if size >= 2:
            for rest in all_partitions(size - 2, max_block):
                results.append([2] + rest)
        return results
    
    def compute_rank(n_blocks, m_blocks, p_blocks):
        """Compute the total rank for a given block partition."""
        rank = 0
        for ns in n_blocks:
            for ms in m_blocks:
                for ps in p_blocks:
                    if ns == 2 and ms == 2 and ps == 2:
                        rank += 7
                    else:
                        rank += ns * ms * ps
        return rank
    
    def build_block_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
        """
        Build the decomposition using the given block partitions.
        """
        # Compute offsets
        n_offsets = [0]
        for s in n_blocks:
            n_offsets.append(n_offsets[-1] + s)
        m_offsets = [0]
        for s in m_blocks:
            m_offsets.append(m_offsets[-1] + s)
        p_offsets = [0]
        for s in p_blocks:
            p_offsets.append(p_offsets[-1] + s)
        
        # Compute total rank
        total_rank = compute_rank(n_blocks, m_blocks, p_blocks)
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # Get Strassen components
        sU, sV, sW, _ = strassen_2x2x2()
        sU = np.array(sU)
        sV = np.array(sV)
        sW = np.array(sW)
        
        col = 0
        for ni in range(len(n_blocks)):
            for mi in range(len(m_blocks)):
                for pi in range(len(p_blocks)):
                    n0 = n_offsets[ni]
                    m0 = m_offsets[mi]
                    p0 = p_offsets[pi]
                    ns = n_blocks[ni]
                    ms = m_blocks[mi]
                    ps = p_blocks[pi]
                    
                    if ns == 2 and ms == 2 and ps == 2:
                        # Use Strassen
                        for c in range(7):
                            for i in range(2):
                                for a in range(2):
                                    U[(n0 + i) * m + (m0 + a), col] += sU[i*2+a, c]
                            for b in range(2):
                                for k in range(2):
                                    V[(m0 + b) * p + (p0 + k), col] += sV[b*2+k, c]
                            for i in range(2):
                                for k in range(2):
                                    W[(p0 + k) * n + (n0 + i), col] += sW[k*2+i, c]
                            col += 1
                    else:
                        # Use schoolbook
                        for i in range(ns):
                            for a in range(ms):
                                for k in range(ps):
                                    if a == k:  # Wait, this is wrong
                                        pass
                        # Actually for schoolbook, each term is (i, a, a, k) but only if a is in the m-block
                        # Let me fix this
                        for i in range(ns):
                            for a in range(ms):
                                for k in range(ps):
                                    # This term contributes to entry (i, k) of C
                                    # with coefficient U[i,a]*V[a,k]*W[i,k]
                                    # For schoolbook, we need sum over a of A[i,a]*B[a,k]
                                    # So each term is: U[i*m+a, c]=1, V[a*p+k, c]=1, W[k*n+i, c]=1
                                    # but only for the case where this term is needed
                                    # Actually, for a block of size (ns, ms, ps), we need
                                    # for each (i, a, k) in the block, a term
                                    # But the constraint is sum_c W[k*n+i, c] * U[i*m+a, c] * V[b*p+k, c] = delta_{a,b}
                                    # For schoolbook, we use one term per (i, a, k) with a as the middle index
                                    # U[i*m+a, c] = 1
                                    # V[a*p+k, c] = 1  (note: b must equal a)
                                    # W[k*n+i, c] = 1
                                    U[(n0 + i) * m + (m0 + a), col] = 1
                                    V[(m0 + a) * p + (p0 + k), col] = 1
                                    W[(p0 + k) * n + (n0 + i), col] = 1
                                    col += 1
        
        if col != total_rank:
            raise ValueError(f"Column mismatch: {col} != {total_rank}")
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, total_rank
    
    # Generate all partitions and find the best
    n_partitions = all_partitions(n)
    m_partitions = all_partitions(m)
    p_partitions = all_partitions(p)
    
    best_rank = n * m * p
    best_result = None
    
    for np_ in n_partitions:
        for mp_ in m_partitions:
            for pp_ in p_partitions:
                r = compute_rank(np_, mp_, pp_)
                if r < best_rank:
                    try:
                        result, actual_rank = build_block_decomposition(n, m, p, np_, mp_, pp_)
                        if actual_rank == r and result is not None:
                            best_rank = r
                            best_result = result
                    except Exception:
                        pass
    
    if best_result is not None:
        return best_result
    else:
        result, _ = schoolbook()
        return result
