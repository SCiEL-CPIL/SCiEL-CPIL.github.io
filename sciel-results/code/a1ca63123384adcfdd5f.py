"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication using Strassen.
    Each term is (U_entry, V_entry, W_entry) where:
    - U has shape (2*2, 1) = (4,1), indexed by i*2+j
    - V has shape (2*2, 1) = (4,1), indexed by j*2+k
    - W has shape (2*2, 1) = (4,1), indexed by k*2+i (trace convention: W[k*n+i])
    
    For 2x2x2: n=2, m=2, p=2
    U index: i*m+j = i*2+j for i in {0,1}, j in {0,1}
    V index: j*p+k = j*2+k for j in {0,1}, k in {0,1}
    W index: k*n+i = k*2+i for k in {0,1}, i in {0,1}
    """
    # Strassen's 7 products
    # A = [[A00, A01], [A10, A11]] where A00 = A[i=0,j=0], A01 = A[i=0,j=1], A10 = A[i=1,j=0], A11 = A[i=1,j=1]
    # B = [[B00, B01], [B10, B11]] where B00 = B[j=0,k=0], B01 = B[j=0,k=1], B10 = B[j=1,k=0], B11 = B[j=1,k=1]
    # C = AB, C[i,k]
    
    # U indices: i*2+j: U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
    # V indices: j*2+k: V[0]=B00, V[1]=B01, V[2]=B10, V[3]=B11
    # W indices: k*2+i: W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11
    
    terms = []
    
    # P1 = (A00 + A11)(B00 + B11)
    # U: A00=1, A11=1 -> U=[1,0,0,1]
    # V: B00=1, B11=1 -> V=[1,0,0,1]
    # W: contributes to C00 (+1), C11 (+1) -> W=[1,0,0,1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # P2 = (A10 + A11)B00
    # U: A10=1, A11=1 -> U=[0,0,1,1]
    # V: B00=1 -> V=[1,0,0,0]
    # W: contributes to C10 (+1) -> W=[0,0,1,0]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    
    # P3 = A00(B01 - B11)
    # U: A00=1 -> U=[1,0,0,0]
    # V: B01=1, B11=-1 -> V=[0,1,0,-1]
    # W: contributes to C01 (+1) -> W=[0,1,0,0]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    
    # P4 = A11(B10 - B00)
    # U: A11=1 -> U=[0,0,0,1]
    # V: B10=1, B00=-1 -> V=[-1,0,1,0]
    # W: contributes to C10 (+1), C00 (+1) -> wait, let me recheck
    # C00 = P1 + P4 - P5 + P7, C10 = P2 + P4
    # So P4 contributes to C00 (+1) and C10 (+1)
    # W: C00=1, C10=1 -> W=[1,0,1,0]
    terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    
    # P5 = (A00 + A01)B11
    # U: A00=1, A01=1 -> U=[1,1,0,0]
    # V: B11=1 -> V=[0,0,0,1]
    # W: C00 = ... - P5, C01 = P3 + P5
    # So P5 contributes to C00 (-1) and C01 (+1)
    # W: C00=-1, C01=1 -> W=[-1,1,0,0]
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    
    # P6 = (A10 - A00)(B00 + B01)
    # U: A10=1, A00=-1 -> U=[-1,0,1,0]
    # V: B00=1, B01=1 -> V=[1,1,0,0]
    # W: C11 = P1 - P2 + P3 + P6
    # So P6 contributes to C11 (+1)
    # W: C11=1 -> W=[0,0,0,1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # P7 = (A01 - A11)(B10 + B11)
    # U: A01=1, A11=-1 -> U=[0,1,0,-1]
    # V: B10=1, B11=1 -> V=[0,0,1,1]
    # W: C00 = P1 + P4 - P5 + P7
    # So P7 contributes to C00 (+1)
    # W: C00=1 -> W=[1,0,0,0]
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms

def _schoolbook_1x2x2_terms():
    """
    Return the 4 rank-1 terms for 1x2x2 matrix multiplication (schoolbook).
    n=1, m=2, p=2
    U index: i*2+j for i in {0}, j in {0,1} -> U[0]=A00, U[1]=A01
    V index: j*2+k for j in {0,1}, k in {0,1} -> V[0]=B00, V[1]=B01, V[2]=B10, V[3]=B11
    W index: k*1+i = k for k in {0,1}, i in {0} -> W[0]=C00, W[1]=C01
    """
    terms = []
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    
    # Term 1: A00*B00 -> C00
    terms.append(([1,0], [1,0,0,0], [1,0]))
    # Term 2: A01*B10 -> C00
    terms.append(([0,1], [0,0,1,0], [1,0]))
    # Term 3: A00*B01 -> C01
    terms.append(([1,0], [0,1,0,0], [0,1]))
    # Term 4: A01*B11 -> C01
    terms.append(([0,1], [0,0,0,1], [0,1]))
    
    return terms

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Decompose: n=1+2, m=2+2, p=2+2+2+2
    # i_blocks: [(0,1), (1,3)] -> sizes [1, 2]
    # j_blocks: [(0,2), (2,4)] -> sizes [2, 2]
    # k_blocks: [(0,2), (2,4), (4,6), (6,8)] -> sizes [2, 2, 2, 2]
    
    i_blocks = [(0, 1), (1, 3)]  # i ranges: [0,1), [1,3)
    j_blocks = [(0, 2), (2, 4)]  # j ranges: [0,2), [2,4)
    k_blocks = [(0, 2), (2, 4), (4, 6), (6, 8)]  # k ranges
    
    # Collect all terms
    all_U = []  # Each entry is a dict or list of (row, value) pairs
    all_V = []
    all_W = []
    
    for i_start, i_end in i_blocks:
        n_i = i_end - i_start
        for j_start, j_end in j_blocks:
            m_j = j_end - j_start
            for k_start, k_end in k_blocks:
                p_k = k_end - k_start
                
                # Check time
                if time.time() > deadline - 10:
                    break
                
                if n_i == 1 and m_j == 2 and p_k == 2:
                    # Use schoolbook 1x2x2
                    terms = _schoolbook_1x2x2_terms()
                    for U_term, V_term, W_term in terms:
                        # U: 2 entries (i=0, j=0,1) -> rows i_start + j_local
                        # V: 4 entries (j=0,1; k=0,1) -> rows (j_start + j_local)*p + (k_start + k_local)
                        # W: 2 entries (k=0,1; i=0) -> rows (k_start + k_local)*n + (i_start + i_local)
                        
                        U_entries = []
                        for idx, val in enumerate(U_term):
                            if val != 0:
                                # idx = i_local * m_j + j_local = 0*2 + j_local = j_local
                                j_local = idx
                                U_entries.append((i_start + j_local, val))
                        
                        V_entries = []
                        for idx, val in enumerate(V_term):
                            if val != 0:
                                # idx = j_local * p_k + k_local
                                j_local = idx // p_k
                                k_local = idx % p_k
                                row = (j_start + j_local) * p + (k_start + k_local)
                                V_entries.append((row, val))
                        
                        W_entries = []
                        for idx, val in enumerate(W_term):
                            if val != 0:
                                # idx = k_local * n_i + i_local = k_local * 1 + 0 = k_local
                                k_local = idx
                                i_local = 0
                                row = (k_start + k_local) * n + (i_start + i_local)
                                W_entries.append((row, val))
                        
                        all_U.append(U_entries)
                        all_V.append(V_entries)
                        all_W.append(W_entries)
                
                elif n_i == 2 and m_j == 2 and p_k == 2:
                    # Use Strassen 2x2x2
                    terms = _strassen_2x2x2_terms()
                    for U_term, V_term, W_term in terms:
                        # U: 4 entries (i=0,1; j=0,1) -> rows i_start + i_local*2 + j_local... 
                        # Actually U index in local: i_local * m_j + j_local = i_local * 2 + j_local
                        # Global row: (i_start + i_local) * m + (j_start + j_local)
                        U_entries = []
                        for idx, val in enumerate(U_term):
                            if val != 0:
                                i_local = idx // m_j
                                j_local = idx % m_j
                                row = (i_start + i_local) * m + (j_start + j_local)
                                U_entries.append((row, val))
                        
                        V_entries = []
                        for idx, val in enumerate(V_term):
                            if val != 0:
                                j_local = idx // p_k
                                k_local = idx % p_k
                                row = (j_start + j_local) * p + (k_start + k_local)
                                V_entries.append((row, val))
                        
                        W_entries = []
                        for idx, val in enumerate(W_term):
                            if val != 0:
                                k_local = idx // n_i
                                i_local = idx % n_i
                                row = (k_start + k_local) * n + (i_start + i_local)
                                W_entries.append((row, val))
                        
                        all_U.append(U_entries)
                        all_V.append(V_entries)
                        all_W.append(W_entries)
    
    rank = len(all_U)
    
    # Build sparse matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
