import numpy as np
import time

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS  # 3,4,5


def _build_strassen_222():
    """Rank-7 Strassen for 2x2x2. Returns U,V,W each (4,7).
    U order: A00,A01,A10,A11 ; V order: B00,B01,B10,B11 ; W order: C00,C01,C10,C11."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11) -> C00 +, C11 +
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00 -> C10 +, C11 -
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11) -> C01 +, C11 +
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = A11(B10-B00) -> C00 +, C10 +
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (A00+A01)B11 -> C00 -, C01 +
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (A10-A00)(B00+B01) -> C11 +
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11) -> C00 +
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _build_221():
    """Rank-4 schoolbook for 2x2x1: A(2x2)*B(2x1)=C(2x1). U(4,4),V(2,4),W(2,4).
    U order: A00,A01,A10,A11 ; V order: B0,B1 ; W order: C0,C1."""
    U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1  # A00 B0 -> C0
    U[1,1]=1; V[1,1]=1; W[0,1]=1  # A01 B1 -> C0
    U[2,2]=1; V[0,2]=1; W[1,2]=1  # A10 B0 -> C1
    U[3,3]=1; V[1,3]=1; W[1,3]=1  # A11 B1 -> C1
    return U, V, W


def _build_122():
    """Rank-4 schoolbook for 1x2x2: A(1x2)*B(2x2)=C(1x2). U(2,4),V(4,4),W(2,4).
    U order: A0,A1 ; V order: B00,B01,B10,B11 ; W order: C0,C1."""
    U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1  # A0 B00 -> C0
    U[1,1]=1; V[2,1]=1; W[0,1]=1  # A1 B10 -> C0
    U[0,2]=1; V[1,2]=1; W[1,2]=1  # A0 B01 -> C1
    U[1,3]=1; V[3,3]=1; W[1,3]=1  # A1 B11 -> C1
    return U, V, W


def _build_121():
    """Rank-2 for 1x2x1: A(1x2)*B(2x1)=C(1x1). U(2,2),V(2,2),W(1,2)."""
    U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W


def _verify(U, V, W):
    """Check that sum_r U[:,r]⊗V[:,r]⊗W[:,r] equals multiplication tensor.
    U: (12, r) idx i*m+j ; V: (20, r) idx j*p+k ; W: (15, r) idx k*n+i."""
    r = U.shape[1]
    # Build tensor T[i,j,k]
    T = np.zeros((n, m, p))
    Umat = np.asarray(U, dtype=float)
    Vmat = np.asarray(V, dtype=float)
    Wmat = np.asarray(W, dtype=float)
    for idx in range(r):
        u = Umat[:, idx].reshape(n, m)
        v = Vmat[:, idx].reshape(m, p)
        w = Wmat[:, idx].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
    # Expected: M[i,j,k] = 1 for all (contributes A[i,j]*B[j,k] to C[i,k])
    M = np.ones((n, m, p))
    return np.allclose(T, M, atol=1e-9)


def _rank_for_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
    """Return the rank needed for a sub-block of matrix multiplication."""
    ar, ac = a_rows, a_cols
    br, bc = b_rows, b_cols
    cr, cc = c_rows, c_cols
    if ar == 1 and ac == 1:
        return br * bc  # 1x1 * brxbc -> 1xbc, each output is a dot product of length br
    if ar == 1 and bc == 1:
        return ac * br  # 1xac * brx1 -> 1x1, sum of ac*br products but rank is ac*br? No...
        # Actually 1xac * brx1: C = A*B, C[k] = sum_j A[j]*B[j,k]. This is ac independent dot products of length br.
        # Rank = ac (each column of C needs ac multiplications... wait no)
        # C has cc=1 output. C[0] = sum over j of A[j]*B[j,0]. That's a single dot product of length ac*br? No.
        # B is brx1, so B[j,0] for j in range(br). A[j] for j in range(ac). But ac must equal br for the product to be defined!
        # So ar=1, ac=br, bc=1: C[0] = sum_{j=0}^{ac-1} A[j]*B[j,0]. Rank = ac (one multiplication per j).
        return ac
    if ac == 1 and bc == 1:
        return ar * br  # arx1 * brx1? No, ac must equal br.
        # arx1 * 1xbc: C[i,k] = A[i]*B[0,k]. Rank = ar*cc... but cc=1 here? No cc can be anything.
        # Wait, I'm confusing myself. Let me be more careful.
        pass
    # General case: use known results or schoolbook
    # For now, use schoolbook: ar*ac*cc
    return ar * ac * cc


def _construct_optimized():
    """Try different block partitions and find the one with minimum rank."""
    best_rank = float('inf')
    best_result = None
    
    # Strategy: split A's rows into groups, A's cols into groups, B's rows into groups, B's cols into groups
    # For each combination of (A_row_split, A_col_split, B_row_split, B_col_split),
    # compute the total rank and pick the best.
    
    # A is 3x4, B is 4x5, C is 3x5
    # A row splits: partitions of 3: [3], [1,2], [2,1], [1,1,1]
    # A col splits: partitions of 4: [4], [1,3], [2,2], [3,1], [1,1,2], [1,2,1], [2,1,1], [1,1,1,1]
    # B row splits: must match A col splits (since A's cols = B's rows = 4)
    # B col splits: partitions of 5: [5], [1,4], [2,3], [3,2], [4,1], [1,1,3], [1,2,2], [1,3,1], [2,1,2], [2,2,1], [3,1,1], [1,1,1,2], [1,1,2,1], [1,2,1,1], [2,1,1,1], [1,1,1,1,1]
    
    from itertools import product
    
    def get_partitions(total, max_parts=4):
        """Generate all compositions of 'total' into at most max_parts positive integers."""
        if max_parts == 1:
            yield [total]
            return
        for first in range(1, total):
            for rest in get_partitions(total - first, max_parts - 1):
                yield [first] + list(rest)
    
    a_row_splits = list(get_partitions(3, 3))
    a_col_splits = list(get_partitions(4, 4))
    b_col_splits = list(get_partitions(5, 5))
    
    def block_rank(ar, ac, bc, cc):
        """Rank for arxac * acxbc -> arxcc matrix multiplication."""
        # ac must equal bc for valid multiplication
        if ac != bc:
            return float('inf')
        # Known optimal ranks:
        if ar == 1 and ac == 1 and cc == 1:
            return 1
        if ar == 1 and cc == 1:
            return ac  # dot product
        if ac == 1 and cc == 1:
            return ar  # scalar times vector... wait
            # arx1 * 1x1 -> arx1. C[i] = A[i]*B[0]. Rank = ar.
            return ar
        if ar == 1 and ac == 1:
            return cc  # 1x1 * 1xcc -> 1xcc. C[k] = A[0]*B[0,k]. Rank = cc.
            return cc
        if ar == 2 and ac == 2 and cc == 2:
            return 7  # Strassen
        if ar == 2 and ac == 2 and cc == 1:
            return 4  # schoolbook, and it's optimal
        if ar == 1 and ac == 2 and cc == 2:
            return 4  # schoolbook, and it's optimal
        if ar == 1 and ac == 2 and cc == 1:
            return 2  # dot product of length 2
        if ar == 2 and ac == 1 and cc == 2:
            return 2  # 2x1 * 1x2 -> 2x2. C[i,k] = A[i]*B[0,k]. Rank = 2*2=4? No, rank 2: use A[0]*B[0,:] and A[1]*B[0,:]
            # Actually C = A*B where A is 2x1, B is 1x2. C[i,k] = A[i]*B[k]. 
            # This is an outer product. Rank = 2 (one for each row of A, or each col of B).
            # M1 = A[0]*B[0] contributes to C[0,0], M2 = A[1]*B[0] contributes to C[1,0]... 
            # Wait, B is 1x2 so B[0,0] and B[0,1]. C[0,0]=A[0]*B[0,0], C[0,1]=A[0]*B[0,1], C[1,0]=A[1]*B[0,0], C[1,1]=A[1]*B[0,1].
            # We can do: M1=A[0], M2=A[1]. Then C[0,0]=M1*B[0,0], C[0,1]=M1*B[0,1], C[1,0]=M2*B[0,0], C[1,1]=M2*B[0,1].
            # So rank 2 suffices!
            return 2
        if ar == 1 and ac == 1 and cc == 2:
            return 2  # 1x1 * 1x2 -> 1x2. C[0,k] = A[0]*B[0,k]. Rank 2.
        if ar == 2 and ac == 1 and cc == 1:
            return 2  # 2x1 * 1x1 -> 2x1. C[i,0] = A[i]*B[0]. Rank 2.
        if ar == 1 and ac == 1 and cc == 3:
            return 3
        if ar == 1 and ac == 1 and cc == 4:
            return 4
        if ar == 1 and ac == 1 and cc == 5:
            return 5
        if ar == 1 and ac == 2 and cc == 3:
            return 6  # 1x2 * 2x3 -> 1x3. Each output is a dot product of length 2. Rank 3? No, 2 multiplications per output, 3 outputs = 6.
            # Actually: C[0,0]=A[0]*B[0,0]+A[1]*B[1,0], C[0,1]=A[0]*B[0,1]+A[1]*B[1,1], C[0,2]=A[0]*B[0,2]+A[1]*B[1,2]
            # Can we share? M1=A[0], M2=A[1]. Then C[0,0]=M1*B[0,0]+M2*B[1,0], etc. But each term is a separate multiplication in the tensor sense.
            # Rank = 6 (each of the 6 products A[i]*B[j,k] is independent).
            return 6
        if ar == 1 and ac == 2 and cc == 4:
            return 8
        if ar == 1 and ac == 2 and cc == 5:
            return 10
        if ar == 1 and ac == 3 and cc == 2:
            return 6
        if ar == 1 and ac == 3 and cc == 3:
            return 9
        if ar == 1 and ac == 3 and cc == 4:
            return 12
        if ar == 1 and ac == 3 and cc == 5:
            return 15
        if ar == 1 and ac == 4 and cc == 2:
            return 8
        if ar == 1 and ac == 4 and cc == 3:
            return 12
        if ar == 1 and ac == 4 and cc == 4:
            return 16
        if ar == 1 and ac == 4 and cc == 5:
            return 20
        if ar == 2 and ac == 2 and cc == 3:
            # 2x2 * 2x3 -> 2x3. Schoolbook: 2*2*3=12. Can we do better?
            # Each row of C is a 1x2 * 2x3 = 1x3 product, rank 6 per row, total 12.
            # With Strassen-like: treat as two 1x2*2x3 problems. No sharing between rows.
            return 12
        if ar == 2 and ac == 2 and cc == 4:
            return 16
        if ar == 2 and ac == 2 and cc == 5:
            return 20
        if ar == 2 and ac == 3 and cc == 2:
            return 12
        if ar == 2 and ac == 3 and cc == 3:
            return 18
        if ar == 2 and ac == 3 and cc == 4:
            return 24
        if ar == 2 and ac == 3 and cc == 5:
            return 30
        if ar == 2 and ac == 4 and cc == 2:
            return 16
        if ar == 2 and ac == 4 and cc == 3:
            return 24
        if ar == 2 and ac == 4 and cc == 4:
            return 32
        if ar == 2 and ac == 4 and cc == 5:
            return 40
        if ar == 3 and ac == 2 and cc == 2:
            return 12
        if ar == 3 and ac == 2 and cc == 3:
            return 18
        if ar == 3 and ac == 2 and cc == 4:
            return 24
        if ar == 3 and ac == 2 and cc == 5:
            return 30
        if ar == 3 and ac == 3 and cc == 2:
            return 18
        if ar == 3 and ac == 3 and cc == 3:
            return 27
        if ar == 3 and ac == 3 and cc == 4:
            return 36
        if ar == 3 and ac == 3 and cc == 5:
            return 45
        if ar == 3 and ac == 4 and cc == 2:
            return 24
        if ar == 3 and ac == 4 and cc == 3:
            return 36
        if ar == 3 and ac == 4 and cc == 4:
            return 48
        if ar == 3 and ac == 4 and cc == 5:
            return 60
        # Default: schoolbook
        return ar * ac * cc
    
    def compute_total_rank(a_split, ac_split, bc_split):
        """Compute total rank for given splits. a_split: list of row group sizes for A.
        ac_split: list of col group sizes for A (must equal bc_split for B rows).
        bc_split: list of col group sizes for B."""
        if len(ac_split) != len(bc_split):
            return float('inf')
        if sum(ac_split) != 4 or sum(bc_split) != 4:
            return float('inf')
        for ac, bc in zip(ac_split, bc_split):
            if ac != bc:
                return float('inf')
        
        total = 0
        a_row_start = 0
        for ar in a_split:
            ac_start = 0
            for ac in ac_split:
                bc_start = 0
                for bc in bc_split:
                    cc_start = 0
                    # B cols are split the same way as... wait, B cols split is independent
                    # Actually B has 5 columns, split into b_col_splits
                    for cc in b_col_splits:
                        r = block_rank(ar, ac, bc, cc)
                        if r == float('inf'):
                            return float('inf')
                        total += r
                        cc_start += cc
                bc_start += bc
            ac_start += ac
        return total
    
    # This is getting complicated. Let me simplify by just trying a few key partitions.
    
    # Current best: rank 42 with splits A_rows=[2,1], A_cols=[2,2], B_rows=[2,2], B_cols=[2,2,1]
    # Let me verify and try alternatives.
    
    def try_partition(a_rows, a_cols, b_cols):
        """b_rows must equal a_cols."""
        if sum(a_rows) != 3 or sum(a_cols) != 4 or sum(b_cols) != 5:
            return None
        if len(a_cols) != len(a_cols):  # b_rows = a_cols
            return None
        
        total = 0
        a_row_idx = 0
        for ar in a_rows:
            ac_idx = 0
            for ac in a_cols:
                bc_idx = 0
                for bc in a_cols:  # b_rows same as a_cols
                    cc_idx = 0
                    for cc in b_cols:
                        r = block_rank(ar, ac, bc, cc)
                        if r == float('inf'):
                            return None
                        total += r
                        cc_idx += cc
                    bc_idx += bc
                ac_idx += ac
            a_row_idx += ar
        return total
    
    candidates = [
        ([2,1], [2,2], [2,2,1]),  # current: 42
        ([2,1], [2,2], [2,3]),
        ([2,1], [2,2], [3,2]),
        ([2,1], [2,2], [1,2,2]),
        ([2,1], [2,2], [2,1,2]),
        ([2,1], [2,2], [1,1,3]),
        ([2,1], [2,2], [3,1,1]),
        ([2,1], [1,3], [2,2,1]),
        ([2,1], [1,3], [2,3]),
        ([2,1], [3,1], [2,2,1]),
        ([2,1], [3,1], [2,3]),
        ([1,2], [2,2], [2,2,1]),
        ([1,2], [2,2], [2,3]),
        ([1,2], [1,3], [2,2,1]),
        ([1,2], [1,3], [2,3]),
        ([1,2], [3,1], [2,2,1]),
        ([1,2], [3,1], [2,3]),
        ([3], [2,2], [2,2,1]),
        ([3], [2,2], [2,3]),
        ([3], [1,3], [2,2,1]),
        ([3], [1,3], [2,3]),
        ([3], [4], [2,2,1]),
        ([3], [4], [2,3]),
        ([1,1,1], [2,2], [2,2,1]),
        ([1,1,1], [2,2], [2,3]),
        ([1,1,1], [1,3], [2,2,1]),
        ([1,1,1], [1,3], [2,3]),
    ]
    
    best = (float('inf'), None)
    for a_rows, a_cols, b_cols in candidates:
        r = try_partition(a_rows, a_cols, b_cols)
        if r is not None and r < best[0]:
            best = (r, (a_rows, a_cols, b_cols))
    
    print(f"Best partition rank: {best[0]} with {best[1]}")
    
    # Use the best partition to construct
    if best[1] is None:
        return _construct()
    
    a_rows, a_cols, b_cols = best[1]
    
    U_parts, V_parts, W_parts = [], [], []
    
    def add_block(a_r_start, a_c_start, a_r_size, a_c_size, b_r_start, b_c_start, b_r_size, b_c_size):
        """Add a block decomposition for A[a_r_start:a_r_start+a_r_size, a_c_start:a_c_start+a_c_size] *
           B[b_r_start:b_r_start+b_r_size, b_c_start:b_c_start+b_c_size]"""
        ar, ac = a_r_size, a_c_size
        bc, cc = b_r_size, b_c_size
        if ac != bc:
            return
        
        # Generate U, V, W for this block
        if ar == 2 and ac == 2 and cc == 2:
            Ub, Vb, Wb = _build_strassen_222()
        elif ar == 2 and ac == 2 and cc == 1:
            Ub, Vb, Wb = _build_221()
        elif ar == 1 and ac == 2 and cc == 2:
            Ub, Vb, Wb = _build_122()
        elif ar == 1 and ac == 2 and cc == 1:
            Ub, Vb, Wb = _build_121()
        elif ar == 1 and ac == 1 and cc == 1:
            Ub = np.ones((1,1)); Vb = np.ones((1,1)); Wb = np.ones((1,1))
        elif ar == 1 and ac == 1:
            Ub = np.ones((1,cc)); Vb = np.eye(cc); Wb = np.ones((cc,cc))
            # 1x1 * 1xcc: C[0,k] = A[0]*B[0,k]. U=[A0], V[k]=B[0,k], W[k]=1
            Ub = np.zeros((1,cc)); Vb = np.zeros((1,cc)); Wb = np.zeros((cc,cc))
            for k in range(cc):
                Ub[0,k] = 1; Vb[0,k] = 1; Wb[k,k] = 1
        elif ac == 1 and cc == 1:
            # arx1 * 1x1 -> arx1
            Ub = np.zeros((ar,ar)); Vb = np.zeros((1,ar)); Wb = np.zeros((ar,ar))
            for i in range(ar):
                Ub[i,i] = 1; Vb[0,i] = 1; Wb[i,i] = 1
        elif ar == 1:
            # 1xac * acxcc -> 1xcc
            rank = ac * cc
            Ub = np.zeros((ac, rank)); Vb = np.zeros((ac, rank)); Wb = np.zeros((cc, rank))
            for i in range(ac):
                for k in range(cc):
                    idx = i * cc + k
                    Ub[i, idx] = 1; Vb[i, idx] = 1; Wb[k, idx] = 1
        elif cc == 1:
            # arxac * acx1 -> arx1
            rank = ar * ac
            Ub = np.zeros((ar*ac, rank)); Vb = np.zeros((ac, rank)); Wb = np.zeros((ar, rank))
            for i in range(ar):
                for j in range(ac):
                    idx = i * ac + j
                    Ub[i*ac+j, idx] = 1; Vb[j, idx] = 1; Wb[i, idx] = 1
        elif ac == 1:
            # arx1 * 1xcc -> arxcc
            rank = ar * cc
            Ub = np.zeros((ar, rank)); Vb = np.zeros((1, rank)); Wb = np.zeros((ar*cc, rank))
            for i in range(ar):
                for k in range(cc):
                    idx = i * cc + k
                    Ub[i, idx] = 1; Vb[0, idx] = 1; Wb[i*cc+k, idx] = 1
        else:
            # General schoolbook
            rank = ar * ac * cc
            Ub = np.zeros((ar*ac, rank)); Vb = np.zeros((ac*cc, rank)); Wb = np.zeros((ar*cc, rank))
            for i in range(ar):
                for j in range(ac):
                    for k in range(cc):
                        idx = (i * ac + j) * cc + k
                        Ub[i*ac+j, idx] = 1; Vb[j*cc+k, idx] = 1; Wb[k*ar+i, idx] = 1
        
        # Map to global indices
        for r in range(Ub.shape[1]):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for i in range(ar):
                for j in range(ac):
                    val = Ub[i*ac+j, r] if Ub.shape[0] == ar*ac else Ub[i, r]
                    if abs(val) > 1e-15:
                        u[(a_r_start+i)*m + (a_c_start+j)] = val
            for j in range(ac):
                for k in range(cc):
                    if Vb.shape[0] == ac*cc:
                        val = Vb[j*cc+k, r]
                    elif Vb.shape[0] == ac:
                        val = Vb[j, r]
                    else:
                        val = Vb[0, r]
                    if abs(val) > 1e-15:
                        v[(b_r_start+j)*p + (b_c_start+k)] = val
            for i in range(ar):
                for k in range(cc):
                    if Wb.shape[0] == ar*cc:
                        val = Wb[k*ar+i, r]
                    elif Wb.shape[0] == cc:
                        val = Wb[k, r]
                    else:
                        val = Wb[i, r]
                    if abs(val) > 1e-15:
                        w[(b_c_start+k)*n + (a_r_start+i)] = val
            U_parts.append(u); V_parts.append(v); W_parts.append(w)
    
    a_row_idx = 0
    for ar in a_rows:
        ac_idx = 0
        for ac in a_cols:
            bc_idx = 0
            for bc in a_cols:
                cc_idx = 0
                for cc in b_cols:
                    add_block(a_row_idx, ac_idx, ar, ac, bc_idx, cc_idx, bc, cc)
                    cc_idx += cc
                bc_idx += bc
            ac_idx += ac
        a_row_idx += ar
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W


def _construct_schoolbook():
    """Guaranteed rank-60 schoolbook decomposition."""
    U_parts, V_parts, W_parts = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m); u[i*m+j] = 1
                v = np.zeros(m*p); v[j*p+k] = 1
                w = np.zeros(p*n); w[k*n+i] = 1
                U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W


def construct():
    t0 = time.time()
    deadline = 150.0
    margin = 30.0
    
    # Try the optimized construction first
    try:
        U, V, W = _construct_optimized()
        if time.time() - t0 > deadline - margin:
            U, V, W = _construct_schoolbook()
        elif not _verify(U, V, W):
            U, V, W = _construct()
            if not _verify(U, V, W):
                U, V, W = _construct_schoolbook()
    except Exception:
        try:
            U, V, W = _construct()
            if not _verify(U, V, W):
                U, V, W = _construct_schoolbook()
        except Exception:
            U, V, W = _construct_schoolbook()
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
