import time
import numpy as np
from itertools import combinations

N, M, P = 3, 3, 8
SCHOOLBOOK = N * M * P  # 72
A_DIM = N * M          # 9
B_DIM = M * P          # 24
C_DIM = P * N          # 24

# Precompute the target tensor T[a,b,c] = 1 iff (a,b,c) is a valid multiplication term
_T = np.zeros((A_DIM, B_DIM, C_DIM), dtype=np.float64)
for i in range(N):
    for j in range(M):
        for k in range(P):
            a = i * M + j
            b = j * P + k
            c = k * N + i
            _T[a, b, c] = 1.0


def _verify_fast(U, V, W, tol=1e-9):
    """Vectorized verification: check U,V,W decompose T exactly."""
    rank = U.shape[1]
    if rank == 0:
        return False
    T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(T_rec, _T, atol=tol, rtol=0)


def _build_schoolbook():
    """Return (U,V,W) for rank-72 schoolbook decomposition."""
    rank = SCHOOLBOOK
    U = np.zeros((A_DIM, rank), dtype=np.float64)
    V = np.zeros((B_DIM, rank), dtype=np.float64)
    W = np.zeros((C_DIM, rank), dtype=np.float64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    return U, V, W


def _add_strassen_block(U, V, W, col, a_rows, a_cols, b_rows, b_cols):
    """
    Add a 2x2x2 Strassen block at the given position.
    a_rows: 2 row indices in A (i indices)
    a_cols: 2 col indices in A (j indices)
    b_rows: 2 row indices in B (j indices) - must match a_cols
    b_cols: 2 col indices in B (k indices)
    
    Returns new col index.
    """
    # A indices: (i, j) -> i*M + j
    a00 = a_rows[0] * M + a_cols[0]
    a01 = a_rows[0] * M + a_cols[1]
    a10 = a_rows[1] * M + a_cols[0]
    a11 = a_rows[1] * M + a_cols[1]
    
    # B indices: (j, k) -> j*P + k
    b00 = b_rows[0] * P + b_cols[0]
    b01 = b_rows[0] * P + b_cols[1]
    b10 = b_rows[1] * P + b_cols[0]
    b11 = b_rows[1] * P + b_cols[1]
    
    # C indices: (k, i) -> k*N + i
    c00 = b_cols[0] * N + a_rows[0]
    c01 = b_cols[0] * N + a_rows[1]
    c10 = b_cols[1] * N + a_rows[0]
    c11 = b_cols[1] * N + a_rows[1]
    
    # M1: (A00+A11)(B00+B11) -> C00 +=, C11 +=, C01 -=, C10 -=
    U[a00, col] += 1; U[a11, col] += 1
    V[b00, col] += 1; V[b11, col] += 1
    W[c00, col] += 1; W[c11, col] += 1
    W[c01, col] -= 1; W[c10, col] -= 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10 +=, C11 -=
    U[a10, col] += 1; U[a11, col] += 1
    V[b00, col] += 1
    W[c10, col] += 1; W[c11, col] -= 1
    col += 1
    
    # M3: A00(B01-B11) -> C01 +=, C11 +=
    U[a00, col] += 1
    V[b01, col] += 1; V[b11, col] -= 1
    W[c01, col] += 1; W[c11, col] += 1
    col += 1
    
    # M4: A11(B10-B00) -> C00 +=, C10 +=
    U[a11, col] += 1
    V[b10, col] += 1; V[b00, col] -= 1
    W[c00, col] += 1; W[c10, col] += 1
    col += 1
    
    # M5: (A00+A01)B11 -> C00 -=, C01 +=
    U[a00, col] += 1; U[a01, col] += 1
    V[b11, col] += 1
    W[c00, col] -= 1; W[c01, col] += 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11 +=
    U[a10, col] += 1; U[a00, col] -= 1
    V[b00, col] += 1; V[b01, col] += 1
    W[c11, col] += 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C00 +=, C10 -=
    U[a01, col] += 1; U[a11, col] -= 1
    V[b10, col] += 1; V[b11, col] += 1
    W[c00, col] += 1; W[c10, col] -= 1
    col += 1
    
    return col


def _add_schoolbook_terms(U, V, W, col, covered_set):
    """
    Add schoolbook terms for all (i,j,k) not in covered_set.
    covered_set: set of (i,j,k) tuples that are already covered by Strassen blocks.
    Returns new col index.
    """
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if (i, j, k) not in covered_set:
                    a = i * M + j
                    b = j * P + k
                    c = k * N + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
    return col


def _get_covered_terms(a_rows, a_cols, b_cols):
    """
    Get the set of (i,j,k) terms covered by a Strassen block.
    A Strassen block covers all 8 terms: (i,j,k) for i in a_rows, j in a_cols, k in b_cols.
    """
    covered = set()
    for i in a_rows:
        for j in a_cols:
            for k in b_cols:
                covered.add((i, j, k))
    return covered


def _try_all_strassen_placements():
    """
    Systematically try all possible placements of 2x2x2 Strassen blocks.
    
    A Strassen block is defined by:
    - 2 rows of A (i indices): choose 2 from {0,1,2}
    - 2 cols of A (j indices): choose 2 from {0,1,2}
    - 2 cols of B (k indices): choose 2 from {0,...,7}
    
    The B rows are determined by A cols (j indices must match).
    
    We try:
    1. Single Strassen blocks (rank 65)
    2. Pairs of non-overlapping Strassen blocks (rank 58)
    3. Triples of non-overlapping Strassen blocks (rank 51)
    """
    results = []
    
    # All possible 2-row choices for A
    a_row_choices = list(combinations(range(N), 2))
    # All possible 2-col choices for A
    a_col_choices = list(combinations(range(M), 2))
    # All possible 2-col choices for B
    b_col_choices = list(combinations(range(P), 2))
    
    # Generate all possible Strassen blocks
    all_blocks = []
    for a_rows in a_row_choices:
        for a_cols in a_col_choices:
            for b_cols in b_col_choices:
                b_rows = a_cols  # B rows must match A cols
                covered = _get_covered_terms(a_rows, a_cols, b_cols)
                all_blocks.append({
                    'a_rows': a_rows,
                    'a_cols': a_cols,
                    'b_rows': b_rows,
                    'b_cols': b_cols,
                    'covered': covered
                })
    
    print(f"Total possible Strassen blocks: {len(all_blocks)}")
    
    # Try single blocks
    for block in all_blocks:
        rank = 72 - 8 + 7  # 71
        U = np.zeros((A_DIM, rank), dtype=np.float64)
        V = np.zeros((B_DIM, rank), dtype=np.float64)
        W = np.zeros((C_DIM, rank), dtype=np.float64)
        
        col = _add_strassen_block(U, V, W, 0, block['a_rows'], block['a_cols'], 
                                   block['b_rows'], block['b_cols'])
        col = _add_schoolbook_terms(U, V, W, col, block['covered'])
        
        if col != rank:
            U = U[:, :col]
            V = V[:, :col]
            W = W[:, :col]
            rank = col
        
        if _verify_fast(U, V, W):
            results.append((rank, U, V, W, [block]))
    
    # Try pairs of non-overlapping blocks
    for i in range(len(all_blocks)):
        for j in range(i + 1, len(all_blocks)):
            block1 = all_blocks[i]
            block2 = all_blocks[j]
            
            # Check non-overlap
            if block1['covered'] & block2['covered']:
                continue
            
            rank = 72 - 16 + 14  # 70
            U = np.zeros((A_DIM, rank), dtype=np.float64)
            V = np.zeros((B_DIM, rank), dtype=np.float64)
            W = np.zeros((C_DIM, rank), dtype=np.float64)
            
            col = _add_strassen_block(U, V, W, 0, block1['a_rows'], block1['a_cols'], 
                                       block1['b_rows'], block1['b_cols'])
            col = _add_strassen_block(U, V, W, col, block2['a_rows'], block2['a_cols'], 
                                       block2['b_rows'], block2['b_cols'])
            
            covered = block1['covered'] | block2['covered']
            col = _add_schoolbook_terms(U, V, W, col, covered)
            
            if col != rank:
                U = U[:, :col]
                V = V[:, :col]
                W = W[:, :col]
                rank = col
            
            if _verify_fast(U, V, W):
                results.append((rank, U, V, W, [block1, block2]))
    
    return results


def construct():
    start = time.time()
    deadline = start + 130.0  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _build_schoolbook()
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = U_school.shape[1]
    
    # Try systematic Strassen placement search
    if time.time() < deadline - 30:
        results = _try_all_strassen_placements()
        
        if results:
            # Sort by rank
            results.sort(key=lambda x: x[0])
            best_found_rank, best_U_found, best_V_found, best_W_found, _ = results[0]
            
            if best_found_rank < best_rank:
                best_U, best_V, best_W = best_U_found.copy(), best_V_found.copy(), best_W_found.copy()
                best_rank = best_found_rank
                print(f"Found better decomposition with rank {best_rank}")
    
    # Final verification
    if _verify_fast(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
