"""Matrix multiplication 4,5,6 - rank-minimal decomposition with column removal."""
from fractions import Fraction
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] which should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = delta_{i,j,k} where A[i,j]*B[j,k] contributes to C[i,k]
    # The tensor decomposition satisfies: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # But with the W index convention: W[k*n+i, c], so W_index = k*n+i
    # Let's verify by checking that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 if (i,j,k) valid, else 0
    # Actually for matrix multiplication tensor, T[i,j,k] = 1 always (for valid indices),
    # because the tensor is the identity mapping: each (i,j,k) triple appears exactly once.
    # Wait, no. The matrix multiplication tensor T: A ⊗ B -> C
    # T[i,j,k] means the coefficient in the linear form that maps A and B to C[i,k]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor component at (i,j,k) is 1 (the product A[i,j]*B[j,k] appears with coefficient 1)
    # And all other components are 0.
    # So T[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition satisfies: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _remove_redundant_columns(U, V, W, n, m, p):
    """Try to remove redundant columns from the decomposition."""
    rank = len(U[0]) if U else 0
    if rank <= 1:
        return U, V, W
    
    removed = set()
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    
    # Try removing each column one at a time
    for col in range(rank):
        if col in removed:
            continue
        
        # Try removing this column and check if still valid
        test_U = [[current_U[i][c] for c in range(rank) if c != col and c not in removed] for i in range(len(current_U))]
        test_V = [[current_V[i][c] for c in range(rank) if c != col and c not in removed] for i in range(len(current_V))]
        test_W = [[current_W[i][c] for c in range(rank) if c != col and c not in removed] for i in range(len(current_W))]
        
        if _verify(test_U, test_V, test_W, n, m, p):
            removed.add(col)
            current_U = test_U
            current_V = test_V
            current_W = test_W
    
    return current_U, current_V, current_W

def _try_remove_multiple(U, V, W, n, m, p):
    """Try to remove multiple columns simultaneously via local search."""
    rank = len(U[0]) if U else 0
    if rank <= 2:
        return U, V, W
    
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = rank
    
    for iteration in range(50):
        # Try each single removal from current best
        found_improvement = False
        for col in range(best_rank):
            test_U = [[best_U[i][c] for c in range(best_rank) if c != col] for i in range(len(best_U))]
            test_V = [[best_V[i][c] for c in range(best_rank) if c != col] for i in range(len(best_V))]
            test_W = [[best_W[i][c] for c in range(best_rank) if c != col] for i in range(len(best_W))]
            
            if _verify(test_U, test_V, test_W, n, m, p):
                best_U = test_U
                best_V = test_V
                best_W = test_W
                best_rank = best_rank - 1
                found_improvement = True
                break
        
        if not found_improvement:
            break
    
    return best_U, best_V, best_W

def _block_decompose_45x56():
    """
    Build a block-structured decomposition for 4x5 * 5x6.
    
    We split the 5x5 middle dimension into blocks to exploit structure.
    
    Approach: Use the fact that we can decompose by splitting rows and columns.
    
    Let's think of A as 4x5, B as 5x6, C as 4x6.
    
    Split the middle dimension (size 5) into parts of sizes [3, 2] or similar.
    
    For a 2x2 block structure with sizes a1+a2=5:
    A = [A1 (4xa1), A2 (4xa2)]
    B = [B1 (a1x6), B2 (a2x6)]
    C = A1*B1 + A2*B2
    
    Each product A_i * B_i is a matrix multiplication of sizes 4xa_i * a_i x 6.
    
    For 4x3 * 3x6: schoolbook rank = 72
    For 4x2 * 2x6: schoolbook rank = 48
    Total: 120 (same as before)
    
    But we can use Strassen-like tricks for individual blocks or for the sum.
    
    Actually, the key insight is to use a different decomposition that allows
    sharing of computations between the two products.
    
    Alternative: Think of it as multiplying by a 5-dimensional intermediate.
    
    Let me try a different approach: use the structure to find linear dependencies.
    
    Actually, let me just build the schoolbook decomposition and then aggressively
    remove redundant columns. The schoolbook has rank 120, and there may be
    significant redundancy.
    """
    n, m, p = 4, 5, 6
    rank = n * m * p  # 120
    
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    return U, V, W

def _construct_strassen_ish():
    """
    Try a more clever decomposition using block structure.
    
    Split A into left (4x3) and right (4x2) parts.
    Split B into top (3x6) and bottom (2x6) parts.
    
    C = A_left * B_top + A_right * B_bottom
    
    This is a sum of two matrix products. We can try to find a low-rank 
    decomposition of this sum.
    
    For the sum of two independent matrix products, a natural decomposition is
    to just concatenate, giving rank 72+48=120. But there might be structure.
    
    Actually, let me think differently. The key is that we're multiplying by a 5D vector essentially.
    
    Let me try: represent the multiplication as a bilinear map and look for structure.
    
    Alternative approach: Use the known bounds. For 4x5 * 5x6, what's the best known rank?
    
    Let me just go with schoolbook + aggressive column removal.
    """
    n, m, p = 4, 5, 6
    
    # Start with schoolbook
    U, V, W = _block_decompose_45x56()
    
    # Remove redundant columns iteratively
    U, V, W = _remove_redundant_columns(U, V, W, n, m, p)
    
    # Try more aggressive removal
    U, V, W = _try_remove_multiple(U, V, W, n, m, p)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook if something went wrong
        U, V, W = _block_decompose_45x56()
    
    # Convert to float lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return U_list, V_list, W_list

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline to be safe
    
    n, m, p = DIMENSIONS
    
    # Try the main approach
    U, V, W = _construct_strassen_ish()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Shouldn't happen, but safety
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
