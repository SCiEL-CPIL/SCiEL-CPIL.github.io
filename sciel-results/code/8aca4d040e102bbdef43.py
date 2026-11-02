"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin before 150s kill
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    def is_valid_entry(x):
        """Check if x is integer or half-integer."""
        if x is None:
            return False
        if isinstance(x, (int, float)):
            return abs(x * 2 - round(x * 2)) < 1e-9
        return False
    
    def verify_validity(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if not is_valid_entry(val):
                        return False
        return True
    
    # Try slice-and-dice construction
    # Partition rows into groups, columns into groups
    # For each (row_group, col_group, all_p_cols) sub-tensor, use a decomposition
    
    def build_subtensor_decomp(rows, cols, p_cols, rank_budget):
        """Build a decomposition for the sub-tensor with given row/col sets.
        Returns (U_sub, V_sub, W_sub, rank_used) where shapes are:
        U_sub: (len(rows)*m, rank_used) but only nonzero for rows in 'rows'
        Actually, let's just build full-size matrices with zeros elsewhere.
        """
        r = len(rows)
        c = len(cols)
        pc = len(p_cols)
        # Schoolbook for this sub-tensor: r*c*pc
        sb_rank = r * c * pc
        if sb_rank <= rank_budget:
            # Use schoolbook
            U_sub = [[0] * rank_budget for _ in range(n * m)]
            V_sub = [[0] * rank_budget for _ in range(m * p)]
            W_sub = [[0] * rank_budget for _ in range(p * n)]
            col = 0
            for i in rows:
                for j in cols:
                    for k in p_cols:
                        U_sub[i * m + j][col] = 1
                        V_sub[j * p + k][col] = 1
                        W_sub[k * n + i][col] = 1
                        col += 1
            return U_sub, V_sub, W_sub, col
        else:
            return None, None, None, 0
    
    # Strategy: try different partitions
    # Partition 1: rows={0,1,2}, cols={0,1,2,3}, p={0..6} -> full schoolbook = 84
    # Partition 2: split rows into {0} and {1,2}, cols into {0,1} and {2,3}
    #   Sub-tensors: 
    #     rows={0}, cols={0,1}, p=all: rank=1*2*7=14
    #     rows={0}, cols={2,3}, p=all: rank=1*2*7=14
    #     rows={1,2}, cols={0,1}, p=all: rank=2*2*7=28
    #     rows={1,2}, cols={2,3}, p=all: rank=2*2*7=28
    #   Total: 14+14+28+28=84 (same as schoolbook)
    
    # Try to find better decompositions for the 2x2x7 sub-tensors
    # For 2x2x7, schoolbook is 28. Can we do better?
    # Lower bound: max(2*2, 2*7, 2*7) = max(4,14,14) = 14
    # Strassen for 2x2x2 has rank 7. For 2x2x7, we can apply Strassen per column:
    #   For each of the 7 columns of B, use Strassen (rank 7) -> total rank 49 (worse)
    # Alternatively, think of it as 7 copies of 2x2x1, each rank 4 -> total 28 (same)
    
    # Let's try a different approach: use a structured search with random initialization
    # and local optimization to find a lower-rank decomposition
    
    # Fallback: return schoolbook
    U, V, W, rank = schoolbook()
    best_result = {"U": U, "V": W, "W": W}  # Will fix below
    
    # Try to improve: search for rank < 84
    # Use a simple approach: try rank values from lower bound upward
    # For each rank, try to find a valid decomposition
    
    # Given time constraints, let's try a few structured approaches
    
    # Approach: Use the fact that for small tensors, we can try to find
    # a decomposition with integer/half-integer entries
    
    # For now, return schoolbook as it's guaranteed valid
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
