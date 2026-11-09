import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    def verify(U, V, W, rank):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Start with schoolbook decomposition
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][column] = 1
                V[idx_v][column] = 1
                W[idx_w][column] = 1
                column += 1

    # Verify the schoolbook decomposition
    if not verify(U, V, W, rank):
        raise ValueError("Schoolbook decomposition failed verification")

    # Try to reduce rank using a structured block decomposition approach
    # Split the 5-dimension into blocks: [2, 2, 1]
    # This gives us a block structure where we can apply Strassen to 2x2 blocks
    
    # For 2x5x5 matrix multiplication:
    # A is 2x5, B is 5x5, C is 2x5
    # Split A into columns: A = [A1 (2x2), A2 (2x2), A3 (2x1)]
    # Split B into rows and columns:
    #   B = [[B11 (2x2), B12 (2x2), B13 (2x1)],
    #        [B21 (2x2), B22 (2x2), B23 (2x1)],
    #        [B31 (1x2), B32 (1x2), B33 (1x1)]]
    # Split C into columns: C = [C1 (2x2), C2 (2x2), C3 (2x1)]
    
    # Block multiplication:
    # C1 = A1*B11 + A2*B21 + A3*B31
    # C2 = A1*B12 + A2*B22 + A3*B32
    # C3 = A1*B13 + A2*B23 + A3*B33
    
    # For each block multiplication, we can use either schoolbook or Strassen
    # Let's try to construct a decomposition using Strassen for 2x2 blocks
    
    # Strassen's algorithm for 2x2 matrix multiplication uses 7 multiplications
    # For 2x2 * 2x2 -> 2x2, the 7 terms are:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's build the decomposition term by term
    # Each term in the decomposition corresponds to one "multiplication" in the algorithm
    
    # Initialize with empty lists
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        """Add a single term to the decomposition."""
        for idx_u, idx_v, idx_w, vu, vv, vw in zip(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
            if idx_u is not None and idx_v is not None and idx_w is not None:
                if vu != 0 and vv != 0 and vw != 0:
                    U_terms.append((idx_u, vu))
                    V_terms.append((idx_v, vv))
                    W_terms.append((idx_w, vw))
    
    # For each block multiplication, determine which terms to use
    # Block (i_block, j_block) where i_block is row block of A, j_block is col block of C
    
    # Let's define the block structure more carefully
    # A row blocks: [0,1] (rows 0-1), [2,3] (rows 2-3), [4] (row 4)
    # B row blocks: [0,1], [2,3], [4]
    # B col blocks: [0,1], [2,3], [4]
    # C col blocks: [0,1], [2,3], [4]
    
    # Block sizes:
    # A: 2x[2,2,1]
    # B: [2,2,1]x[2,2,1]
    # C: 2x[2,2,1]
    
    # For each pair of A-block and B-block that contribute to a C-block:
    # We need to multiply A_block (2xb_size) by B_block (b_size x c_size)
    
    # Let's just use schoolbook for simplicity and correctness first,
    # but try to see if we can find a lower rank through optimization
    
    # Actually, let's try a different approach: use a random initialization
    # and try to minimize the rank while satisfying the constraints
    
    # The constraint is:
    # For all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of equations. We can try to find a solution with
    # as few non-zero terms as possible.
    
    # Let's try a greedy approach: start with the schoolbook solution and
    # try to eliminate terms one by one
    
    # Convert to numpy for easier manipulation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to eliminate terms
    for c in range(rank - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Check if we can eliminate term c
        # A term can be eliminated if it's not needed for any (i,j,k)
        
        # Check if this term is the only one contributing to any (i,j,k)
        can_eliminate = True
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    
                    # Check if this term contributes to (i,j,k)
                    if U_np[idx_u, c] != 0 and V_np[idx_v, c] != 0 and W_np[idx_w, c] != 0:
                        # Check if there are other terms that also contribute
                        other_terms = 0
                        for c2 in range(rank):
                            if c2 == c:
                                continue
                            if U_np[idx_u, c2] != 0 and V_np[idx_v, c2] != 0 and W_np[idx_w, c2] != 0:
                                other_terms += 1
                        
                        if other_terms == 0:
                            can_eliminate = False
                            break
                if not can_eliminate:
                    break
            if not can_eliminate:
                break
        
        if can_eliminate:
            # Eliminate this term
            U_np[:, c] = 0
            V_np[:, c] = 0
            W_np[:, c] = 0
    
    # Check if the modified decomposition still works
    if verify(U_np, V_np, W_np, rank):
        # Find the actual rank (number of non-zero columns)
        actual_rank = 0
        for c in range(rank):
            if np.any(U_np[:, c] != 0) or np.any(V_np[:, c] != 0) or np.any(W_np[:, c] != 0):
                actual_rank += 1
        
        if actual_rank < rank:
            # Create new matrices with only the non-zero columns
            non_zero_cols = [c for c in range(rank) 
                           if np.any(U_np[:, c] != 0) or np.any(V_np[:, c] != 0) or np.any(W_np[:, c] != 0)]
            
            U_new = U_np[:, non_zero_cols]
            V_new = V_np[:, non_zero_cols]
            W_new = W_np[:, non_zero_cols]
            
            # Convert to lists
            U = U_new.tolist()
            V = V_new.tolist()
            W = W_new.tolist()
            rank = len(non_zero_cols)
    
    # Final verification
    if not verify(U, V, W, rank):
        # If verification fails, return the original schoolbook decomposition
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][column] = 1
                    V[idx_v][column] = 1
                    W[idx_w][column] = 1
                    column += 1
        rank = schoolbook_rank
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
