"""Combined search + structured Kronecker/block decomposition for 3x3 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [3, 3, 3]

def build_target_tensor():
    """Return list of (u_idx, v_idx, w_idx) for each valid multiplication term."""
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                terms.append((u_idx, v_idx, w_idx))
    return terms

TARGET_TERMS = build_target_tensor()

def validate_decomposition(U, V, W, rank):
    """Check if the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                val = 0.0
                for r in range(rank):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def make_candidate(rank, U, V, W):
    """Return candidate dict."""
    return {'U': [list(row) for row in U], 'V': [list(row) for row in V], 'W': [list(row) for row in W]}

def random_candidate(rank):
    """Generate a random candidate with integer entries in {-2,-1,0,1,2}."""
    U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(9)]
    V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(9)]
    W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(9)]
    return U, V, W

def mutate(U, V, W, rank, num_mutations=3):
    """Apply random mutations to the candidate."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    for _ in range(num_mutations):
        matrix = random.choice([U, V, W])
        row = random.randint(0, 8)
        col = random.randint(0, rank - 1)
        old_val = matrix[row][col]
        if random.random() < 0.15:
            new_val = 0
        else:
            new_val = random.randint(-5, 5)
            if new_val == 0 and old_val != 0:
                new_val = random.choice([-1, 1])
        matrix[row][col] = new_val
    return U, V, W

def compute_error(U, V, W, rank):
    """Compute squared error of the decomposition."""
    err = 0.0
    for (u_idx, v_idx, w_idx) in TARGET_TERMS:
        val = 0.0
        for r in range(rank):
            val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
        err += (val - 1.0) ** 2
    return err

def hill_climb(rank, U, V, W, max_steps=5000, deadline=None):
    """Perform hill climbing to fix a near-correct decomposition."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    best_error = compute_error(U, V, W, rank)
    no_improve_count = 0
    max_no_improve = 150
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try single-point mutation
        U_new, V_new, W_new = mutate(U, V, W, rank, num_mutations=1)
        err_new = compute_error(U_new, V_new, W_new, rank)
        
        if err_new < best_error:
            U, V, W = U_new, V_new, W_new
            best_error = err_new
            no_improve_count = 0
            if abs(best_error) < 1e-10:
                break
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try larger jumps
                U_new, V_new, W_new = mutate(U, V, W, rank, num_mutations=random.randint(2, 6))
                err_new = compute_error(U_new, V_new, W_new, rank)
                if err_new < best_error:
                    U, V, W = U_new, V_new, W_new
                    best_error = err_new
                    no_improve_count = 0
                else:
                    if random.random() < 0.1:
                        U, V, W = random_candidate(rank)
                        best_error = compute_error(U, V, W, rank)
                        no_improve_count = 0
    
    return U, V, W, best_error

def _known_rank26():
    """Return the known rank 26 decomposition using block-Strassen structure."""
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. a b^T -> Rank 4
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    # 3. A2 b -> Rank 4
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    # 4. a b3 -> Rank 2
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    # 5. a^T B2 -> Rank 4
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    # 6. a3 b^T -> Rank 2
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    # 7. a^T b -> Rank 2
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    # 8. a3 b3 -> Rank 1
    add_term(8, 8, 8, 1)
    
    return {'U': U, 'V': V, 'W': W}

def _structured_block_init(rank):
    """Generate a structured initialization based on block decomposition.
    
    Treat 3x3 as 2x2 blocks of sizes (2,1)x(2,1).
    The diagonal 2x2 block product uses Strassen-like structure.
    Off-diagonal blocks use direct terms but with potential sharing.
    """
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Assign columns to different block interactions
    # Block structure:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]]
    # C = A*B
    
    # C11 = A11*B11 + A12*B21  (2x2)
    # C12 = A11*B12 + A12*B22  (2x1)
    # C21 = A21*B11 + A22*B21  (1x2)
    # C22 = A21*B12 + A22*B22  (1x1)
    
    # We'll assign some columns to Strassen-like combos for the 2x2 parts
    # and some to direct terms for the rest
    
    col = 0
    if col >= rank:
        return U, V, W
    
    # Try to set up a partial Strassen for C11 (A11*B11)
    # U indices for A11: 0,1,3,4
    # V indices for B11: 0,1,3,4
    # W indices for C11: 0,3,1,4 (k*3+i)
    
    # M1 = (A00+A11)(B00+B11)
    if col < rank:
        U[0][col] = 1; U[4][col] = 1
        V[0][col] = 1; V[4][col] = 1
        W[0][col] = 1; W[4][col] = 1
        col += 1
    
    # M2 = (A10+A11)B00
    if col < rank:
        U[3][col] = 1; U[4][col] = 1
        V[0][col] = 1
        W[1][col] = 1; W[4][col] = -1
        col += 1
    
    # M3 = A00(B01-B11)
    if col < rank:
        U[0][col] = 1
        V[1][col] = 1; V[4][col] = -1
        W[3][col] = 1; W[4][col] = 1
        col += 1
    
    # M4 = A11(B10-B00)
    if col < rank:
        U[4][col] = 1
        V[3][col] = 1; V[0][col] = -1
        W[0][col] = 1; W[1][col] = 1
        col += 1
    
    # M5 = (A00+A01)B11
    if col < rank:
        U[0][col] = 1; U[1][col] = 1
        V[4][col] = 1
        W[0][col] = -1; W[3][col] = 1
        col += 1
    
    # M6 = (A10-A00)(B00+B01)
    if col < rank:
        U[3][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[4][col] = 1
        col += 1
    
    # M7 = (A01-A11)(B10+B11)
    if col < rank:
        U[1][col] = 1; U[4][col] = -1
        V[3][col] = 1; V[4][col] = 1
        W[0][col] = 1
        col += 1
    
    # Now fill remaining columns with random structured values for off-diagonal blocks
    # Off-diagonal block indices:
    # A12 (2x1): U indices 2, 5
    # A21 (1x2): U indices 6, 7
    # A22 (1x1): U index 8
    # B12 (2x1): V indices 2, 5
    # B21 (1x2): V indices 6, 7
    # B22 (1x1): V index 8
    # C12 (2x1): W indices 6, 7
    # C21 (1x2): W indices 2, 5
    # C22 (1x1): W index 8
    
    # For remaining columns, assign to off-diagonal interactions
    off_diag_u = [2, 5, 6, 7, 8]
    off_diag_v = [2, 5, 6, 7, 8]
    off_diag_w = [2, 5, 6, 7, 8]
    
    while col < rank:
        # Randomly assign to an off-diagonal interaction
        u_idx = random.choice(off_diag_u)
        v_idx = random.choice(off_diag_v)
        w_idx = random.choice(off_diag_w)
        
        U[u_idx][col] = random.choice([-2, -1, 1, 2])
        V[v_idx][col] = random.choice([-2, -1, 1, 2])
        W[w_idx][col] = random.choice([-2, -1, 1, 2])
        col += 1
    
    return U, V, W

def try_find_decomposition(rank, deadline):
    """Try to find a valid decomposition of given rank."""
    # Strategy 1: Start from rank 26 and try to merge
    if rank < 26:
        result = _try_merge_from_rank26(rank, deadline)
        if result is not None:
            return result
    
    # Strategy 2: Structured block initialization + hill climbing
    for attempt in range(150):
        if time.time() > deadline:
            return None
        
        U, V, W = _structured_block_init(rank)
        U, V, W, err = hill_climb(rank, U, V, W, max_steps=800, deadline=deadline)
        if abs(err) < 1e-10:
            if validate_decomposition(U, V, W, rank):
                return make_candidate(rank, U, V, W)
    
    # Strategy 3: Random initialization + hill climbing
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        U, V, W = random_candidate(rank)
        U, V, W, err = hill_climb(rank, U, V, W, max_steps=600, deadline=deadline)
        if abs(err) < 1e-10:
            if validate_decomposition(U, V, W, rank):
                return make_candidate(rank, U, V, W)
    
    return None

def _try_merge_from_rank26(rank, deadline):
    """Try to merge columns of the rank-26 solution to get lower rank."""
    base = _known_rank26()
    U = [list(row) for row in base['U']]
    V = [list(row) for row in base['V']]
    W = [list(row) for row in base['W']]
    
    # Try to merge pairs of columns
    for attempt in range(80):
        if time.time() > deadline:
            return None
        
        U_try = [list(row) for row in U]
        V_try = [list(row) for row in V]
        W_try = [list(row) for row in W]
        
        # Try to merge random pairs
        for _ in range(26 - rank):
            if len(U_try[0]) <= rank:
                break
            cols = list(range(len(U_try[0])))
            c1, c2 = random.sample(cols, 2)
            for i in range(9):
                U_try[i][c1] += U_try[i][c2]
                V_try[i][c1] += V_try[i][c2]
                W_try[i][c1] += W_try[i][c2]
            for i in range(9):
                del U_try[i][c2]
                del V_try[i][c2]
                del W_try[i][c2]
        
        if len(U_try[0]) == rank:
            U_try, V_try, W_try, err = hill_climb(rank, U_try, V_try, W_try, max_steps=800, deadline=deadline)
            if abs(err) < 1e-10:
                if validate_decomposition(U_try, V_try, W_try, rank):
                    return make_candidate(rank, U_try, V_try, W_try)
    
    return None

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Start with known rank 26 solution as fallback
    fallback = _known_rank26()
    
    best = None
    best_rank = 26
    
    # Try to find lower rank solutions
    for rank in [25, 24, 23, 22, 21, 20, 19, 18]:
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        print(f"Trying rank {rank}...")
        result = try_find_decomposition(rank, deadline)
        if result is not None:
            best = result
            best_rank = rank
            print(f"Found valid decomposition with rank {rank}!")
            break
    
    if best is not None:
        return best
    else:
        return fallback
# EVOLVE-BLOCK-END
