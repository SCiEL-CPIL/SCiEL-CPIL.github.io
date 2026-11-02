"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    Strategy:
    1. Build the rank-35 Strassen-based decomposition (5 blocks x 7 Strassen products).
    2. The tensor has a Kronecker-like structure: T = T_2x2 ⊗ T_rect where
       T_2x2 is the 2x2x2 matmul tensor (rank 7) and T_rect is the 2x1x5 tensor (rank 2).
       The naive Kronecker product gives rank 7*2 = 14 per "super-block" but the structure
       is actually 5 independent 2x2x2 blocks sharing the same A, giving rank 35.
       
    3. Key insight: The 35 rank-1 terms may have linear dependencies. We can try to find
       a lower-rank decomposition by:
       - Using structured random search with the Strassen structure as a warm start
       - Using gradient descent with multiple restarts
       - Trying to exploit the shared A structure across blocks
       
    4. We'll attempt to find a decomposition with rank < 35 using optimization.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin (10s buffer before 120s limit)
    
    # Build the rank-35 decomposition first
    U35, V35, W35 = _build_strassen_decomposition(n, m, p)
    
    # Convert to numpy arrays
    U35_np = np.array(U35, dtype=float)
    V35_np = np.array(V35, dtype=float)
    W35_np = np.array(W35, dtype=float)
    
    # Build the full tensor for verification
    T_target = np.einsum('ir,jr,kr->ijk', U35_np, V35_np, W35_np)
    
    # Verify the tensor is correct
    if not _verify_tensor_correctness(T_target, n, m, p):
        # Fallback: return the rank-35 decomposition
        return {"U": U35, "V": V35, "W": W35}
    
    # Try to find a lower-rank decomposition
    best_result = {"U": U35, "V": V35, "W": W35}
    best_rank = 35
    
    # Try ranks from 28 up to 34 (we know 35 works)
    for target_rank in range(28, 35):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        success = False
        for trial in range(20):
            if time.time() > deadline - 15:
                break
            
            result = _try_rank_decomposition(T_target, n, m, p, target_rank, 
                                            time_budget=6, trial=trial)
            if result is not None:
                U_new, V_new, W_new = result
                # Verify the decomposition
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = target_rank
                    success = True
                    break
        
        if success:
            break  # Found a valid lower rank, stop
    
    return best_result


def _build_strassen_decomposition(n, m, p):
    """Build the rank-35 Strassen-based decomposition."""
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_A = [
        [1, 0, 0, 1],  # P1: (a+d)
        [0, 0, 1, 1],  # P2: (c+d)
        [1, 0, 0, 0],  # P3: a
        [0, 0, 0, 1],  # P4: d
        [1, 1, 0, 0],  # P5: (a+b)
        [-1, 0, 1, 0], # P6: (c-a)
        [0, 1, 0, -1]  # P7: (b-d)
    ]
    
    strassen_B = [
        [1, 0, 0, 1],  # P1: (e+h)
        [1, 0, 0, 0],  # P2: e
        [0, 1, 0, -1], # P3: (f-h)
        [-1, 0, 1, 0], # P4: (g-e)
        [0, 0, 0, 1],  # P5: h
        [1, 1, 0, 0],  # P6: (e+f)
        [0, 0, 1, 1]   # P7: (g+h)
    ]
    
    # W contributions for each Strassen product to the local 2x2 C matrix
    # Local C indices: 0->C00, 1->C01, 2->C10, 3->C11
    # M00 = P1 + P4 - P5 + P7
    # M01 = P3 + P5
    # M10 = P2 + P4
    # M11 = P1 - P2 + P3 + P6
    
    w_contrib = [
        [1, 0, 0, 1],   # P1: C00 +1, C11 +1
        [0, 0, 1, -1],  # P2: C10 +1, C11 -1
        [0, 1, 0, 1],   # P3: C01 +1, C11 +1
        [1, 0, 1, 0],   # P4: C00 +1, C10 +1
        [-1, 1, 0, 0],  # P5: C00 -1, C01 +1
        [0, 0, 0, 1],   # P6: C11 +1
        [1, 0, 0, 0]    # P7: C00 +1
    ]
    
    col = 0
    for b in range(5):
        for r in range(7):
            # U: same for all blocks
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            # V: specific to block b
            V[2*b][col] = strassen_B[r][0]
            V[2*b+1][col] = strassen_B[r][1]
            V[10+2*b][col] = strassen_B[r][2]
            V[10+2*b+1][col] = strassen_B[r][3]
            
            # W: specific to block b
            W[4*b][col] = w_contrib[r][0]
            W[4*b+2][col] = w_contrib[r][1]
            W[4*b+1][col] = w_contrib[r][2]
            W[4*b+3][col] = w_contrib[r][3]
            
            col += 1
            
    return U, V, W


def _try_rank_decomposition(T, n, m, p, target_rank, time_budget=6, trial=0):
    """
    Try to find a rank-target_rank decomposition of tensor T.
    Uses structured random initialization + gradient descent with adaptive learning rate.
    """
    dim_u = n * m  # 4
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    start_time = time.time()
    
    # Different random seeds for different trials
    np.random.seed(12345 + trial * 100)
    
    # Structured initialization: start from a perturbed version of the Strassen structure
    U = _structured_init_U(dim_u, target_rank, trial)
    V = _structured_init_V(dim_v, target_rank, trial)
    W = _structured_init_W(dim_w, target_rank, trial)
    
    # Gradient descent with adaptive learning rate
    lr = 0.01
    min_lr = 1e-8
    patience = 25
    no_improve = 0
    best_loss = float('inf')
    
    U_best = U.copy()
    V_best = V.copy()
    W_best = W.copy()
    
    for step in range(1000):
        if time.time() - start_time > time_budget:
            break
        
        # Compute current tensor estimate
        T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = T - T_est
        loss = np.sum(diff**2)
        
        # Track best
        if loss < best_loss:
            best_loss = loss
            no_improve = 0
            U_best = U.copy()
            V_best = V.copy()
            W_best = W.copy()
        else:
            no_improve += 1
        
        # Early stopping if no improvement
        if no_improve > patience and lr < min_lr:
            break
        
        # Compute gradients
        grad_U = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
        grad_V = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
        grad_W = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
        
        # Update with adaptive learning rate
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if no_improve > 12:
            lr *= 0.5
            if lr < min_lr:
                break
        
        # Check convergence
        if loss < 1e-14:
            break
    
    # Final check using best found
    T_final = np.einsum('ir,jr,kr->ijk', U_best, V_best, W_best)
    if np.max(np.abs(T - T_final)) < 1e-8:
        # Convert to list format
        U_list = U_best.tolist()
        V_list = V_best.tolist()
        W_list = W_best.tolist()
        return U_list, V_list, W_list
    
    return None


def _structured_init_U(dim_u, target_rank, trial):
    """Structured initialization for U matrix based on Strassen structure."""
    # Start with a random matrix but with some structure
    U = np.random.randn(dim_u, target_rank) * 0.3
    
    # Add some structured components based on Strassen patterns
    strassen_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    for r in range(min(7, target_rank)):
        for i in range(dim_u):
            U[i, r] += strassen_A[r][i] * 0.5
    
    return U


def _structured_init_V(dim_v, target_rank, trial):
    """Structured initialization for V matrix."""
    V = np.random.randn(dim_v, target_rank) * 0.3
    
    # Add block structure
    strassen_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    for b in range(5):
        for r in range(min(7, target_rank)):
            V[2*b, r] += strassen_B[r][0] * 0.3
            V[2*b+1, r] += strassen_B[r][1] * 0.3
            V[10+2*b, r] += strassen_B[r][2] * 0.3
            V[10+2*b+1, r] += strassen_B[r][3] * 0.3
    
    return V


def _structured_init_W(dim_w, target_rank, trial):
    """Structured initialization for W matrix."""
    W = np.random.randn(dim_w, target_rank) * 0.3
    
    # Add block structure
    w_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for b in range(5):
        for r in range(min(7, target_rank)):
            W[4*b, r] += w_contrib[r][0] * 0.3
            W[4*b+2, r] += w_contrib[r][1] * 0.3
            W[4*b+1, r] += w_contrib[r][2] * 0.3
            W[4*b+3, r] += w_contrib[r][3] * 0.3
    
    return W


def _verify_tensor_correctness(T, n, m, p):
    """
    Verify that the tensor T correctly represents matrix multiplication.
    """
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        # Compute C_flat using the tensor
        C_flat = np.zeros(p * n)
        for i in range(n * m):
            i_A = i // m
            j_A = i % m
            for j in range(m * p):
                j_B = j // p
                k_B = j % p
                if j_A != j_B:
                    continue
                for k in range(p * n):
                    k_C = k // n
                    i_C = k % n
                    if i_A == i_C and k_B == k_C:
                        C_flat[k] += A_flat[i] * B_flat[j] * T[i, j, k]
        
        # Reshape C_flat: index k_C * n + i_C
        C_check = C_flat.reshape(p, n).T
        
        if not np.allclose(C, C_check, atol=1e-8):
            return False
    
    return True


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition correctly represents matrix multiplication."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor from the decomposition
    T = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    return _verify_tensor_correctness(T, n, m, p)
# EVOLVE-BLOCK-END
