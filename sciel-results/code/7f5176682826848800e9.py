"""Matrix multiplication 4x5 by 5x5 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    Returns: {'U', 'V', 'W'} with shapes (20, rank), (25, rank), (20, rank)
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    
    # Deadline guard: 150s wall-clock limit, use 115s for safety
    start_time = time.time()
    deadline = start_time + 115
    
    # Build schoolbook decomposition (rank = 100)
    schoolbook_rank = n * m * p  # 100
    U_school = [[0] * schoolbook_rank for _ in range(nm)]
    V_school = [[0] * schoolbook_rank for _ in range(mp)]
    W_school = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Verify using numpy for speed."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        r = U_np.shape[1]
        
        # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We need T[i,j,k] = 1 for all valid (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def snap_to_half_integers(arr, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.round(arr * 2.0) / 2.0
        if np.max(np.abs(arr - snapped)) < tol:
            return snapped
        return arr
    
    def try_sgd_sparse(rank, max_iters=5000, lr=0.01, l1_lambda=0.01, num_restarts=5):
        """
        Try SGD with L1 regularization to find a sparse rank-r decomposition.
        """
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with random small values
            np.random.seed(restart * 12345 + 678)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Precompute indices
            indices = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        indices.append((i * m + j, j * p + k, k * n + i))
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Sample a random subset of equations
                batch_size = min(50, len(indices))
                batch = random.sample(indices, batch_size)
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                loss = 0.0
                for ui, vi, wi in batch:
                    # T[ui, vi, wi] = sum_r U[ui,r] * V[vi,r] * W[wi,r]
                    # Target = 1.0
                    # dL/dU[ui,r] = V[vi,r] * W[wi,r] * (T - 1)
                    # dL/dV[vi,r] = U[ui,r] * W[wi,r] * (T - 1)
                    # dL/dW[wi,r] = U[ui,r] * V[vi,r] * (T - 1)
                    
                    T_val = np.dot(U[ui], V[vi] * W[wi])
                    diff = T_val - 1.0
                    loss += diff * diff
                    
                    grad_U[ui] += V[vi] * W[wi] * diff
                    grad_V[vi] += U[ui] * W[wi] * diff
                    grad_W[wi] += U[ui] * V[vi] * diff
                
                # L1 gradient
                grad_U += l1_lambda * np.sign(U)
                grad_V += l1_lambda * np.sign(V)
                grad_W += l1_lambda * np.sign(W)
                
                # SGD update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if it % 100 == 0:
                    # Verify
                    valid = True
                    for ui, vi, wi in indices:
                        T_val = np.dot(U[ui], V[vi] * W[wi])
                        if abs(T_val - 1.0) > 1e-6:
                            valid = False
                            break
                    if valid:
                        # Snap to half-integers
                        U_snapped = snap_to_half_integers(U)
                        V_snapped = snap_to_half_integers(V)
                        W_snapped = snap_to_half_integers(W)
                        
                        # Verify snapped version
                        if verify_fast(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()):
                            return {
                                "U": U_snapped.tolist(),
                                "V": V_snapped.tolist(),
                                "W": W_snapped.tolist()
                            }
        
        return None
    
    def try_structured_reduction(target_rank):
        """
        Try structured rank reduction using algebraic identities.
        For 4x5*5x5, try to merge terms.
        """
        if time.time() > deadline:
            return None
        
        # Try different structured patterns
        for trial in range(20):
            if time.time() > deadline:
                return None
            
            # Start with schoolbook
            U = [row[:] for row in U_school[:nm]]
            V = [row[:] for row in V_school[:mp]]
            W = [row[:] for row in W_school[:pn]]
            
            # Try to eliminate terms by setting to 0 and adjusting
            indices = list(range(schoolbook_rank))
            random.shuffle(indices)
            
            eliminated = 0
            for idx in indices:
                if eliminated >= schoolbook_rank - target_rank:
                    break
                if time.time() > deadline:
                    break
                
                # Try to eliminate this term
                for i in range(nm):
                    U[i][idx] = 0
                for j in range(mp):
                    V[j][idx] = 0
                for k in range(pn):
                    W[k][idx] = 0
                
                if verify_fast(U, V, W):
                    eliminated += 1
            
            if eliminated >= schoolbook_rank - target_rank:
                U = [row[:target_rank] for row in U]
                V = [row[:target_rank] for row in V]
                W = [row[:target_rank] for row in W]
                if verify_fast(U, V, W):
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    def try_random_perturbation(target_rank, num_trials=30):
        """Try random perturbations from schoolbook."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            U = []
            for i in range(nm):
                row = []
                for c in range(target_rank):
                    if c < schoolbook_rank:
                        val = U_school[i][c]
                        if random.random() < 0.15:
                            val += random.choice([-1, 1])
                    else:
                        val = random.choice([-1, 0, 1])
                    row.append(val)
                U.append(row)
            
            V = []
            for j in range(mp):
                row = []
                for c in range(target_rank):
                    if c < schoolbook_rank:
                        val = V_school[j][c]
                        if random.random() < 0.15:
                            val += random.choice([-1, 1])
                    else:
                        val = random.choice([-1, 0, 1])
                    row.append(val)
                V.append(row)
            
            W = []
            for k in range(pn):
                row = []
                for c in range(target_rank):
                    if c < schoolbook_rank:
                        val = W_school[k][c]
                        if random.random() < 0.15:
                            val += random.choice([-1, 1])
                    else:
                        val = random.choice([-1, 0, 1])
                    row.append(val)
                W.append(row)
            
            if verify_fast(U, V, W):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from 99 down to 85
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 15, 0), -1):
        if time.time() > deadline:
            break
        
        # Try structured approach first
        result = try_structured_reduction(target_rank)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
        
        # Try random perturbation
        result = try_random_perturbation(target_rank, num_trials=15)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
        
        # Try SGD with L1 regularization for this rank
        result = try_sgd_sparse(target_rank, max_iters=2000, lr=0.02, l1_lambda=0.02, num_restarts=3)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
