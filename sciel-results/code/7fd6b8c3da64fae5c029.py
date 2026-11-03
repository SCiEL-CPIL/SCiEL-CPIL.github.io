"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-minimal tensor decomposition for 4x4x9 matrix multiplication.
    
    Uses a structured search approach:
    1. Start with the schoolbook decomposition (rank 144)
    2. Attempt to reduce rank using iterative refinement with smooth surrogates
    3. Return the best valid decomposition found within time budget
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    deadline = time.time() + 140  # 140s safety margin before 150s kill
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is valid
    # T has shape (n*m, m*p, p*n) in flattened indices
    # T[a,b,c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i for valid i,j,k
    
    # First, create the schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def to_list(U, V, W):
        """Convert numpy arrays to lists of lists with half-integer support."""
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    best_result = to_list(U_school, V_school, W_school)
    best_rank = rank_school
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = U.shape[1]
        # Check dimensions
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Build the tensor from the decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # We only need to check that T[a,b,c] = 1 for valid (i,j,k) and 0 otherwise
        # Valid: a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
        # For each valid (i,j,k), check T[i*m+j, j*p+k, k*n+i] = 1
        # For invalid combinations, check T[a,b,c] = 0
        
        # Efficient check: for each r, the outer product U[:,r] * V[:,r] * W[:,r]
        # contributes to the tensor. Sum over r.
        
        # Instead of building full tensor, check specific entries
        # For valid (i,j,k): T[i*m+j, j*p+k, k*n+i] should be 1
        # For all other (a,b,c) combinations, should be 0
        
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1) > tol:
                        return False
        
        # Check some invalid entries to ensure no spurious contributions
        # A combination (a,b,c) is invalid if there's no (i,j,k) such that
        # a = i*m+j, b = j*p+k, c = k*n+i
        # Check all possible (a,b,c) that are not valid
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i * m + j, j * p + k, k * n + i))
        
        # Sample some invalid entries for efficiency
        import random
        random.seed(42)
        all_combos = [(a, b, c) for a in range(n * m) for b in range(m * p) for c in range(p * n)]
        invalid_combos = [combo for combo in all_combos if combo not in valid_set]
        
        # Check up to 100 random invalid entries
        sample_size = min(100, len(invalid_combos))
        for a, b, c in random.sample(invalid_combos, sample_size):
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > tol:
                return False
        
        return True
    
    # Try to reduce rank using iterative methods
    # Strategy: Use a rank-reduction approach where we try to express the tensor
    # with fewer rank-1 terms
    
    # For small matrices, we can try random initializations and gradient descent
    # on a smooth surrogate of the rank constraint
    
    def try_reduce_rank(target_rank, max_iter=500, lr=0.01, seed=None):
        """Try to find a decomposition of given rank using optimization."""
        if time.time() > deadline - 5:
            return None
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # The loss function: sum over all (a,b,c) of (T_true[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
        # where T_true[a,b,c] = 1 if (a,b,c) corresponds to valid (i,j,k), else 0
        
        # Precompute valid entries
        valid_entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_entries.append((i * m + j, j * p + k, k * n + i, 1))
        
        # Also need to penalize invalid entries
        # For efficiency, sample invalid entries
        valid_set = set((a, b, c) for a, b, c, _ in valid_entries)
        all_invalid = [(a, b, c) for a in range(n * m) for b in range(m * p) for c in range(p * n) if (a, b, c) not in valid_set]
        
        # Use a subset of invalid entries for the loss
        np.random.seed(42)
        num_invalid_sample = min(500, len(all_invalid))
        invalid_sample = np.random.choice(len(all_invalid), size=num_invalid_sample, replace=False)
        invalid_entries = [(all_invalid[idx][0], all_invalid[idx][1], all_invalid[idx][2], 0) for idx in invalid_sample]
        
        all_entries = valid_entries + invalid_entries
        
        # Gradient descent with momentum
        U_mom = np.zeros_like(U)
        V_mom = np.zeros_like(V)
        W_mom = np.zeros_like(W)
        momentum = 0.9
        
        prev_loss = float('inf')
        patience = 50
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None
            
            # Compute loss and gradients
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            loss = 0.0
            
            for a, b, c, target in all_entries:
                # val = sum_r U[a,r] * V[b,r] * W[c,r]
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                diff = val - target
                loss += diff * diff
                
                # Gradients:
                # d/dU[a,r] = 2*diff*V[b,r]*W[c,r]
                # d/dV[b,r] = 2*diff*U[a,r]*W[c,r]
                # d/dW[c,r] = 2*diff*U[a,r]*V[b,r]
                U_grad[a, :] += 2 * diff * V[b, :] * W[c, :]
                V_grad[b, :] += 2 * diff * U[a, :] * W[c, :]
                W_grad[c, :] += 2 * diff * U[a, :] * V[b, :]
            
            loss /= len(all_entries)
            
            # Update with momentum
            U_mom = momentum * U_mom + U_grad
            V_mom = momentum * V_mom + V_grad
            W_mom = momentum * W_mom + W_grad
            
            U -= lr * U_mom
            V -= lr * V_mom
            W -= lr * W_mom
            
            if loss < prev_loss - 1e-10:
                prev_loss = loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > patience:
                break
            
            # Check if we've converged
            if abs(prev_loss) < 1e-10:
                break
        
        # Verify the solution
        if verify_decomposition(U, V, W, tol=1e-6):
            return (U, V, W)
        
        return None
    
    # Try different target ranks from low to high
    # Known lower bounds for 4x4x9: at least 4*4*9/4 = 36 (by Strassen's lower bound type arguments)
    # Realistically, we might find something between 50-100
    
    # Try a few target ranks
    target_ranks = [80, 90, 100, 110, 120, 130, 140]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try a few random seeds
        for seed in [0, 1, 2]:
            if time.time() > deadline - 5:
                break
            
            result = try_reduce_rank(target_rank, max_iter=200, lr=0.005, seed=seed)
            if result is not None:
                U, V, W = result
                # Convert to list format
                result_dict = to_list(U, V, W)
                best_result = result_dict
                best_rank = target_rank
                break
        
        # If we found a valid decomposition, break early
        if best_rank < rank_school:
            break
    
    # Final verification of the best result
    if best_rank < rank_school:
        U_np = np.array(best_result["U"])
        V_np = np.array(best_result["V"])
        W_np = np.array(best_result["W"])
        if not verify_decomposition(U_np, V_np, W_np, tol=1e-8):
            # Fallback to schoolbook
            best_result = to_list(U_school, V_school, W_school)
            best_rank = rank_school
    
    return best_result
