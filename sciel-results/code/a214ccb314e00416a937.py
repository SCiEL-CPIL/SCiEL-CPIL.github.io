import numpy as np
import random
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 130  # Safety margin

    # Schoolbook baseline (Rank 42)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.int64)
        V = np.zeros((mp, rank), dtype=np.int64)
        W = np.zeros((pn, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def compute_error(U, V, W, rank):
        """Compute error tensor and count non-zeros."""
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        # Target is identity-like? No, for matrix multiplication A (n x m) * B (m x p) -> C (n x p)
        # The tensor T[i,j,k] should be 1 if we are computing the product?
        # Wait, the decomposition is:
        # C_{ik} = sum_j A_{ij} B_{jk}
        # In tensor form, the multiplication tensor T has T[i,j,k] = 1.
        # The decomposition is T = sum_c U[:, :, c] * V[:, :, c] * W[:, :, c]
        # So we want sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k.
        # Note: The indices in the task description:
        # U (n*m, rank), V (m*p, rank), W (p*n, rank)
        # The standard trace convention W[k*n+i, r] implies W is indexed by (k, i).
        # The contraction is usually C_{ik} = sum_j A_{ij} B_{jk}.
        # The tensor T_{ijk} = 1.
        # Decomposition: T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
        # So yes, T should be all ones.
        err = T - 1
        return int(np.count_nonzero(err)), err

    def verify(U, V, W, rank):
        count, _ = compute_error(U, V, W, rank)
        return count == 0

    def simulated_annealing(target_rank, max_iters=20000, initial_temp=1.0, cooling=0.999):
        """
        Simulated Annealing to find a rank-target_rank decomposition.
        """
        # Initialize sparsely
        U = np.zeros((nm, target_rank), dtype=np.int64)
        V = np.zeros((mp, target_rank), dtype=np.int64)
        W = np.zeros((pn, target_rank), dtype=np.int64)
        
        # Seed with some random non-zeros to avoid trivial zero solution
        for c in range(target_rank):
            if random.random() < 0.5:
                U[random.randint(0, nm-1), c] = random.choice([-1, 1])
                V[random.randint(0, mp-1), c] = random.choice([-1, 1])
                W[random.randint(0, pn-1), c] = random.choice([-1, 1])
        
        current_err_count, _ = compute_error(U, V, W, target_rank)
        best_err_count = current_err_count
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        temp = initial_temp
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick a random matrix and entry to perturb
            mat_choice = random.random()
            if mat_choice < 0.33:
                idx = (random.randint(0, nm-1), random.randint(0, target_rank-1))
                mat = U
            elif mat_choice < 0.66:
                idx = (random.randint(0, mp-1), random.randint(0, target_rank-1))
                mat = V
            else:
                idx = (random.randint(0, pn-1), random.randint(0, target_rank-1))
                mat = W
            
            old_val = mat[idx]
            delta = random.choice([-2, -1, 1, 2])
            new_val = old_val + delta
            
            # Keep values small to help sparsity and integer nature
            if abs(new_val) > 5:
                continue
                
            mat[idx] = new_val
            
            new_err_count, _ = compute_error(U, V, W, target_rank)
            
            if new_err_count < current_err_count:
                current_err_count = new_err_count
                if new_err_count < best_err_count:
                    best_err_count = new_err_count
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err_count == 0:
                        return True, best_U, best_V, best_W
            else:
                # Accept with probability exp(-delta_err / temp)
                # Here delta_err is increase in error count
                delta_err = new_err_count - current_err_count
                if random.random() < np.exp(-delta_err / max(temp, 1e-6)):
                    current_err_count = new_err_count
                else:
                    mat[idx] = old_val # Reject
            
            temp *= cooling
            
        # Return best found
        return best_err_count == 0, best_U, best_V, best_W

    # Search for lower ranks
    # Theoretical lower bound is 21.
    # We try from 21 upwards.
    for target_rank in range(21, best_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # Multiple restarts
        found = False
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            success, U, V, W = simulated_annealing(target_rank, max_iters=5000, initial_temp=1.0, cooling=0.9995)
            if success:
                found = True
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
        
        if found:
            # If we found a solution at this rank, we can stop if it's the lower bound
            if target_rank == 21:
                break
            # Otherwise, continue to see if we can go lower? No, we are iterating up.
            # If we found 22, we already tried 21 and failed. So 22 is the best so far.
            # We should break if we found a solution, as lower ranks were already tried.
            break

    return best_result
# EVOLVE-BLOCK-END
