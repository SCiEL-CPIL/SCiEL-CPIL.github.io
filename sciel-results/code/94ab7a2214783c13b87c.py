"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline (rank 42)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        # Reshape to 3D tensors
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Using einsum for efficiency
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        
        # Check if all entries are 1
        return np.all(T == 1)

    # Local search for a lower rank
    def local_search(target_rank, max_steps=5000):
        """
        Attempt to find a valid decomposition of target_rank.
        Starts from a random initialization with values in {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}.
        Uses hill climbing on the error metric (number of non-1 entries).
        """
        # We use integer arithmetic internally to avoid floating point issues,
        # but the problem allows half-integers. 
        # To handle half-integers, we can multiply all entries by 2 and search for integers,
        # then divide by 2 at the end. 
        # Let's stick to integers {-1, 0, 1} first for speed, then expand if needed.
        # Actually, half-integers are often key in tensor decompositions (e.g. Strassen).
        # Let's use values from {-1, -0.5, 0, 0.5, 1}.
        # To keep it simple and fast, we'll use integers {-2, -1, 0, 1, 2} representing half-integers * 2.
        
        vals = [-2, -1, 0, 1, 2]
        
        U = np.zeros((nm, target_rank), dtype=int)
        V = np.zeros((mp, target_rank), dtype=int)
        W = np.zeros((pn, target_rank), dtype=int)
        
        # Initialize with random values
        U[:] = np.random.choice(vals, size=(nm, target_rank))
        V[:] = np.random.choice(vals, size=(mp, target_rank))
        W[:] = np.random.choice(vals, size=(pn, target_rank))
        
        # Compute initial error
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # We want T[i,j,k] == 2 (since we scaled by 2, 1 becomes 2? No, 1 * 1 * 1 = 1. 
        # If we use half-integers, e.g., 0.5, 0.5, 4.0 -> 1.0.
        # Let's just use floats for the search to be safe, but check for near-integers.
        # Or better: Use integers and allow values {-1, 0, 1}. 
        # If no solution found with integers, try half-integers.
        
        # For speed, let's try integer search first with {-1, 0, 1}.
        vals_int = [-1, 0, 1]
        U[:] = np.random.choice(vals_int, size=(nm, target_rank))
        V[:] = np.random.choice(vals_int, size=(mp, target_rank))
        W[:] = np.random.choice(vals_int, size=(pn, target_rank))
        
        U3 = U.reshape(n, m, target_rank)
        V3 = V.reshape(m, p, target_rank)
        W3 = W.reshape(p, n, target_rank)
        
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        error = np.sum(T != 1)
        
        if error == 0:
            return U, V, W
            
        best_error = error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and entry
            mat_choice = np.random.randint(3)
            
            if mat_choice == 0:
                r, c = np.random.randint(nm), np.random.randint(target_rank)
                old_val = U[r, c]
                new_val = np.random.choice(vals_int)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                # Update T
                # T[i,j,k] += delta * V[j,k,c] * W[k,i,c]
                i, j = divmod(r, m)
                # V3[j, :, c] is shape (p,)
                # W3[:, i, c] is shape (p,) -> need to align indices
                # T[i, j, :] += delta * V3[j, :, c] * W3[:, i, c]
                T[i, j, :] += delta * V3[j, :, c] * W3[:, i, c]
                U[r, c] = new_val
                U3[i, j, c] = new_val
                
            elif mat_choice == 1:
                r, c = np.random.randint(mp), np.random.randint(target_rank)
                old_val = V[r, c]
                new_val = np.random.choice(vals_int)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                j, k = divmod(r, p)
                # T[:, j, k] += U[:, j, c] * delta * W[k, :, c]
                # U3[:, j, c] is shape (n,)
                # W3[k, :, c] is shape (n,)
                T[:, j, k] += U3[:, j, c] * delta * W3[k, :, c]
                V[r, c] = new_val
                V3[j, k, c] = new_val
                
            else:
                r, c = np.random.randint(pn), np.random.randint(target_rank)
                old_val = W[r, c]
                new_val = np.random.choice(vals_int)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                k, i = divmod(r, n)
                # T[i, :, k] += U[i, :, c] * V[:, k, c] * delta
                # U3[i, :, c] is shape (m,)
                # V3[:, k, c] is shape (m,)
                T[i, :, k] += U3[i, :, c] * V3[:, k, c] * delta
                W[r, c] = new_val
                W3[k, i, c] = new_val
            
            # Calculate new error
            # To save time, we only check if error decreased or if we are close to 0
            # Full error check is O(nmp)
            # We can approximate by counting changes, but exact is safer for correctness.
            # Since nmp=42, it's very fast.
            current_error = np.sum(T != 1)
            
            if current_error < best_error:
                best_error = current_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_error == 0:
                    return best_U, best_V, best_W
            else:
                # Revert move
                if mat_choice == 0:
                    T[i, j, :] -= delta * V3[j, :, c] * W3[:, i, c]
                    U[r, c] = old_val
                    U3[i, j, c] = old_val
                elif mat_choice == 1:
                    T[:, j, k] -= U3[:, j, c] * delta * W3[k, :, c]
                    V[r, c] = old_val
                    V3[j, k, c] = old_val
                else:
                    T[i, :, k] -= U3[i, :, c] * V3[:, k, c] * delta
                    W[r, c] = old_val
                    W3[k, i, c] = old_val
                    
        return None

    # Try to find a lower rank
    # Start from best_rank - 1 down to a reasonable lower bound
    # Lower bound is max(nm, mp, pn) = 21
    # We try a few ranks near the top first, as they are more likely to have solutions if any exist.
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random restarts
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            result = local_search(target_rank, max_steps=2000)
            if result is not None:
                U, V, W = result
                # Final verification
                if verify(U.tolist(), V.tolist(), W.tolist(), target_rank):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # If we found a solution, we can stop searching for lower ranks 
            # because we want the minimal rank. 
            # However, finding a solution at rank 41 doesn't mean 40 is impossible.
            # But given time constraints, we take the first one found.
            # To be safe, we could continue, but it's risky.
            # Let's assume the first found is good enough for an improvement.
            break

    return best
# EVOLVE-BLOCK-END
