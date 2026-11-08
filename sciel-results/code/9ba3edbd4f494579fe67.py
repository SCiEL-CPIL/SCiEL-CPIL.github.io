import numpy as np
import time
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # 1. Generate the default schoolbook decomposition (Rank 125)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook

    # 2. Attempt to find a lower rank decomposition using randomized search
    # We try ranks from 124 down to 100 (or until time runs out)
    # This is a heuristic search.
    
    # Define the tensor T indices for the multiplication
    # T[i, j, k] = 1 if i == k else 0
    # We need sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i, j, k]
    
    # To speed up, we can fix U to be random sparse and solve for V, W? 
    # Or fix U, V and solve for W?
    # Fixing U and V and solving for W is a linear system:
    # For each (i, k), we have a linear combination of W[k*n+i, :] 
    # Actually, for fixed i, k, the equation is:
    # sum_j sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1 if i==k else 0
    # This is not linear in W if U, V are fixed? 
    # Wait, W is a factor. The product is U * V * W? No, it's a tensor contraction.
    # The term is sum_t U_{im+j, t} V_{jp+k, t} W_{kn+i, t}.
    # If we fix U and V, the coefficient of W_{kn+i, t} is sum_j U_{im+j, t} V_{jp+k, t}.
    # Let C_{ik, t} = sum_j U_{im+j, t} V_{jp+k, t}.
    # Then the equation for each (i, k) is:
    # sum_t C_{ik, t} W_{kn+i, t} = delta_{ik}
    # This is a linear system for W.
    # The unknowns are W_{kn+i, t} for t=1..r.
    # There are n*p equations (for each i, k) and n*p*r unknowns?
    # No, W has shape (p*n, r). So there are p*n*r unknowns.
    # The equations are indexed by (i, k). There are n*p equations.
    # This is underdetermined if r > 1.
    # We can solve for W using least squares.
    
    for current_rank in range(rank_schoolbook - 1, max(50, rank_schoolbook - 20), -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        trials = 10
        for _ in range(trials):
            if time.time() > deadline:
                break
            
            # Generate random sparse U and V with entries in {-1, 0, 1}
            # Use a density of 0.5
            U_rand = np.random.choice([-1, 0, 1], size=(n * m, current_rank))
            V_rand = np.random.choice([-1, 0, 1], size=(m * p, current_rank))
            
            # Compute C_{ik, t} = sum_j U_{im+j, t} V_{jp+k, t}
            # C is a 3D tensor of shape (n, p, current_rank)
            # We can compute this efficiently using matrix multiplication or loops
            # C[i, k, t] = sum_j U[i, j, t] V[j, k, t]
            # Reshape U to (n, m, r), V to (m, p, r)
            U_reshaped = U_rand.reshape(n, m, current_rank)
            V_reshaped = V_rand.reshape(m, p, current_rank)
            
            # Sum over j (axis 1 of U, axis 0 of V)
            # We want C[i, k, t] = sum_j U_reshaped[i, j, t] * V_reshaped[j, k, t]
            # This is a batched matrix multiplication: U_reshaped (n, m, r) and V_reshaped (m, p, r)
            # We can do this by permuting V to (m, r, p) and then multiplying?
            # Or simply loop over t? No, vectorize.
            # C[i, k, t] = (U_reshaped[:, :, t] @ V_reshaped[:, :, t])? No.
            # U is (n, m, r), V is (m, p, r).
            # We want sum over j.
            # Let's use einsum: 'ijt,jkt->ikt'
            C = np.einsum('ijt,jkt->ikt', U_reshaped, V_reshaped)
            
            # Now we have C of shape (n, p, current_rank)
            # We need to solve for W of shape (p * n, current_rank)
            # The equation for each i, k is:
            # sum_t C[i, k, t] * W[k * n + i, t] = delta_{ik}
            
            # We can set up a linear system for each i, k?
            # Actually, the unknowns are the rows of W.
            # Let W_row_ik = W[k * n + i, :] (length current_rank)
            # The equation is C[i, k, :] @ W_row_ik = delta_{ik}
            # This is a system of n*p equations, each with current_rank unknowns.
            # We can solve this as a least squares problem for each row of W independently?
            # No, the rows of W are independent in the sense that the equation for (i, k)
            # only involves W[k*n+i, :].
            # So we can solve for each row of W separately!
            
            W_candidate = np.zeros((p * n, current_rank))
            valid = True
            max_residual = 0.0
            
            for i in range(n):
                for k in range(p):
                    # Equation: C[i, k, :] @ W_candidate[k*n+i, :] = delta_{ik}
                    # This is a system: A x = b, where A is 1 x current_rank, x is current_rank x 1, b is scalar.
                    # This is underdetermined (1 equation, current_rank unknowns).
                    # We can find the minimum norm solution using pseudo-inverse.
                    # Or we can just check if a solution exists with small integer/half-integer entries.
                    # Since it's underdetermined, there are infinite solutions.
                    # We can pick one.
                    # Let A = C[i, k, :]
                    # If A is all zeros, we need delta_{ik} == 0. If i==k, it's 1, so no solution.
                    # If i!=k, delta is 0, so any W works (e.g., 0).
                    # If A is not zero, we can find a solution.
                    
                    A = C[i, k, :]
                    b = 1.0 if i == k else 0.0
                    
                    if np.allclose(A, 0):
                        if b != 0:
                            valid = False
                            break
                        else:
                            # Any solution works, pick 0
                            W_candidate[k * n + i, :] = 0
                    else:
                        # Find minimum norm solution x = A^T / (A @ A^T)
                        # x_t = A_t * b / sum(A^2)
                        norm_sq = np.dot(A, A)
                        if norm_sq < 1e-10:
                            if b != 0:
                                valid = False
                                break
                            W_candidate[k * n + i, :] = 0
                        else:
                            W_candidate[k * n + i, :] = (A / norm_sq) * b
                        
            if not valid:
                continue
                
            # Check if W_candidate has integer or half-integer entries
            # Multiply by 2 and check if close to integers
            W_check = W_candidate * 2
            W_rounded = np.round(W_check)
            if np.allclose(W_check, W_rounded, atol=1e-6):
                # We found a valid decomposition!
                best_U = U_rand
                best_V = V_rand
                best_W = W_candidate / 2.0  # Convert back to original scale
                best_rank = current_rank
                # We can break early if we find a very low rank, but let's try to go lower
                # However, finding a valid one is rare, so we might want to stop if rank is low
                if best_rank < 120:
                    break

    # Convert to list of lists as required by the interface
    # The interface expects {'U', 'V', 'W'} with entries
    # The seed program returns lists of lists.
    # Let's return numpy arrays? The seed returns lists.
    # "factor matrices of shapes (n*m, rank)... with integer/half-integer entries"
    # Usually, returning numpy arrays is fine if the evaluator handles it, 
    # but the seed returns lists. Let's stick to lists to be safe, 
    # or check if the evaluator expects lists.
    # The seed code: U = [[0] * rank for _ in range(n * m)]
    # So it expects lists.
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
