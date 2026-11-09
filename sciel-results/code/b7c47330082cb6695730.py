"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: ensure we finish well before 330s
    start_time = time.time()
    deadline = start_time + 300  # 300s safety margin
    
    # Estimate the linear operator A by probing with basis vectors
    # f(t, y) = A @ y
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
        
    # Check deadline after A extraction
    if time.time() > deadline:
        # Fallback: use dense expm if we're close to deadline
        dt = t1 - t0
        M = sla.expm(A * dt)
        y_end = M @ y0
        return y_end
    
    # Try to exploit Kronecker sum structure: A = L ⊗ I + I ⊗ L
    # For 2D heat on 10x10 grid, n=100, so grid_size=10
    grid_size = int(np.sqrt(n))
    
    if grid_size * grid_size == n:
        # Try to extract 1D Laplacian structure
        # A should be of form L ⊗ I + I ⊗ L where L is 1D Laplacian
        # Reshape A to 4D tensor and try to separate
        A_4d = A.reshape(grid_size, grid_size, grid_size, grid_size)
        
        # Check if A has Kronecker sum structure
        # A[i,j,k,l] should be L[i,k]*delta[j,l] + delta[i,j]*L[k,l]
        # Extract L from diagonal blocks
        L = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                # A[i, j, :, j] should give L[:, j] * delta[i,j] + delta[i,j] * L[:, :]
                # When i==j: A[i,i,:,i] = L[:,i] + L[:,i] ... wait, let me think more carefully
                # A = L⊗I + I⊗L
                # A[(i,j),(k,l)] = L[i,k]*delta[j,l] + delta[i,j]*L[k,l]
                # So A[i,j,k,l] = L[i,k]*(1 if j==l else 0) + (1 if i==j else 0)*L[k,l]
                
                if i == j:
                    # A[i,i,k,l] = L[i,k]*(1 if i==l else 0) + L[k,l]
                    # When l == i: A[i,i,k,i] = L[i,k] + L[k,i]
                    # When l != i: A[i,i,k,l] = L[k,l]
                    pass
        
        # Simpler approach: extract L from the structure
        # A[(i,j),(k,l)] = L[i,k]*δ(j,l) + δ(i,j)*L[k,l]
        # Set j=l=0: A[(i,0),(k,0)] = L[i,k] + δ(i,0)*L[k,0]
        # Set i=j=0: A[(0,j),(0,l)] = L[0,0]*δ(j,l) + L[0,l]
        
        # Better: use the fact that for i≠j and k≠l, A[i,j,k,l] = 0
        # For i=j, k≠l: A[i,i,k,l] = L[k,l] (since δ(i,i)=1 but L[i,k]*δ(i,l)=0 when l≠i)
        # Wait, that's not right either. Let me be more careful.
        
        # A[i,j,k,l] = L[i,k] * (1 if j==l else 0) + (1 if i==j else 0) * L[k,l]
        
        # Case 1: i ≠ j and k ≠ l: A[i,j,k,l] = 0
        # Case 2: i = j, k ≠ l: A[i,i,k,l] = L[k,l] (first term: L[i,k]*δ(i,l)=0 since l≠i, second: L[k,l])
        # Case 3: i ≠ j, k = l: A[i,j,k,k] = L[i,k] (first: L[i,k]*δ(j,k), second: 0)
        # Case 4: i = j, k = l: A[i,i,k,k] = L[i,k]*δ(i,k) + L[k,k]
        
        # Extract L from Case 2: for i, k≠l where l≠i
        L_extracted = np.zeros((grid_size, grid_size))
        for k in range(grid_size):
            for l in range(grid_size):
                if k != l:
                    # Use any i where i != l
                    i_val = 0 if l != 0 else 1
                    L_extracted[k, l] = A[i_val * grid_size + i_val, k * grid_size + l]
                    # Wait, I'm indexing wrong. A is flat (n,n), reshape to (gs,gs,gs,gs)
                    # A_flat[i*gs+j, k*gs+l] = A_4d[i,j,k,l]
        
        # Let me redo this with proper indexing
        # A_flat[row, col] where row = i*gs + j, col = k*gs + l
        # A_4d[i,j,k,l] = A_flat[i*gs+j, k*gs+l]
        
        L_extracted = np.zeros((grid_size, grid_size))
        for k in range(grid_size):
            for l in range(grid_size):
                if k != l:
                    # Choose i such that i != l to avoid the δ(i,l) term
                    i_val = 0 if l != 0 else 1
                    row = i_val * grid_size + i_val  # i=j=i_val
                    col = k * grid_size + l
                    L_extracted[k, l] = A[row, col]
        
        # Now fill diagonal of L from Case 4 or other constraints
        # For i=j, k=l: A[i,i,k,k] = L[i,k]*δ(i,k) + L[k,k]
        # If i≠k: A[i,i,k,k] = L[k,k]
        # If i=k: A[i,i,i,i] = L[i,i] + L[i,i] = 2*L[i,i]
        
        for k in range(grid_size):
            # Find i ≠ k
            i_val = 0 if k != 0 else 1
            row = i_val * grid_size + i_val
            col = k * grid_size + k
            L_extracted[k, k] = A[row, col]
        
        # Verify the structure by reconstructing A and comparing
        I_gs = np.eye(grid_size)
        A_reconstructed = np.kron(L_extracted, I_gs) + np.kron(I_gs, L_extracted)
        
        # Check if reconstruction is close to original
        if np.allclose(A, A_reconstructed, rtol=1e-10, atol=1e-12):
            # Use eigenvalue decomposition for efficient expm
            # exp(A*dt) = exp((L⊗I + I⊗L)*dt) = exp(L*dt) ⊗ exp(L*dt)
            # Since L is symmetric (for standard heat equation), use eigendecomposition
            try:
                eigvals_L, eigvecs_L = np.linalg.eigh(L_extracted)
                exp_eigvals = np.exp(eigvals_L * (t1 - t0))
                exp_L = (eigvecs_L * exp_eigvals) @ eigvecs_L.T
                M = np.kron(exp_L, exp_L)
                y_end = M @ y0
                return y_end
            except np.linalg.LinAlgError:
                pass
        
        # Fallback if Kronecker structure doesn't match
        dt = t1 - t0
        M = sla.expm(A * dt)
        y_end = M @ y0
        return y_end
    else:
        # Fallback for non-square grid
        dt = t1 - t0
        M = sla.expm(A * dt)
        y_end = M @ y0
        return y_end
# EVOLVE-BLOCK-END
