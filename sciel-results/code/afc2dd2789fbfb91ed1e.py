"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Golden-angle (Fibonacci) spiral points on the unit sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def energy(X):
        # Calculate pairwise squared distances
        XX = X @ X.T
        D2 = 2.0 - 2.0 * XX
        # Ensure diagonal is 1 to avoid log(0) issues, though we mask it later
        np.fill_diagonal(D2, 1.0)
        D = np.sqrt(D2)
        
        # Sum log(1/D) for i < j
        iu = np.triu_indices(N, k=1)
        # Use np.sum for speed in optimization loop
        return -np.sum(np.log(D[iu]))

    def gradient(X):
        XX = X @ X.T
        D2 = 2.0 - 2.0 * XX
        np.fill_diagonal(D2, 1.0)
        
        # Weights 1/D2
        W = 1.0 / D2
        np.fill_diagonal(W, 0.0)
        
        # G_i = sum_{j!=i} (x_j - x_i) / D2_ij
        # G = W @ X - X @ (W.sum(axis=1))
        V = W @ X
        S = W.sum(axis=1, keepdims=True)
        G = V - X * S
        return G

    E_curr = energy(X)
    alpha = 1e-3
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    while time.time() < deadline:
        G = gradient(X)
        
        # Project gradient to tangent space
        # G_tan = G - X (G . X^T)
        proj = G @ X.T
        G_tan = G - X * proj.T
        
        # Check if gradient is effectively zero
        g_norm = np.linalg.norm(G_tan)
        if g_norm < 1e-10:
            break
            
        # Line search
        accepted = False
        for _ in range(30):
            X_trial = X - alpha * G_tan
            # Normalize to sphere
            norms = np.linalg.norm(X_trial, axis=1, keepdims=True)
            X_trial = X_trial / norms
            
            E_trial = energy(X_trial)
            
            # Check for non-finite values
            if not np.isfinite(E_trial):
                alpha *= 0.5
                continue
                
            if E_trial < E_curr:
                X = X_trial
                E_curr = E_trial
                accepted = True
                # Increase alpha slightly if step was small
                if alpha < 1e-2:
                    alpha *= 1.05
                break
            else:
                alpha *= 0.5
                
        if not accepted:
            # If we couldn't find a step, reset alpha to a moderate value
            # to try to escape potential numerical stalls
            alpha = 1e-3
            # If alpha is still too small after reset, we might be stuck
            # But with reset, it's 1e-3 again.
            # To avoid infinite loop of resets, we can track failures.
            # For simplicity, we just break if we fail too many times in a row?
            # Or just continue, as time will run out.
            # Let's add a failure counter.
            
        # Small decay to ensure convergence
        alpha *= 0.999

    return X
# EVOLVE-BLOCK-END
