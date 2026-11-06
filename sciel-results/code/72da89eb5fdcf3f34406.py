import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _compute_dft_l1(coeffs, N):
    """
    Compute the L1 norm of the DFT of the polynomial square.
    coeffs: array of coefficients in ascending order.
    N: size of the DFT (power of 2).
    """
    # Pad coefficients to length N
    padded = np.zeros(N)
    padded[:len(coeffs)] = coeffs
    
    # Compute polynomial square in frequency domain
    # P(x)^2 evaluated at roots of unity is |P(w^k)|^2
    P_hat = np.fft.fft(padded)
    square_hat = P_hat * P_hat  # This is P(w^k)^2, not |P(w^k)|^2
    
    # The time-domain coefficients of P^2 are the inverse DFT of square_hat
    # We want to minimize the L1 norm of these time-domain coefficients
    # L1 norm of time domain = L1 norm of inverse DFT of square_hat
    # However, for optimization, we can work with the magnitude of the spectrum
    # or directly with the time-domain L1 norm.
    # Let's use the time-domain L1 norm of the square directly, 
    # but computed via DFT for speed? No, inverse DFT is O(N log N).
    # The square coefficients are:
    sq_coeffs = np.fft.ifft(square_hat).real
    # The L1 norm is sum(|sq_coeffs|)
    return np.sum(np.abs(sq_coeffs))

def _compute_grad_dft_l1(coeffs, N):
    """
    Approximate gradient of the L1 norm of the DFT square.
    This is a heuristic gradient.
    """
    padded = np.zeros(N)
    padded[:len(coeffs)] = coeffs
    P_hat = np.fft.fft(padded)
    square_hat = P_hat * P_hat
    sq_coeffs = np.fft.ifft(square_hat).real
    
    # Gradient of sum(|x|) is sign(x)
    # Chain rule: d/dcoeffs sum(|ifft(fft(coeffs)^2)|)
    # This is complex. Let's use a finite difference approximation for the gradient
    # or a simpler surrogate.
    
    # Surrogate: Minimize sum(|sq_coeffs|).
    # sq_coeffs = ifft(P_hat^2)
    # Let's use the gradient of the squared L2 norm of the square as a smooth surrogate
    # which often correlates with sparsity, or just use the L1 norm directly with subgradients.
    
    # For simplicity and speed, let's use a random perturbation approach (simulated annealing style)
    # or a simple gradient step on the L2 norm of the square coefficients, which is differentiable.
    # L2 norm of square = sum(sq_coeffs^2) = N * sum(|P_hat|^4) / N^2 ?
    # Actually, Parseval's theorem: sum(|sq_coeffs|^2) = (1/N) sum(|square_hat|^2)
    # sum(|P_hat^2|^2) = sum(|P_hat|^4)
    # So minimizing L2 norm of square is equivalent to minimizing sum(|P_hat|^4).
    
    P_abs_sq = np.abs(P_hat) ** 2
    # Objective: sum(P_abs_sq^2)
    # Gradient w.r.t P_hat: 4 * P_hat * conj(P_hat)^2
    # Gradient w.r.t coeffs: ifft( grad_P_hat )
    
    grad_P_hat = 4 * P_hat * (np.conj(P_hat) ** 2)
    grad_coeffs = np.fft.ifft(grad_P_hat).real
    
    return grad_coeffs[:len(coeffs)]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_score = float('inf')
    best_coeffs = None
    
    # Initial best: the seed program [1]*12
    seed_coeffs = np.array([1.0] * 12)
    seed_score = _compute_dft_l1(seed_coeffs, 4096)
    best_score = seed_score
    best_coeffs = seed_coeffs.copy()
    
    # We will run multiple random restarts
    num_restarts = 50
    steps_per_restart = 100
    N = 4096  # DFT size
    
    # We want to find a polynomial with 12 nonzero terms.
    # We will represent the polynomial as a vector of length 41 (max degree 40).
    # We enforce sparsity by only updating a subset of coefficients or by resetting others.
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random sparse vector
        # Choose 12 random positions for non-zero coefficients
        positions = np.random.choice(41, 12, replace=False)
        coeffs = np.zeros(41)
        coeffs[positions] = np.random.choice([-1, 1], size=12)
        
        # Optional: Initialize with some structure, e.g., arithmetic progression
        if r % 2 == 0:
            # Arithmetic progression initialization
            start = np.random.randint(0, 30)
            step = np.random.randint(1, 5)
            positions = [start + i * step for i in range(12) if start + i * step < 41]
            if len(positions) < 12:
                # Fill remaining with random
                remaining = [p for p in range(41) if p not in positions]
                positions += random.sample(remaining, 12 - len(positions))
            coeffs = np.zeros(41)
            coeffs[positions] = np.random.choice([-1, 1], size=12)
        
        # Gradient descent on the surrogate objective (L2 norm of square)
        # We will alternate between continuous optimization and integer rounding
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Compute gradient
            grad = _compute_grad_dft_l1(coeffs, N)
            
            # Learning rate decay
            lr = 0.1 / (1 + step * 0.01)
            
            # Update
            coeffs = coeffs - lr * grad
            
            # Periodically round to integers and enforce sparsity
            if step % 10 == 0:
                # Round to nearest integer
                coeffs_int = np.round(coeffs)
                
                # Enforce bounds [-64, 64]
                coeffs_int = np.clip(coeffs_int, -64, 64)
                
                # Enforce sparsity: keep only the 12 largest magnitude coefficients
                # This is a heuristic to maintain the "12 nonzero terms" constraint
                # while allowing the positions to change
                magnitudes = np.abs(coeffs_int)
                # Get indices of top 12 magnitudes
                # If there are ties, break randomly
                indices = np.argsort(magnitudes)[::-1][:12]
                new_coeffs = np.zeros(41)
                new_coeffs[indices] = coeffs_int[indices]
                
                # If the new configuration is valid (12 nonzeros, int, in bounds),
                # evaluate its true score (L1 norm of square coefficients)
                # Note: _compute_dft_l1 returns L1 norm, which is what we want to minimize
                # But wait, the task says "Minimize nonzero terms in the EXACT integer square".
                # The score is the number of nonzero terms in the square.
                # My surrogate minimizes the L1 norm of the square coefficients.
                # These are correlated but not identical.
                # I should evaluate the actual number of nonzero terms in the square for the best candidates.
                
                # Let's compute the actual square and count nonzeros
                sq_coeffs = np.convolve(new_coeffs, new_coeffs)
                # Remove floating point noise
                sq_coeffs = np.round(sq_coeffs)
                nonzero_count = np.sum(np.abs(sq_coeffs) > 1e-9)
                
                if nonzero_count < best_score:
                    best_score = nonzero_count
                    best_coeffs = new_coeffs.copy()
                    
                # Reset coeffs to the integer version for next iteration
                # But add some noise to escape local minima?
                # Or just continue with the continuous version from the rounded point
                coeffs = new_coeffs + np.random.normal(0, 0.1, 41)
                
                # Ensure 12 nonzeros in coeffs for the next step?
                # The gradient step will likely make them all non-zero.
                # That's fine, the rounding step will prune them.

    # Final check: ensure best_coeffs meets all constraints
    # 1. Length <= 41
    # 2. Nonzero final coefficient
    # 3. Entries integers in [-64, 64]
    # 4. Exactly 12 nonzero terms
    
    if best_coeffs is None:
        return [1] * 12
        
    best_coeffs = np.round(best_coeffs)
    best_coeffs = np.clip(best_coeffs, -64, 64).astype(int)
    
    # Check 12 nonzero terms
    nonzero_idx = np.nonzero(best_coeffs)[0]
    if len(nonzero_idx) != 12:
        # If not 12, we need to fix it.
        # This is a failure of the heuristic. Fallback to seed.
        # Or try to fix: if >12, zero out smallest. If <12, add random 1s.
        if len(nonzero_idx) > 12:
            magnitudes = np.abs(best_coeffs)
            indices = np.argsort(magnitudes)[:len(nonzero_idx) - 12]
            best_coeffs[indices] = 0
        else:
            zero_idx = np.nonzero(best_coeffs == 0)[0]
            if len(zero_idx) > 0:
                to_add = 12 - len(nonzero_idx)
                chosen = np.random.choice(zero_idx, to_add, replace=False)
                best_coeffs[chosen] = 1
            else:
                # No space, just return seed
                return [1] * 12
                
    # Check nonzero final coefficient
    if best_coeffs[-1] == 0:
        # Find last nonzero
        last_nz = np.nonzero(best_coeffs)[0][-1]
        if last_nz < 40:
            # Move one term to the end? Or just set last to 1 and zero out another?
            # This might break the 12 count if we just set it.
            # Let's zero out the smallest magnitude term and set the last to 1
            magnitudes = np.abs(best_coeffs)
            # Exclude the last index (which is 0)
            mask = np.ones(41, dtype=bool)
            mask[-1] = False
            indices = np.argsort(magnitudes * mask)
            idx_to_zero = indices[0]
            best_coeffs[idx_to_zero] = 0
            best_coeffs[-1] = 1
        else:
            # Last term is already nonzero? No, we checked best_coeffs[-1] == 0.
            # This case is impossible if len(nonzero_idx)==12 and last is 0.
            pass
            
    # Convert to list
    result = best_coeffs.tolist()
    
    # Final validation
    if len(result) != 41:
        # Pad or trim? The task says length <= 41.
        # If we have leading zeros, we can trim? No, "ascending degree" usually implies fixed length or trimmed.
        # "Return coefficient list in ascending degree, length<=41"
        # If I return [1, 0, 1], it's degree 2.
        # If I return [1, 0, 0, 1], it's degree 3.
        # The evaluator likely checks the list properties.
        # Let's trim trailing zeros? No, "nonzero final coefficient" implies the last element must be nonzero.
        # So we must keep length such that last is nonzero.
        # My code ensures best_coeffs[-1] != 0.
        # So length is 41.
        pass
        
    return result

# EVOLVE-BLOCK-END
