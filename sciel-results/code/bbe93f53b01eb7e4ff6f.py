# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv

# Constants
L = 10
N_THETA = 70
N_PHI = 130

# Precompute quadrature nodes and weights
u_nodes, w_weights = np.polynomial.legendre.leggauss(N_THETA)
phi_nodes = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

# Precompute flattened weights
w_flat = np.repeat(w_weights, N_PHI) * (2.0 * np.pi / N_PHI)

# Precompute Spherical Harmonics Y_matrix
Y_matrix = np.zeros((N_THETA * N_PHI, 21), dtype=np.float64)

def fact(n):
    return math.factorial(n)

for m in range(-L, L + 1):
    abs_m = abs(m)
    N_m = math.sqrt(21.0 / (4.0 * np.pi) * fact(L - abs_m) / fact(L + abs_m))
    P_vals = lpmv(abs_m, L, u_nodes)
    
    if m == 0:
        Y_flat = np.tile(P_vals * N_m, (1, N_PHI)).flatten()
    else:
        if m > 0:
            cos_term = np.cos(m * phi_nodes)
            Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, cos_term)
            Y_flat = Y_2d.flatten()
        else:
            sin_term = np.sin(abs_m * phi_nodes)
            Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, sin_term)
            Y_flat = Y_2d.flatten()
    
    Y_matrix[:, m + 10] = Y_flat

def compute_ratio(c):
    """
    Computes R = ||f||_4 / ||f||_2
    """
    f = Y_matrix @ c
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.dot(f2, w_flat)
    norm4_sq = np.dot(f4, w_flat)
    
    if norm2_sq <= 1e-15:
        return 1e15
    
    norm2 = math.sqrt(norm2_sq)
    norm4 = math.sqrt(math.sqrt(norm4_sq))
    
    return norm4 / norm2

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0 # 10s safety margin
    
    best_c = None
    best_score = 1e15
    
    # Strategy: Stochastic Differential Equation (SDE) based optimization on the sphere.
    # We optimize on the unit sphere of coefficient space (S^20) to handle scale invariance.
    # The dynamics are: dS_t = -\eta \nabla_S J(S_t) dt + \sqrt{2 \eta} dW_t
    # where J(S) = R(S). Since we are on the sphere, we project the gradient.
    # However, computing the gradient of R with respect to S is expensive.
    # Instead, we use a heuristic: Random Walk with "Hill Climbing" steps.
    # Or, we can use the fact that R is smooth and use finite differences or analytic gradients.
    
    # Let's try a different structural approach:
    # Minimize R using a Gradient Descent method with analytic gradients.
    # R(c) = (sum w_i f_i^4)^0.25 / (sum w_i f_i^2)^0.5
    # Let N4 = (sum w_i f_i^4)^0.25 and N2 = (sum w_i f_i^2)^0.5
    # R = N4 / N2
    # dR/dc = (1/N2) dN4/dc - (N4/N2^2) dN2/dc
    # dN4/dc = (1/4) (sum w_i f_i^4)^(-0.75) * 4 * sum w_i f_i^3 * (Y_matrix[:, k])
    #        = N4^(-3) * sum w_i f_i^3 * Y_i
    # dN2/dc = (1/2) (sum w_i f_i^2)^(-0.5) * 2 * sum w_i f_i * (Y_matrix[:, k])
    #        = N2^(-1) * sum w_i f_i * Y_i
    
    # Let A = sum w_i f_i^3 * Y_i  (vector of size 21)
    # Let B = sum w_i f_i * Y_i    (vector of size 21)
    # dR/dc_k = (1/N2) * N4^(-3) * A_k - (N4/N2^2) * N2^(-1) * B_k
    #        = (N4^(-3) / N2) * A_k - (N4 / N2^3) * B_k
    
    # This allows for efficient gradient computation.
    
    def gradient(c):
        f = Y_matrix @ c
        f2 = f * f
        f4 = f2 * f2
        
        norm2_sq = np.dot(f2, w_flat)
        norm4_sq = np.dot(f4, w_flat)
        
        if norm2_sq <= 1e-15 or norm4_sq <= 1e-15:
            return np.zeros(21)
            
        norm2 = math.sqrt(norm2_sq)
        norm4 = math.sqrt(math.sqrt(norm4_sq))
        
        # A_k = sum_i w_i f_i^3 Y_{i,k} = (f^3 * w_flat) @ Y_matrix
        # B_k = sum_i w_i f_i Y_{i,k} = (f * w_flat) @ Y_matrix
        
        A = (f**3 * w_flat) @ Y_matrix
        B = (f * w_flat) @ Y_matrix
        
        # dR/dc = (N4^(-3) / N2) * A - (N4 / N2^3) * B
        coeff_A = (norm4 ** -3) / norm2
        coeff_B = norm4 / (norm2 ** 3)
        
        grad = coeff_A * A - coeff_B * B
        
        return grad
    
    # We will run multiple independent runs of Gradient Descent with momentum.
    # Since the objective is scale invariant, we will normalize c to unit norm after each step.
    # This effectively performs optimization on the sphere.
    
    n_runs = 50
    steps_per_run = 1000
    
    # Initialize with diverse starting points
    rng = np.random.default_rng(42)
    
    # 1. Single m seeds
    for i in range(21):
        c = np.zeros(21)
        c[i] = 1.0
        c = c / np.linalg.norm(c)
        
        # Run Gradient Descent
        lr = 0.01
        momentum = 0.9
        v = np.zeros(21)
        
        for step in range(steps_per_run):
            if time.time() > deadline:
                break
                
            g = gradient(c)
            
            # Project gradient onto tangent space of sphere (optional but good for stability)
            # grad_tan = g - (g . c) c
            # Since we normalize anyway, this might not be strictly necessary, but helps.
            grad_tan = g - np.dot(g, c) * c
            
            v = momentum * v + grad_tan
            c = c - lr * v
            
            # Normalize
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-6:
                break
            c = c / norm_c
            
            score = compute_ratio(c)
            if score < best_score:
                best_score = score
                best_c = c.copy()
        
        if time.time() > deadline:
            break
            
    # 2. Random seeds
    for _ in range(n_runs):
        if time.time() > deadline:
            break
            
        c = rng.standard_normal(21)
        c = c / np.linalg.norm(c)
        
        # Adaptive Learning Rate
        lr = 0.01
        momentum = 0.9
        v = np.zeros(21)
        
        # Annealing: reduce lr over time
        for step in range(steps_per_run):
            if time.time() > deadline:
                break
                
            g = gradient(c)
            grad_tan = g - np.dot(g, c) * c
            
            v = momentum * v + grad_tan
            c = c - lr * v
            
            norm_c = np.linalg.norm(c)
            if norm_c < 1e-6:
                break
            c = c / norm_c
            
            # Simple annealing
            lr = max(0.001, 0.01 * (1.0 - step / steps_per_run))
            
            score = compute_ratio(c)
            if score < best_score:
                best_score = score
                best_c = c.copy()
            
    # 3. Pairs of m
    for i in range(21):
        for j in range(i+1, 21):
            if time.time() > deadline:
                break
                
            c = np.zeros(21)
            c[i] = 1.0
            c[j] = 1.0
            c = c / np.linalg.norm(c)
            
            lr = 0.01
            momentum = 0.9
            v = np.zeros(21)
            
            for step in range(steps_per_run):
                if time.time() > deadline:
                    break
                    
                g = gradient(c)
                grad_tan = g - np.dot(g, c) * c
                
                v = momentum * v + grad_tan
                c = c - lr * v
                
                norm_c = np.linalg.norm(c)
                if norm_c < 1e-6:
                    break
                c = c / norm_c
                
                lr = max(0.001, 0.01 * (1.0 - step / steps_per_run))
                
                score = compute_ratio(c)
                if score < best_score:
                    best_score = score
                    best_c = c.copy()
        
        if time.time() > deadline:
            break

    if best_c is None:
        best_c = np.zeros(21)
        best_c[20] = 1.0
        
    # Ensure validity
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c = best_c / max_c * 1.0
    elif max_c > 1e5:
        best_c = best_c / max_c * 1.0
        
    return best_c

# EVOLVE-BLOCK-END
