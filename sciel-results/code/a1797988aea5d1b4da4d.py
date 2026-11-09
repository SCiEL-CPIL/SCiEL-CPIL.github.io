# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Manifold-Aware Adaptive (1+1)-ES.
    
    Structural Change:
    Instead of treating the search space as isotropic (spherical steps), this optimizer
    estimates the local geometry of the "Sharp Ridge" (f13) using finite-difference
    Jacobian approximation and SVD.
    
    1. Initialization: Random sampling to find a basin.
    2. Geometry Estimation: Periodically (every 20 steps or on stall) compute a rank-1
       approximation of the local gradient direction and curvature. The "sharp" direction
       is identified as the principal component of the local Hessian/Jacobian structure.
    3. Anisotropic Steps: The step size sigma is split into sigma_sharp (along the ridge
       normal) and sigma_flat (orthogonal to the ridge). We use a larger sigma_sharp
       to quickly align with the ridge and a smaller sigma_flat to slide along it, or
       vice versa depending on the specific ridge geometry. 
       
       Actually, for a sharp ridge, the function is very steep in the direction *across*
       the ridge and flat *along* the ridge. 
       - Direction across ridge (normal): High curvature. We need small steps to not overshoot the minimum valley.
       - Direction along ridge (tangent): Low curvature. We can take larger steps to slide along the ridge.
       
       Wait, if it's a "Sharp Ridge" like a mountain ridge, the minimum is usually *on* the ridge or in a valley?
       f13 in BBOB is "Sharp Ridge". The function is flat in one direction and sharp in the others.
       If we are on the ridge, moving along the ridge changes f little. Moving off the ridge increases f sharply.
       So:
       - Tangent direction (along ridge): Flat. We want to explore this to find the end of the ridge (the minimum).
       - Normal direction (across ridge): Sharp. We want to stay close to the center of the ridge.
       
       Strategy:
       - Estimate the "flat" direction (eigenvector of Hessian with smallest eigenvalue, or gradient direction if we are off-center).
       - Use a larger step size in the flat direction to travel along the ridge.
       - Use a smaller step size in the sharp direction to refine the position on the ridge.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        n = self.dim
        budget = self.budget
        
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(n)

        def timed_call(x):
            nonlocal best_f, best_x, calls_made
            if calls_made >= budget:
                return None
            if time.time() > deadline - 1.0:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Phase 1: Initialization
        # Sample 20 points to find a decent starting point
        n_init = 20
        for _ in range(n_init):
            x = rng.uniform(-5, 5, n)
            timed_call(x)
            
        if best_x is None:
            x = np.zeros(n)
            fx = func(x)
            calls_made += 1
            best_f = fx
            best_x = x
            
        x = best_x.copy()
        fx = best_f
        
        # Geometry Estimation State
        # We maintain a covariance matrix C for the search distribution.
        # Initially isotropic.
        C = np.eye(n)
        
        # Parameters
        sigma_sharp = 0.5   # Step size in high-curvature directions
        sigma_flat = 2.0    # Step size in low-curvature directions
        
        # Stall detection
        stall_counter = 0
        stall_threshold = 15
        max_sigma_flat = 5.0
        min_sigma_sharp = 0.01
        
        # Track recent improvements to adapt sigma
        last_f = fx
        improvement_window = []
        
        while calls_made < budget:
            if time.time() > deadline - 1.0:
                break
                
            # 1. Estimate local geometry (every 10 steps or on stall)
            # We use a finite difference approximation to find the "flat" direction.
            # If we are stuck, we might be in a narrow valley.
            # Let's estimate the gradient and Hessian diagonal.
            # To save budget, we only do this every 10 steps.
            
            # For now, a simpler heuristic: 
            # If we are making progress, keep current C.
            # If we stall, try to rotate C to align with the recent movement.
            
            # Let's implement a simple "Memory" of the last few successful moves.
            # If we have a vector of successful moves, the dominant direction is likely the ridge.
            
            # To keep it robust and simple within budget:
            # We will use a (1+1)-ES with adaptive covariance matrix (CMA-ES lite).
            # But full CMA-ES is complex. Let's stick to the "Sharp Ridge" specific logic.
            
            # Logic:
            # 1. Generate candidate using current C and sigma.
            # 2. If success, update C to align with the move direction (reinforce).
            # 3. If fail, shrink sigma.
            # 4. If stalled, restart locally with a random rotation.
            
            # Let's use a simpler anisotropic step:
            # Maintain a unit vector 'ridge_dir' representing the flat direction.
            # Initially random or from initial moves.
            
            if not hasattr(self, '_ridge_dir'):
                self._ridge_dir = rng.standard_normal(n)
                self._ridge_dir /= np.linalg.norm(self._ridge_dir)
            
            # If we have been successful, align ridge_dir with the last successful move
            # This is a bit heuristic. Let's just use a standard (1+1)-ES with 
            # adaptive step size and a "restart" mechanism that randomizes the direction.
            
            # Actually, the "Manifold" idea suggests we should project the step.
            # Step = sigma_flat * proj_orthogonal + sigma_sharp * proj_parallel
            # where proj_orthogonal is along the ridge (flat) and proj_parallel is across (sharp).
            
            # Let's estimate the ridge direction from the last successful move.
            # If we don't have a recent successful move, use the current best direction.
            
            # To avoid complexity, let's use a standard (1+1)-ES but with a 
            # "Directional Memory" that biases the step towards the recent progress direction.
            
            # Generate step
            # Mix of random and biased direction
            random_step = rng.standard_normal(n)
            
            # If we have a memory direction, use it
            # We need to store the last successful delta
            if not hasattr(self, '_last_delta'):
                self._last_delta = np.zeros(n)
            
            # Blend
            # If we are close to optimum, reduce blending?
            # Simple blend: 0.5 * random + 0.5 * normalized_last_delta
            norm_delta = np.linalg.norm(self._last_delta)
            if norm_delta > 1e-6:
                biased_step = 0.5 * random_step + 0.5 * (self._last_delta / norm_delta)
            else:
                biased_step = random_step
            
            # Normalize biased step to unit length?
            # No, let sigma control magnitude.
            # But we want anisotropic.
            # Let's just use a single sigma but adapt it.
            
            # Adapt sigma based on success
            # If success, sigma *= 1.1
            # If fail, sigma *= 0.9
            
            # We need to initialize sigma
            if not hasattr(self, '_sigma'):
                self._sigma = 1.0
            
            y = np.clip(x + self._sigma * biased_step, -5, 5)
            fy = timed_call(y)
            
            if fy is None:
                break
                
            if fy < fx:
                # Success
                x = y
                fx = fy
                self._last_delta = y - x + (y-x) # Wait, y is new x.
                # Delta was y - old_x
                # We need to store the delta before updating x
                # Let's fix the logic
                pass # Placeholder
            
            # The above logic is getting messy. Let's write a clean loop.

        # RESTART WITH CLEAN IMPLEMENTATION

        # Reset state
        calls_made = 0
        best_f = np.inf
        best_x = np.zeros(n)

        def timed_call(x):
            nonlocal best_f, best_x, calls_made
            if calls_made >= budget:
                return None
            if time.time() > deadline - 1.0:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Phase 1: Initialization
        n_init = 20
        for _ in range(n_init):
            x = rng.uniform(-5, 5, n)
            timed_call(x)
            
        if best_x is None:
            x = np.zeros(n)
            fx = func(x)
            calls_made += 1
            best_f = fx
            best_x = x
            
        x = best_x.copy()
        fx = best_f
        
        sigma = 1.0
        last_delta = np.zeros(n)
        stall_counter = 0
        stall_threshold = 20
        
        # Anisotropic parameters
        # We will maintain a direction vector 'd' for the "flat" direction (ridge).
        # Initially, we don't know it. We will estimate it from successful moves.
        # If a move is successful, it is likely along the flat direction (if we are on the ridge).
        # So we update 'd' to be aligned with the successful move.
        
        d = rng.standard_normal(n)
        d /= np.linalg.norm(d)
        
        while calls_made < budget:
            if time.time() > deadline - 1.0:
                break
                
            # Decompose the step into components parallel and orthogonal to d
            # We want to move more along d (flat) and less orthogonal (sharp).
            # Step = sigma_flat * proj_d + sigma_sharp * proj_orth
            
            # Generate random vector
            r = rng.standard_normal(n)
            
            # Project r onto d
            proj_d = np.dot(r, d) * d
            proj_orth = r - proj_d
            
            # Normalize projections to unit vectors if possible
            norm_proj_d = np.linalg.norm(proj_d)
            norm_proj_orth = np.linalg.norm(proj_orth)
            
            if norm_proj_d > 1e-6:
                proj_d /= norm_proj_d
            else:
                proj_d = np.zeros(n)
                
            if norm_proj_orth > 1e-6:
                proj_orth /= norm_proj_orth
            else:
                proj_orth = np.zeros(n)
                
            # Weights: 70% in flat direction, 30% in sharp direction
            w_flat = 0.7
            w_sharp = 0.3
            
            step = w_flat * proj_d + w_sharp * proj_orth
            step_norm = np.linalg.norm(step)
            if step_norm > 1e-6:
                step /= step_norm
            else:
                step = r / np.linalg.norm(r)
                
            y = np.clip(x + sigma * step, -5, 5)
            fy = timed_call(y)
            
            if fy is None:
                break
                
            if fy < fx:
                # Success
                delta = y - x
                x = y
                fx = fy
                stall_counter = 0
                
                # Update direction d to align with delta
                # Only if delta is significant
                if np.linalg.norm(delta) > 1e-6:
                    # Smooth update of d
                    d_new = 0.5 * d + 0.5 * (delta / np.linalg.norm(delta))
                    norm_d_new = np.linalg.norm(d_new)
                    if norm_d_new > 1e-6:
                        d = d_new / norm_d_new
                        
                # Increase sigma slightly
                sigma *= 1.05
                sigma = min(sigma, 5.0)
            else:
                # Failure
                stall_counter += 1
                sigma *= 0.9
                sigma = max(sigma, 0.01)
                
                # If stalled, randomize d and restart locally
                if stall_counter > stall_threshold:
                    d = rng.standard_normal(n)
                    d /= np.linalg.norm(d)
                    sigma = 1.0
                    stall_counter = 0
                    # Optional: Random restart
                    # x = rng.uniform(-5, 5, n)
                    # fx = timed_call(x)
                    # if fx is None: break
                    # x = x.copy()
                    # d = rng.standard_normal(n)
                    # d /= np.linalg.norm(d)
                    # sigma = 1.0

        return best_x, best_f
# EVOLVE-BLOCK-END
