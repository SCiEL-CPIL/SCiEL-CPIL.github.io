import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_value(h):
        """Compute the exact objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        # The problem defines value = abs(2 * N * np.max(conv) / np.sum(h) ** 2)
        # Note: np.max(conv) takes the maximum signed entry.
        # If the max entry is negative, the value is negative? 
        # "upper bound on constant C3... lower is better"
        # Usually C3 is positive. Let's assume the standard definition where 
        # we want to minimize the peak sidelobe relative to the main lobe.
        # The prompt says: value = abs(2 * N * np.max(conv) / np.sum(h) ** 2)
        # If max(conv) is negative, abs() makes it positive.
        # However, usually the main lobe is at index N-1 (or 0 depending on conv definition) 
        # and is positive if h is real and sum is non-zero? 
        # Actually, for real h, conv[h, h] is symmetric. 
        # If h is all positive, max is positive.
        # If h has negative parts, max could be negative? 
        # Let's stick to the formula exactly.
        return abs(2.0 * N * np.max(conv) / (s * s))

    def smooth_max(x, tau=0.01):
        """Soft approximation of max using log-sum-exp."""
        x_max = np.max(x)
        if tau < 1e-8:
            return x_max
        # Shift for numerical stability
        shifted = (x - x_max) / tau
        return x_max + tau * np.log(np.sum(np.exp(shifted)))

    def phase_to_h(phase):
        """
        Convert phase vector to time-domain heights h.
        We fix the magnitude of the DFT to 1 (constant magnitude spectrum).
        h = IDFT( exp(1j * phase) )
        Since h must be real, we enforce conjugate symmetry in H.
        H[k] = exp(1j * theta[k]) for k=0..N/2
        H[N-k] = conj(H[k])
        
        Alternatively, simpler:
        Generate a random/optimized complex vector H with |H[k]|=1.
        h = real(IDFT(H)).
        This h will be real.
        We optimize the phases theta.
        """
        # Construct H with conjugate symmetry to ensure real h
        # H is length N.
        # For real h, H[k] = conj(H[N-k]).
        # We can define the first N//2 + 1 phases and mirror them.
        
        # Let's use a simpler parameterization:
        # h = real( IDFT( exp(1j * phase) ) )
        # But IDFT of arbitrary phase doesn't guarantee real h unless phase is symmetric.
        # So we construct H explicitly.
        
        H = np.zeros(N, dtype=complex)
        # k=0 is DC, must be real for h to be real? No, H[0] must be real if h is real.
        # Actually, if h is real, H is Hermitian.
        # H[0] is real. H[N/2] is real (if N even).
        # H[k] and H[N-k] are conjugates.
        
        # Let's parameterize by the phases of the positive frequencies.
        # theta[0] is phase of DC (must be 0 or pi for H[0] to be real? 
        # If H[0] is complex, h is complex. We take real part.
        # But taking real part of IDFT(H) where H has arbitrary phases 
        # is equivalent to projecting onto real subspace.
        
        # A cleaner way:
        # Let phi be a vector of length N.
        # H = exp(1j * phi)
        # h_complex = IDFT(H)
        # h = real(h_complex)
        # This h is real.
        # We optimize phi.
        
        H = np.exp(1j * phase)
        h_complex = np.fft.ifft(H)
        h = np.real(h_complex)
        
        # Normalize to avoid zero sum issues? 
        # The objective is scale invariant: 
        # value(h * c) = abs(2*N*max(conv(h*c))/(sum(h*c))^2) 
        # = abs(2*N*c^2*max(conv(h))/(c^2*sum(h)^2)) = value(h)
        # So scale doesn't matter. 
        # However, numerical stability might prefer normalization.
        # We'll return h as is.
        return h

    def objective_from_phase(phase, tau=0.05):
        """
        Objective function for phase optimization.
        Uses smooth max to make it differentiable.
        """
        h = phase_to_h(phase)
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        conv = np.convolve(h, h)
        sm = smooth_max(conv, tau=tau)
        return 2.0 * N * sm / (s * s)

    # Initial best: seed
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Strategy: Optimize the phase of a constant-magnitude spectrum.
    # We start with a random phase or a structured phase (e.g., linear chirp phase).
    # A linear phase in frequency corresponds to a time shift (doesn't change autocorrelation peak).
    # A quadratic phase (chirp) in frequency corresponds to a chirp in time.
    # Let's initialize with a random phase and then optimize.
    
    # To improve reproducibility and quality, we can try a few initializations.
    # 1. Random
    # 2. Linear Chirp (Zadoff-Chu like)
    
    rng = np.random.default_rng(42)
    
    # Initialization 1: Random Phases
    phase_init1 = rng.uniform(-np.pi, np.pi, N)
    
    # Initialization 2: Linear Chirp Phases (quadratic in freq)
    # H[k] ~ exp(1j * a * k^2)
    k = np.arange(N)
    # Center k for symmetry
    k_centered = k - N//2
    phase_init2 = 0.5 * np.pi * (k_centered / (N//2))**2
    
    candidates = [phase_init1, phase_init2]
    
    # We will run L-BFGS-B on the phase vector.
    # N=400 parameters. L-BFGS-B should handle this.
    
    best_phase = None
    best_phase_val = np.inf
    
    for i, p_init in enumerate(candidates):
        if time.time() > deadline - 30.0:
            break
            
        # Wrap objective to handle periodicity? 
        # L-BFGS-B works on R^N. Phases are periodic.
        # We can just optimize in R^N and let it find a local min.
        # The landscape might have many local minima.
        
        # Use a tight tau for the final polish, looser for initial steps?
        # Let's use a fixed moderate tau for the optimization, then refine with exact max.
        
        def obj_wrapper(p, tau=0.05):
            return objective_from_phase(p, tau=tau)
        
        try:
            # L-BFGS-B with bounds? Phases are unbounded in R, but periodic.
            # We can just let it run.
            result = minimize(
                obj_wrapper,
                p_init,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            h_opt = phase_to_h(result.x)
            val_opt = compute_value(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
                best_phase = result.x.copy()
                best_phase_val = val_opt
                print(f"Phase Opt {i}: val={best_val:.6f}", file=sys.stderr)
                
        except Exception as e:
            print(f"Phase Opt {i} error: {e}", file=sys.stderr)

    # If we found a good phase, we can try to refine it further with a smaller tau
    # to sharpen the max approximation.
    if best_phase is not None and time.time() < deadline - 10.0:
        try:
            def obj_tight(p, tau=0.005):
                return objective_from_phase(p, tau=tau)
            
            result2 = minimize(
                obj_tight,
                best_phase,
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 20
                }
            )
            
            h_opt2 = phase_to_h(result2.x)
            val_opt2 = compute_value(h_opt2)
            
            if val_opt2 < best_val:
                best_val = val_opt2
                best_h = h_opt2.copy()
                print(f"Phase Refine: val={best_val:.6f}", file=sys.stderr)
                
        except Exception as e:
            print(f"Phase Refine error: {e}", file=sys.stderr)

    # Final safety check
    if time.time() > deadline - 2.0:
        return np.ones(N)

    if best_val < 2.0 and np.all(np.isfinite(best_h)) and abs(np.sum(best_h)) > 1e-6:
        return best_h
    else:
        return np.ones(N)
