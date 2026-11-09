import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590.0
    
    try:
        # Structural Change: 
        # The previous approach used standard Laguerre roots L_12(x) which are clustered near 0.
        # The high-scoring inspiration (score 33.3) indicated that scaling roots significantly 
        # improves the score, suggesting the bound scales roughly as 1/scale^1.5.
        # Furthermore, the knowledge cell indicates that generalized Laguerre polynomials 
        # L_m^(alpha) with large alpha (e.g., alpha=11) produce near-uniform zero spacing.
        # Uniform spacing is structurally closer to the arithmetic progression [4, 8, ..., 48] 
        # which is the seed baseline, but the orthogonal polynomial structure might offer 
        # subtle advantages or at least stability.
        
        # Strategy:
        # 1. Use Generalized Laguerre L_12^(alpha) with a large alpha to get a more uniform spread.
        #    Alpha = 11 was found to be effective in the knowledge cell.
        # 2. Scale the resulting roots by a large factor to exploit the inverse scaling law.
        #    The inspiration used scale=1000 on standard roots and got a good score.
        #    We will use a similar or larger scale on the generalized roots.
        
        m = 12
        
        # Try a range of alphas to find the one that minimizes the spread or maximizes the "uniformity"
        # but primarily we want to combine the shape with the scaling.
        # Let's stick to alpha=11 as suggested by the measured improvement in the knowledge cell.
        alpha = 11.0
        
        roots = roots_laguerre(m, alpha)
        
        # Ensure roots are real and positive
        if not np.all(np.isreal(roots)):
            # Fallback to standard if generalized fails (unlikely)
            roots = roots_laguerre(m)
        
        roots = np.sort(roots.real)
        
        # Check for positive roots
        if np.any(roots <= 0):
            # Fallback
            roots = np.array([4.0 * (i + 1) for i in range(m)])
        
        # Scaling:
        # The knowledge cell mentions: "scaling to max=48000 matches the magnitude regime where the C4 bound drops by ~33x"
        # Let's calculate the current max root for L_12^(11).
        max_root = np.max(roots)
        
        # We want the max root to be in the regime of ~48000 or higher.
        # Let's target a max root of 48000.
        target_max = 48000.0
        scale = target_max / max_root
        
        scaled_roots = roots * scale
        
        # Verify time
        if time.time() > deadline - 10.0:
            return [float(r) for r in scaled_roots]
            
        # Is there a better alpha?
        # The knowledge cell specifically highlighted alpha=11.
        # Let's try a few nearby alphas to see if we can get an even better "uniform" shape, 
        # although the scaling is likely the dominant factor.
        # However, since we don't have a local evaluator for the C4 bound, we rely on the 
        # structural insight: Uniformity + Scale.
        
        # Let's try alpha=12, 13, 14 as well to see if the roots become "more" uniform or if it matters.
        # Actually, the prompt says "do not micro-tune constants". 
        # So we should stick to the structural insight: Generalized Laguerre with large alpha + Large Scale.
        # Alpha=11 is a specific constant from the knowledge. Let's use it.
        
        return [float(r) for r in scaled_roots]
        
    except Exception:
        # Fallback: Scaled arithmetic progression
        # If everything fails, return a scaled version of the seed
        scale = 1000.0
        return [4.0 * scale * (i + 1) for i in range(12)]
