import numpy as np
from scipy.special import roots_laguerre
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590.0
    
    try:
        # Strategy: 
        # The current program uses generalized Laguerre L_12^(11) scaled to max=48000.
        # The strongest inspiration (score 33.3095) used standard Laguerre L_12 scaled by 10000 (max ~33900).
        # Both scored ~33.3, suggesting the shape (standard vs generalized) matters less than the scale/magnitude.
        # The directive asks to optimize the shape parameter beta (alpha in scipy notation) to maximize effective bandwidth 
        # or minimize variance of the node distribution.
        
        # However, "minimizing variance" of nodes for a fixed span would mean clustering them, which likely worsens the bound 
        # if the bound depends on the spread (as scaling up improves it). 
        # Conversely, "maximizing effective bandwidth" might imply a more uniform spread over the domain to cover more frequency range efficiently.
        # Standard Laguerre (alpha=0) has roots clustered near 0.
        # Large alpha (e.g., 11) makes roots more uniform.
        # The seed (arithmetic progression) is perfectly uniform.
        # Score 33.3 was achieved with both standard+scale=10000 and generalized(alpha=11)+scale=48000/max.
        
        # Let's analyze the scaling more carefully.
        # If score ~ 1/Bound, and Bound ~ 1/Scale^k.
        # Seed: roots ~ [4..48], max=48. Score=1.
        # Insp1: roots ~ [0.5..33.9] * 10000 = [5000..339000]. Wait, roots_laguerre(12) max is ~33.9.
        # 339000 / 48 = ~7000. If scale factor is 7000, and score is 33, then 33 ~ 7000^k? No.
        # Let's look at the ratio of max roots.
        # Seed max: 48.
        # Insp1 max: 33.9 * 10000 = 339000. Ratio: 339000/48 = 7062.
        # Current max: 48000. Ratio: 48000/48 = 1000.
        # If Insp1 (Ratio 7062) scored 33.3 and Current (Ratio 1000) scored 33.3, then the score does NOT scale with max root simply.
        # This implies there is an optimal scale or the bound has a complex dependency.
        
        # Wait, the current program code says:
        # "scaling to max=48000 matches the magnitude regime where the C4 bound drops by ~33x"
        # And Insp1 says: "Scale=1000 gave score ~33.3" (on roots_laguerre(12)).
        # Let's check Insp1's max: roots_laguerre(12) max is approx 33.9. Scale 1000 -> 33900.
        # Current max: 48000.
        # 33900 vs 48000. Both give ~33.3.
        # This suggests the bound is roughly constant in this range [30k, 50k]? Or perhaps the "score" is normalized differently?
        # The prompt says: "score is the upper bound on C4". "raw value (lower is better)". "search score is seed/value".
        # So if raw value is the bound B, score = B_seed / B.
        # If score = 33, then B = B_seed / 33.
        # If both ~33k and ~48k max roots yield score 33, then B is similar.
        
        # Hypothesis: The bound depends on the *spacing* or *density* of roots, not just the max.
        # Or, there is a specific optimal scale.
        # The directive says: "optimize the shape parameter beta ... to maximize the effective bandwidth".
        # Let's try to find a better scale by testing a few candidates around the known good region.
        # Since we cannot evaluate locally, we must rely on the structural insight.
        # "Uniform spacing" (alpha -> infinity) is the seed structure.
        # Standard Laguerre (alpha=0) is clustered.
        # Alpha=11 is intermediate.
        
        # Let's try a larger alpha to get even more uniform roots, and adjust the scale to maintain a similar max magnitude, 
        # OR try to push the scale further if the previous "plateau" was due to sub-optimal shape.
        
        # Let's test alpha = 100 (very uniform) and scale to max = 48000.
        # And also test alpha = 0 (standard) and scale to max = 48000 (which is scale ~1415).
        
        # Actually, let's look at the "effective bandwidth".
        # If the bound is related to the integral of the Laguerre combination, larger alpha spreads roots out.
        # Let's try a range of alphas and pick the one that minimizes the *relative* variance of the roots?
        # No, the directive says "maximize effective bandwidth".
        
        # Let's assume the bound is minimized when the roots are as spread out as possible (uniform) 
        # AND scaled appropriately.
        
        # Let's try alpha=50 (very uniform) and scale to max=48000.
        # And alpha=100 and scale to max=48000.
        
        # Since we can't evaluate, we have to pick one.
        # The current program uses alpha=11.
        # Inspiration 1 uses alpha=0.
        # Both scored 33.3.
        
        # Maybe the scale is the key.
        # Insp1: max=33900.
        # Current: max=48000.
        # What if max=100000 is better?
        # Let's try max=100000 with alpha=11.
        
        # Let's try a few configurations and return the one with the largest max root, 
        # assuming larger scale is generally better until numerical limits, 
        # but constrained by the "shape" to be uniform.
        
        # Let's use alpha=20 (more uniform than 11) and scale to max=100000.
        
        m = 12
        alpha = 20.0
        
        roots = roots_laguerre(m, alpha)
        roots = np.sort(roots.real)
        
        if np.any(roots <= 0):
            roots = np.array([4.0 * (i + 1) for i in range(m)])
        
        max_root = np.max(roots)
        target_max = 100000.0
        scale = target_max / max_root
        
        scaled_roots = roots * scale
        
        if time.time() > deadline - 10.0:
            return [float(r) for r in scaled_roots]
            
        return [float(r) for r in scaled_roots]
        
    except Exception:
        # Fallback: Scaled arithmetic progression
        scale = 1000.0
        return [4.0 * scale * (i + 1) for i in range(12)]
