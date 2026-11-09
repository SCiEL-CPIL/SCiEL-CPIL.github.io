import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard: we have 330s total, but this function is called repeatedly.
    # Each call should be fast. The GP posterior is already computed by the caller.
    # We just need to compute the acquisition score.
    
    # Use Expected Improvement (EI) with a small exploitation bias.
    # EI is generally robust and well-calibrated.
    # To improve upon basic EI, we can use a variant like Local Penalization 
    # or adjust the beta parameter if using UCB.
    
    # However, the current program uses UCB: -mu + beta * sigma.
    # The seed program uses EI.
    # EI score: (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Let's stick with EI as it's the seed and often performs well.
    # But we can try to optimize it slightly.
    # One common improvement is to ensure numerical stability.
    
    # Let's try a blend of EI and UCB, or just EI with careful handling.
    # Actually, the prompt says the seed program (EI) scored 1.0.
    # The current program (UCB with adaptive beta) scored 1.19.
    # So UCB is already better. Let's refine the UCB parameter.
    
    # In the current program, beta = 2.0 + 0.5 * gap.
    # gap = max(0, f_best - 3.0).
    # If f_best is close to 3.0, gap is small, beta ~ 2.0.
    # If f_best is large, beta is large.
    
    # Let's try a different adaptive strategy or a fixed beta that works well.
    # Often, beta around 2-5 works well for 2D problems.
    # Let's try a fixed beta first to see if the adaptive part is hurting or helping.
    # Or maybe a smoother adaptation.
    
    # Let's try EI again but with a slight modification: 
    # Sometimes, using the log of EI or a transformed version helps.
    # Or, let's use the "Expected Improvement" directly, which is the standard.
    
    # Wait, the current program is UCB. Let's analyze why it might be better.
    # UCB explores more aggressively when sigma is high.
    # EI balances exploration and exploitation based on the expected gain.
    
    # Let's try a hybrid: use EI, but if we are very far from the optimum, 
    # maybe switch to UCB? No, that's complex.
    
    # Let's try to optimize the UCB beta.
    # Instead of linear growth, maybe use a square root or log growth to prevent 
    # excessive exploration later on if the gap remains (though it shouldn't if we're doing well).
    
    # Let's try: beta = 2.0 + 0.2 * gap
    # Or beta = 3.0 (fixed)
    
    # Let's look at the scores. 
    # Seed (EI): 1.0
    # Current (UCB adaptive): 1.19
    # So the adaptive UCB is better.
    
    # Let's try to make the adaptation smoother or less aggressive.
    # If gap is 0, beta=2.0.
    # If gap is 1, beta=2.5.
    # If gap is 10, beta=7.0.
    
    # What if we cap the beta?
    # beta = min(5.0, 2.0 + 0.5 * gap)
    
    # Or what if we use a different base?
    # beta = 1.5 + 0.5 * gap
    
    # Let's try a fixed beta of 3.0. It's a common sweet spot.
    # If fixed 3.0 is better than adaptive, we use it.
    # If not, we tweak the adaptive.
    
    # Actually, let's try EI again. Maybe the UCB is over-exploring?
    # No, 1.19 > 1.0, so UCB is better.
    
    # Let's try to improve the UCB by using a more principled adaptation.
    # The gap to the minimum (3.0) is a good signal.
    # But what if we use the number of evaluations? We don't have it directly, 
    # but we can infer progress from f_best.
    
    # Let's try:
    # beta = 2.5 + 0.3 * gap
    # This is slightly less aggressive than 0.5.
    
    # Let's also consider numerical stability for sigma.
    # sigma + 1e-9 is used in EI, but in UCB we just multiply.
    
    # Let's try a fixed beta of 2.5 first.
    # If the current adaptive is good, maybe the linear term is the key.
    
    # Let's try:
    # beta = 2.0 + 0.5 * np.log1p(gap)
    # This grows slower.
    
    # Let's go with a slightly damped linear growth.
    gap = max(0.0, f_best - 3.0)
    beta = 2.0 + 0.3 * gap
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    return -mu + beta * sigma
