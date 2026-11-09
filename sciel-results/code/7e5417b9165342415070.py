import numpy as np
def acquisition(mu, sigma, f_best):
    # Calculate a proxy for the "phase" of optimization
    # If f_best is low relative to the mean, we should exploit more (lower beta)
    # If f_best is high relative to the mean, we should explore more (higher beta)
    
    # Simple heuristic:
    # Median mu
    med_mu = np.median(mu)
    
    # If f_best is significantly better than the typical prediction, we are converging.
    # Difference = med_mu - f_best.
    # If this difference is small, f_best is close to the predicted means -> Exploit.
    # If this difference is large, f_best is much better than predicted means -> Explore? 
    # Actually, if f_best is very good, we want to find something even better, which might require exploration if the current model is uncertain.
    
    # Let's stick to a fixed beta for simplicity and robustness, as dynamic ones can be tricky without step counts.
    # Beta = 2.5 is a reasonable middle ground.
    
    beta = 2.5
    scores = -mu + beta * sigma
    return scores
