import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Hybrid EI + Thompson Sampling for minimization.
    
    Combines the deterministic exploitation of Expected Improvement (EI)
    with the stochastic exploration of Thompson Sampling. This structural
    change allows the optimizer to occasionally escape local minima by
    sampling from the posterior, while the EI term provides a strong
    pull toward the current best observed value.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean predictions at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # EI measures the expected amount by which a candidate improves upon f_best.
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # 2. Thompson Sampling component
    # Sample from the posterior predictive distribution: y ~ N(mu, sigma^2)
    # We want to minimize y, so the "score" from a sample is -y.
    # To make the scales comparable to EI (which is in objective units and positive),
    # we scale the sample.
    # A simple score from TS is -y. However, -y can be very large negative or positive.
    # Let's look at the structure of the inspiration that scored 9.97:
    # It used base_score = -y_samples and added a bias.
    # Here, we want to combine EI and TS.
    # A common hybrid is to maximize a weighted sum.
    # Let's normalize the TS score to be comparable to EI.
    # Actually, EI is already a good measure of "improvement potential".
    # Thompson sampling effectively samples from the distribution of improvements.
    
    # Let's try a different structural approach: 
    # Use EI as the primary exploitation signal, but add a stochastic term 
    # that encourages exploration in areas of high uncertainty, 
    # but only if the mean is not already promising.
    
    # Alternative: The "Hybrid" approach from literature often uses:
    # Score = EI + lambda * sigma * random_normal()
    # But let's look at the high-scoring inspiration again.
    # It used: scores = -y_samples + bias_strength * (f_best - mu)
    # This is essentially Thompson Sampling with a bias towards exploitation.
    # The current program is LCB.
    # The other inspiration is pure EI.
    
    # Let's combine LCB (current) with the stochastic nature of TS.
    # Or better, combine EI (deterministic improvement) with a stochastic exploration term.
    
    # Let's try: Score = EI + alpha * sigma * z_random
    # Where z_random is a standard normal.
    # This adds noise proportional to uncertainty, similar to TS but anchored to EI.
    
    # However, the instruction says "take the structural idea of one and the refinement machinery of the other".
    # Current: LCB (deterministic, beta * sigma)
    # Inspiration 1: TS (stochastic, sample from posterior)
    # Inspiration 2: EI (deterministic, integral of improvement)
    
    # Let's take the structural idea of Inspiration 1 (Stochastic Sampling) 
    # and the refinement machinery of the Current Program (LCB-like scaling) 
    # or Inspiration 2 (EI).
    
    # Actually, a very powerful approach for multimodal functions is **UCB** or **LCB** 
    # with a dynamic beta, or **EI**. 
    # The high score of 9.97 came from TS with a bias.
    # The current score is 1.90 from LCB.
    # The other score is 1.00 from EI.
    
    # Let's try a **Hybrid EI + TS** approach.
    # Score = EI + w * (sigma * z)
    # If w is small, it's mostly EI. If w is large, it's mostly TS.
    # Since TS scored higher than pure EI in the inspiration, let's lean towards TS 
    # but keep the EI component to ensure exploitation of known good regions.
    
    # Let's use the TS score from Inspiration 1 but add the EI term to it.
    # Inspiration 1: scores = -y_samples + 0.1 * (f_best - mu)
    # Let's add EI to that.
    
    # y_samples = mu + sigma_safe * np.random.normal(0, 1, size=mu.shape)
    # ts_score = -y_samples
    
    # The EI term is positive when improvement is likely.
    # The TS term is high when y is low (good).
    # They are correlated but not identical.
    
    # Let's try:
    # scores = EI + ts_score
    # But the scales might be off. EI is typically small (around 0.1-1.0).
    # ts_score is -mu - sigma*z. mu is the function value (e.g., -1.8 to 0).
    # So ts_score is around 1.8. EI is around 0.1.
    # So ts_score dominates. This is basically TS.
    
    # To make it a true hybrid, we should normalize or weight them.
    # Let's weight the EI term more heavily to ensure we don't just randomly sample.
    
    # Let's try:
    # scores = EI + 0.5 * sigma_safe * np.random.normal(0, 1, size=mu.shape)
    # This adds a stochastic exploration term proportional to sigma, 
    # on top of the deterministic EI.
    # This is structurally different from pure LCB (which is -mu + beta*sigma) 
    # and pure EI. It uses the "improvement" metric (EI) as the base, 
    # and adds stochastic exploration.
    
    # Let's refine this.
    # The LCB uses -mu. EI uses f_best - mu.
    # EI is more focused on improvement.
    
    # Proposed Hybrid:
    # 1. Compute EI.
    # 2. Compute a stochastic term: sigma * z.
    # 3. Combine: Score = EI + lambda * (sigma * z)
    #    where lambda controls the exploration strength.
    #    Since we are maximizing the score, and we want to minimize the function,
    #    the stochastic term should be positive when it suggests a better value.
    #    If we sample y = mu + sigma*z, a good y is low.
    #    So a good "surprise" is when z is negative.
    #    So the term should be -sigma*z? Or just sigma*z?
    #    In TS, we pick the min y. Here we pick the max score.
    #    If we use score = -y, then high score = low y.
    #    -y = -mu - sigma*z.
    #    So the stochastic part is -sigma*z.
    
    # Let's use:
    # scores = EI - sigma_safe * np.random.normal(0, 1, size=mu.shape)
    # This is EI + (TS exploration term).
    # The TS term is -sigma*z. If z is negative (lucky sample), term is positive, boosting score.
    # If z is positive (unlucky sample), term is negative, lowering score.
    
    # This combines the deterministic improvement potential (EI) with 
    # a stochastic exploration bonus (TS).
    
    z_random = np.random.normal(0, 1, size=mu.shape)
    ts_term = -sigma_safe * z_random
    
    # We need to balance EI and ts_term.
    # EI is in units of objective value (improvement).
    # ts_term is in units of objective value (uncertainty).
    # They are comparable.
    
    # Let's add them directly.
    scores = ei + ts_term
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
