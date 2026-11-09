"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization using Thompson Sampling.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure finite inputs and proper types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: Draw a single sample from the GP posterior
    # y_sample = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.standard_normal(mu.shape)
    y_sample = mu + sigma * z
    
    # In Thompson Sampling for minimization, we want to select the point
    # with the lowest sampled value. However, the interface requires returning
    # scores where HIGHER is better.
    # So we invert the sampled values: score = -y_sample
    # To incorporate cost-awareness (since lower cost is better for same quality),
    # we can add a small penalty for high cost, or simply rely on the stochastic nature.
    # Given the cost is 1 + mean(x), and we don't have x, we only have the cost value.
    # A common approach is to score based on the probability of being the best,
    # or simply the negative sampled value.
    
    # Let's use the negative of the sampled value as the primary score.
    # To make it strictly cost-aware in a way that prefers cheaper points when values are close,
    # we can subtract a small multiple of the cost, but this is risky.
    # Instead, let's stick to pure Thompson Sampling logic:
    # The "score" for the next evaluation is the negative of the sampled objective.
    # The caller picks the argmax.
    
    scores = -y_sample
    
    # To ensure we prefer lower cost when sampled values are very close (numerical tie-breaking),
    # we can subtract a tiny epsilon times cost. But this might distort the TS property.
    # Let's try a hybrid: The probability of improvement (POI) is a valid acquisition function too.
    # POI = P(y < f_best) = Phi((f_best - mu) / sigma)
    # This is deterministic. But TS is stochastic.
    
    # Let's stick to TS. The stochasticity naturally explores.
    # However, if we just return -y_sample, we are ignoring cost entirely.
    # The task says "Cost-aware acquisition".
    # If we ignore cost, we might evaluate expensive points unnecessarily.
    # But TS is known to be robust.
    # Let's look at the cost structure: cost = 1 + mean(x).
    # The cost is proportional to the "distance" from the origin (in scaled [0,1] space, it's the average coordinate).
    # Since we don't have coordinates, we can't do spatial reasoning.
    
    # Alternative: Probability of Improvement (POI) divided by cost?
    # Or Expected Improvement (EI) / cost? (Current approach)
    
    # Let's try a different structural change: **Knowledge Gradient (KG) approximation** or **Probability of Improvement (POI)**.
    # POI is computationally cheap and often outperforms EI in early stages.
    # But the directive asks for a MATERIALLY DIFFERENT strategy.
    # Thompson Sampling is a different family (stochastic vs deterministic).
    
    # To incorporate cost into TS:
    # We can sample from the posterior, and then among the points that are "good" (e.g., below f_best),
    # we prefer cheaper ones? No, that's complex.
    
    # Let's try: Score = -y_sample / cost.
    # This is "Sampled Improvement per Cost".
    # This is a stochastic version of EI/cost.
    
    scores = -y_sample / cost
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
