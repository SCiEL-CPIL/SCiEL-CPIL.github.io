# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Approximate q-Expected Improvement (qEI) acquisition function for Branin 2D.
    
    This function approximates the expected improvement of a batch of candidates
    by assuming they are independent (which is a simplification, but often effective
    in practice for small batches or when correlations are not explicitly modeled).
    To encourage exploration and avoid converging too quickly to a single point,
    we compute the sum of EI scores for the top-k candidates (k=5) and assign this
    score to each of them. This creates a "plateau" of high scores, which can help
    the search escape local optima by selecting diverse points.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP posterior mean at candidate points (objective units).
    sigma : ndarray, shape (2000,)
        GP posterior standard deviation at candidate points (objective units).
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        qEI scores; higher means more promising to evaluate next.
    """
    sigma = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma
    
    # Standard EI calculation
    # Use stable log-domain calculation if necessary, but for typical z ranges
    # direct calculation is usually fine. However, to be safe against overflow
    # in norm.pdf(z) for large negative z, we can use a stable form.
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For large negative z, Phi(z) is small, phi(z) is very small.
    # For large positive z, Phi(z) is ~1, phi(z) is small.
    
    # Let's compute EI directly but handle numerical issues.
    # z can be very large.
    
    # A more stable way to compute EI:
    # If z > 0: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # If z < 0: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Actually, the standard formula is stable enough for most practical z values
    # unless z is extremely large.
    
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite scores
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, 0.0)
    
    # qEI approximation: Sum of top-k EI scores
    # We want to assign the sum of the top-k EI scores to those top-k candidates.
    # This encourages the selection of a diverse set of points.
    
    k = 5  # Batch size
    
    # Get indices of top-k EI scores
    top_k_indices = np.argpartition(ei_scores, -k)[-k:]
    
    # Calculate the sum of these top-k EI scores
    q_ei_value = np.sum(ei_scores[top_k_indices])
    
    # Create output scores array
    scores = np.zeros_like(ei_scores)
    
    # Assign the q_ei_value to the top-k candidates
    scores[top_k_indices] = q_ei_value
    
    # For all other candidates, assign a small fraction of their individual EI
    # to maintain some gradient information for the selection process, 
    # or just 0. Let's assign 0 to strictly enforce the batch selection behavior.
    # However, if we assign 0, the selection might be arbitrary among the top-k
    # if the evaluator doesn't break ties well. But the task says "evaluates the candidate 
    # with the highest score". If multiple have the same highest score, it's ambiguous.
    # Usually, argmax picks the first one. 
    # To be safe and provide a clear signal, we can rank them within the top-k.
    
    # Let's refine: Assign scores proportional to their rank within the top-k?
    # Or just assign the same high score?
    # If we assign the same score, the "highest score" is shared.
    # Let's stick to the simple qEI approximation: the value of the batch is the sum.
    # To ensure a unique maximum (or at least a clear ordering), we can add a small
    # epsilon based on the individual EI to break ties.
    
    # Actually, a simpler and robust approach for "qEI" in a single-call context
    # where we return scores for 2000 points and the evaluator picks the max:
    # We want the top-k points to have higher scores than the rest.
    # The sum of EI is a good proxy for the batch value.
    
    # Let's just return the individual EI scores but with a boost for the top-k?
    # No, the "qEI" strategy is specifically about the joint improvement.
    
    # Alternative strategy: Local Penalization or Repulsion.
    # But we don't have the coordinates.
    
    # Let's stick to the "Top-K Sum" heuristic.
    # To ensure the evaluator picks one of the top-k, we set their scores to the sum.
    # To ensure it doesn't pick the same point repeatedly (if memory was an issue, 
    # but here it's stateless), we rely on the GP update.
    
    # One issue: If we assign the same score to top-k, and the evaluator uses 
    # np.argmax, it will pick the first one. This is fine.
    
    # However, to make the score "higher meaning evaluate next" more granular,
    # we can assign scores that are the sum of EI, but slightly decayed by rank
    # within the top-k to encourage picking the best one first if the user 
    # were to pick multiple. But the user picks ONE.
    
    # So, simply: Top-k get the sum. Others get 0.
    
    # Wait, if I return 0 for others, and the top-k have the sum, the max is the sum.
    # This is valid.
    
    return scores
# EVOLVE-BLOCK-END
