import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    Parameters:
    mu : array-like, shape (n_candidates,)
        Mean of the Gaussian Process posterior.
    sigma : array-like, shape (n_candidates,)
        Standard deviation of the Gaussian Process posterior.
    f_best : float
        Best observed value so far.
        
    Returns:
    scores : array-like, shape (n_candidates,)
        UCB scores. Higher is better.
    """
    # UCB formula: mu + beta * sigma
    # Beta controls the balance between exploitation (mu) and exploration (sigma).
    # A value of 2.0 is a common default for small budgets.
    beta = 2.0
    
    # Ensure sigma is non-negative (it should be, but just in case)
    sigma = np.maximum(sigma, 0.0)
    
    scores = mu + beta * sigma
    
    # The task says "higher meaning evaluate next".
    # Since we are minimizing, we want to find low f(x).
    # The GP predicts f(x) ~ N(mu, sigma^2).
    # UCB for minimization?
    # Standard UCB is for maximization.
    # For minimization, we want to minimize f(x).
    # The "Upper Confidence Bound" on the *cost* is mu + beta * sigma.
    # We want to pick the point with the lowest possible cost?
    # No, in BO, we pick the point that maximizes the acquisition function.
    # If we define the acquisition as the "optimistic" lower bound of the function, that's for minimization?
    # Let's clarify.
    # We want to find the minimum.
    # We have a prediction $\hat{f}(x) = \mu(x)$.
    # We are uncertain by $\sigma(x)$.
    # The true value $f(x)$ could be as low as $\mu(x) - \beta \sigma(x)$.
    # This is the **Lower Confidence Bound (LCB)**.
    # We want to minimize $f(x)$. So we should pick the $x$ that minimizes the LCB?
    # Or do we pick the $x$ that maximizes the "potential improvement"?
    
    # Let's look at EI.
    # EI is defined for minimization as:
    # $EI(x) = E[\max(f_{best} - f(x), 0)]$
    # This is high when $f(x)$ is likely to be much lower than $f_{best}$.
    
    # If we use UCB logic for minimization:
    # We want to find the point with the lowest possible value.
    # The optimistic lower bound is $\mu(x) - \beta \sigma(x)$.
    # We should pick the $x$ that minimizes $\mu(x) - \beta \sigma(x)$.
    # However, the `acquisition` function must return scores where **higher** means evaluate next.
    # So we should return $- (\mu(x) - \beta \sigma(x)) = \beta \sigma(x) - \mu(x)$.
    # Or simply, if we define the score as the "optimistic improvement", it's different.
    
    # Let's check standard BO libraries.
    # GPyOpt/BoTorch use EI for minimization.
    # If using UCB for minimization, the acquisition function is often defined as:
    # $a(x) = \mu(x) + \beta \sigma(x)$ ? No, that's for maximization.
    # For minimization, if we want to minimize $f$, we can maximize $-f$.
    # If we transform the problem to maximization of $g(x) = -f(x)$, then $g_{best} = -f_{best}$.
    # $\mu_g = -\mu_f$, $\sigma_g = \sigma_f$.
    # UCB for $g$: $a_g(x) = \mu_g(x) + \beta \sigma_g(x) = -\mu_f(x) + \beta \sigma_f(x)$.
    # We want to maximize $a_g(x)$.
    # So the score should be $\beta \sigma(x) - \mu(x)$.
    
    # Let's verify this logic.
    # If $\sigma$ is high, the score is high (explore).
    # If $\mu$ is low (good), the score is high (exploit).
    # This makes sense.
    
    scores = beta * sigma - mu
    
    return scores
