<<<<<<< SEARCH
def _de_search(materials, problem, max_total, max_layer, deadline, pop_size=30, max_iter=30):
    """Placeholder for removed DE search."""
    return None, float('inf')
=======
def _de_search(materials, problem, max_total, max_layer, deadline, pop_size=30, max_iter=30):
    """Differential Evolution search over log-scaled thicknesses."""
    if not _HAVE_SCIPY:
        return None, float('inf')
    
    eval_x, grad_x, lo, hi = _make_evaluator(materials, problem, max_total, max_layer)
    
    def objective(x):
        return eval_x(x)
    
    try:
        res = _de(
            objective, list(zip(lo, hi)),
            maxiter=max_iter, popsize=pop_size,
            tol=1e-5, seed=42,
            polish=False,
            init='latinhypercube',
            updating='deferred',
            seed=42,
        )
        t = np.exp(np.clip(res.x, lo, hi))
        total = t.sum()
        if total > max_total:
            t = t * (max_total / total)
        t = np.clip(t, 0.0, max_layer)
        score = _mean_reflectance(materials, list(t), problem)
        return list(t), score
    except Exception:
        return None, float('inf')
>>>>>>> REPLACE

