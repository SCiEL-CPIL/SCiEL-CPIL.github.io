def _numerical_rank_search(T, n, m, p, max_rank, time_budget):
    # T is (n*m, m*p, p*n) sparse tensor with 1 at consistent triples
    import numpy as np
    best = None
    U0 = np.random.randn(n*m, max_rank)
    V0 = np.random.randn(m*p, max_rank)
    W0 = np.random.randn(p*n, max_rank)
    # Simple: use a rank-1 greedy / ALS
    ...
