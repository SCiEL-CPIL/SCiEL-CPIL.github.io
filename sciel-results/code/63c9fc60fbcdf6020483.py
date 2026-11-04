<<<<<<< SEARCH
"""Rewritten QAP solver: continuous relaxation + Hungarian + local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization for a strong starting point.
    2. Hungarian-based linearization refinement (multiple rounds).
    3. Fast local search with O(n) incremental updates per swap.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    np.random.seed(42)
=======
"""Rewritten QAP solver: spectral init + ILS local search + Hungarian refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Multiple spectral/rank initializations.
    2. Fast local search with correct O(n) delta computation.
    3. Hungarian-based linearization refinement.
    4. Iterated Local Search (ILS) for deeper exploration.
    5. Final high-precision polish pass.
    """
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    np.random.seed(42)
>>>>>>> REPLACE

