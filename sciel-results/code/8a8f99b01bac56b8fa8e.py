<<<<<<< SEARCH
"""
Strategy: Multi-start optimization with improved smooth surrogate and Nelder-Mead polish.
The value function is scale-invariant (value = l2/(norm_1*norm_inf)), so only the shape of h matters.
We use a smooth log-sum-exp approximation for max(|g|) during optimization, then polish
with Nelder-Mead on the exact value for final refinement.

Key ideas:
1. Smooth surrogate: replace norm_inf = max(|g|) with a soft-max (log-sum-exp) with high beta.
2. Logarithmic objective for better conditioning of the ratio.
3. Multiple diverse initializations + perturbation of best.
4. Time-budgeted search with safety margin.
5. Final Nelder-Mead polish on exact value for high-precision local refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
=======
"""
Strategy: Multi-start optimization with genetic algorithm over support structure.
The value function is scale-invariant (value = l2/(norm_1*norm_inf)), so only the shape of h matters.
We combine:
1. Genetic algorithm over discrete support patterns (which indices are active) + relative magnitudes
2. Smooth log-sum-exp surrogate for norm_inf during continuous refinement
3. Coordinate descent and Nelder-Mead polish for final local optimization
4. Time-budgeted search with safety margin

Key insight: The non-convexity of the non-negativity constraint and sparsity patterns
that maximize l2/(norm_1*norm_inf) can be better explored by evolving discrete support
structures rather than relying solely on continuous gradient signals.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
>>>>>>> REPLACE

