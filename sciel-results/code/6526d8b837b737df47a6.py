<<<<<<< SEARCH
"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain.
Uses 24 free parameters (no symmetry constraint), pure-NumPy Nelder-Mead
with precomputed matrix factorization for speed, and multi-restart strategy."""

import time
import numpy as np

_START = time.time()
DEADLINE = _START + 135.0
=======
"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain.
Uses structured coefficient parameterization (symmetric patterns) with
scipy.optimize.minimize (BFGS) on reduced parameter space for efficiency."""

import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
DEADLINE = _START + 135.0
>>>>>>> REPLACE

