<<<<<<< SEARCH
"""Analytic uncertainty bound with vectorized gradient + multi-start optimization."""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200
=======
"""Analytic uncertainty bound with differential evolution + analytic gradient polish."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200
>>>>>>> REPLACE

