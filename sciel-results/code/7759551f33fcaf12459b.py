<<<<<<< SEARCH
"""Dzugutov cluster global minima: icosahedral-shell seeds + L-BFGS-B + kick-and-relax."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient, relax
=======
"""Dzugutov cluster global minima: Parameterized icosahedral model + Differential Evolution + L-BFGS-B."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient, relax
>>>>>>> REPLACE

