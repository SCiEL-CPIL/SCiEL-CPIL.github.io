<<<<<<< SEARCH
"""Multi-objective optimizer on RE21 using an Evolutionary Strategy with Hypervolume-based selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
=======
"""Multi-objective optimizer on RE21 using Sobol sampling + Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc
>>>>>>> REPLACE

