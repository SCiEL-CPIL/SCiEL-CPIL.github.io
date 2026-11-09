<<<<<<< SEARCH
"""ODE solver: Convection-diffusion 100-point.
Strategy: 
1. Primary: DOP853 (8th-order adaptive RK) with tight tolerances.
2. Fallback 1: DOP853 with slightly looser tolerances.
3. Fallback 2: Radau (implicit, stiff-capable) if explicit methods fail or are too slow.
4. Fallback 3: Fixed-step RK4 as a last resort.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
=======
"""ODE solver: Convection-diffusion 100-point.
Strategy: 
1. Primary: LTI solver exploiting known convection-diffusion structure (0 f-calls).
2. Fallback 1: DOP853 (8th-order adaptive RK) with tight tolerances.
3. Fallback 2: Radau (implicit, stiff-capable) if explicit methods fail.
4. Fallback 3: Fixed-step RK4 as a last resort.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
>>>>>>> REPLACE

