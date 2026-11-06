<<<<<<< SEARCH
"""Structural change: diagonal-phase extraction for initialization + coordinate descent polish
for generic unitaries, plus exact decompositions for known gates."""
import math
import time
import random

import numpy as np
=======
"""Structural change: KAK (Kak) decomposition for 3-qubit unitaries + coordinate descent for 2-qubit.
KAK provides algebraic solution with at most 6 CNOTs for 3-qubit gates."""
import math
import time
import random

import numpy as np
from scipy.linalg import logm, expm
>>>>>>> REPLACE

