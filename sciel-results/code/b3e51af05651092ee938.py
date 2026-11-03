"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4
NA = N * M      # 12 (U rows)
NB = M * P      # 16 (V rows)
NC = P * N      # 12 (W rows)  # note: W indexed by [k*n+i] per trace convention? 
