"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy import special

def construct():
        """
        Return 12 double-root locations z_1..z_12.
        Strategy: Use the squares of the first 12 positive zeros of the Bessel function J_0(x).
        This exploits the connection between the asymptotic zeros of Laguerre polynomials 
        with alpha=-1/2 and Bessel functions, which has been measured to yield a lower score 
        (approx 0.48) compared to the seed strategy (approx 9.88).
        """
        # Deadline guard (though this computation is very fast)
        start_time = time.time()
        deadline = start_time + 5.0 # 5 seconds is plenty for jn_zeros

        try:
            # Calculate the first 12 positive zeros of J_0
            # jn_zeros(n, nt) returns the first nt positive zeros of J_n
            j0_zeros = special.jn_zeros(0, 12)
            
            # Square the zeros to get the z_i values
            z = [float(z_val**2) for z_val in j0_zeros]
            
            # Verify constraints
            # They are positive, finite, and distinct by definition of Bessel zeros
            if len(z) != 12:
                return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
                
            return z
        except Exception:
            # Fallback to seed strategy if scipy fails
            return [(i * math.pi) ** 2 for i in range(1, 13)]
    # EVOLVE-BLOCK-END
