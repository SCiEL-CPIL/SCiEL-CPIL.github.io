"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a direct search with trial division for totient calculation.
    Includes an explicit deadline guard to ensure termination within time limits.
    """
    start_time = time.time()
    # Set a deadline with a safety margin (e.g., 50s out of 60s limit)
    deadline = start_time + 50.0
    
    def phi(n):
        """Compute Euler's totient function for n using trial division."""
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    # Search for the smallest n
    for n in range(1, 1000001):
        # Check deadline
        if time.time() > deadline:
            # If we reach here, we haven't found a solution yet.
            # However, the solution is small (143), so this should not happen.
            # Return -1 as a fallback for invalid/timeout, though it should score 0.
            # To be safer, we could return a known valid large number if we had one,
            # but -1 indicates failure.
            return -1
            
        if phi(n) % 120 == 0:
            return n
            
    return -1 # Should not reach here if search space is sufficient
# EVOLVE-BLOCK-END
