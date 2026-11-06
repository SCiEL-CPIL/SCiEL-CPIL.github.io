import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55  # seconds
    best_sum = None
    best_coeffs = None

    # Precompute the denominators
    denoms = [Fraction(1, i) for i in range(1, 31)]

    # Initialize with a random configuration
    def random_config():
        return [random.choice([-1, 0, 1]) for _ in range(30)]

    def compute_sum(coeffs):
        return sum(c * d for c, d in zip(coeffs, denoms))

    def local_search(coeffs, max_iter=10000):
        current_sum = compute_sum(coeffs)
        best_local_sum = current_sum
        best_local_coeffs = coeffs[:]
        
        for _ in range(max_iter):
            improved = False
            for i in range(30):
                for new_c in [-1, 0, 1]:
                    if new_c == coeffs[i]:
                        continue
                    new_coeffs = coeffs[:]
                    new_coeffs[i] = new_c
                    new_sum = compute_sum(new_coeffs)
                    if abs(new_sum) < abs(current_sum):
                        coeffs = new_coeffs
                        current_sum = new_sum
                        if abs(current_sum) < abs(best_local_sum):
                            best_local_sum = current_sum
                            best_local_coeffs = coeffs[:]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        return best_local_coeffs, best_local_sum

    # Run multiple restarts
    num_restarts = 1000
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        coeffs = random_config()
        # Initial local search
        coeffs, current_sum = local_search(coeffs, max_iter=5000)
        
        if best_sum is None or abs(current_sum) < abs(best_sum):
            best_sum = current_sum
            best_coeffs = coeffs[:]

    if best_coeffs is None:
        # Fallback to a simple solution if no better one is found
        best_coeffs = [0] * 28 + [1, -1]
    
    return best_coeffs
