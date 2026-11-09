"""New program: Pack distinct multiples using Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import math

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _cost(xs):
    """Calculate the span of the multiples."""
    return max(xs) - min(xs)

def _is_valid(xs):
    """Check if all multiples are distinct and within bounds."""
    if len(set(xs)) != N:
        return False
    if min(xs) < 1 or max(xs) > MAX_VAL:
        return False
    return True

def _initial_solution():
    """
    Generate an initial valid solution.
    We pick a random center and assign nearest multiples, then resolve collisions.
    """
    # Start with a reasonably tight cluster
    # Let's try to start near the divisors themselves or a random multiple
    # To ensure validity, we can start with a known valid set (e.g., divisors * large factor)
    # and then optimize.
    # A simple valid start: x_i = d_i * (i + 100) 
    # This ensures distinctness easily but might be wide.
    # Better: Randomize k_i around a common base.
    
    base = random.randint(1000, 50000)
    ks = [base // d + random.randint(-5, 5) for d in DIVISORS]
    # Ensure k_i >= 1
    ks = [max(1, k) for k in ks]
    
    xs = [k * d for k, d in zip(ks, DIVISORS)]
    
    # Resolve collisions by shifting
    # Simple loop to ensure distinctness
    used = set()
    for i in range(N):
        while xs[i] in used:
            # Shift to next multiple
            xs[i] += DIVISORS[i]
        used.add(xs[i])
        
    # If any exceed MAX_VAL, pull them down
    for i in range(N):
        if xs[i] > MAX_VAL:
            # Find largest multiple <= MAX_VAL
            max_k = MAX_VAL // DIVISORS[i]
            xs[i] = max_k * DIVISORS[i]
            # This might cause collision again, so we re-run collision resolution
            # For simplicity, we'll just restart if this happens too often
            # But let's just do a quick fix
            used = set()
            for j in range(N):
                while xs[j] in used:
                    xs[j] += DIVISORS[j] if xs[j] < MAX_VAL else -DIVISORS[j]
                    # Ensure positive
                    if xs[j] < 1:
                        xs[j] = DIVISORS[j]
                used.add(xs[j])
                
    # Final check
    if not _is_valid(xs):
        # Fallback to a very safe solution
        xs = [d * 100000 for d in DIVISORS] # These are distinct and < 1e6? 39*100000 = 3.9e6 > 1e6.
        # 1000000 / 39 approx 25641.
        xs = [d * 20000 for d in DIVISORS]
        # 39 * 20000 = 780,000. OK.
        # Are they distinct? Yes, divisors are distinct.
        
    return ks, xs

def _simulated_annealing():
    """
    Run Simulated Annealing to minimize the span.
    """
    ks, xs = _initial_solution()
    best_ks = ks[:]
    best_xs = xs[:]
    best_cost = _cost(xs)
    
    # Annealing parameters
    T = 100.0
    cooling = 0.99
    min_T = 0.1
    iterations = 20000
    
    random.seed(42) # For reproducibility in evaluation
    
    current_cost = best_cost
    
    for _ in range(iterations):
        if T < min_T:
            break
            
        # Pick a random index to change
        i = random.randint(0, N - 1)
        delta_k = random.choice([-1, 1])
        
        new_k_i = ks[i] + delta_k
        if new_k_i < 1:
            continue
            
        new_x_i = new_k_i * DIVISORS[i]
        
        # Check bounds
        if new_x_i > MAX_VAL:
            continue
            
        # Check distinctness
        # We need to ensure new_x_i is not equal to any other x_j (j != i)
        is_distinct = True
        for j in range(N):
            if j != i and xs[j] == new_x_i:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        # Calculate new cost
        # Old min/max
        old_min = min(xs)
        old_max = max(xs)
        
        # New min/max
        # If new_x_i is not the min or max, and doesn't change them, cost change is 0?
        # No, if it replaces the min, new min is min(old_min_second, new_x_i)
        # This is expensive to calculate exactly every time.
        # But N is small (10). Just calculate new xs and cost.
        
        new_xs = xs[:]
        new_xs[i] = new_x_i
        new_cost = _cost(new_xs)
        
        delta_cost = new_cost - current_cost
        
        # Metropolis acceptance
        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
            # Accept
            xs = new_xs
            ks[i] = new_k_i
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_ks = ks[:]
                best_xs = xs[:]
                
        T *= cooling
        
    return best_xs

# Precompute the solution
_solution = None

def _get_solution():
    global _solution
    if _solution is None:
        _solution = _simulated_annealing()
    return _solution

def construct():
    sol = _get_solution()
    # Ensure we return Python ints in a list
    return [int(x) for x in sol]

# EVOLVE-BLOCK-END
