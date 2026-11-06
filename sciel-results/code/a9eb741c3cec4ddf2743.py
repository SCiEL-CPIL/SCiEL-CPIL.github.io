"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 512

def _can_add(s, x):
    """
    Check if adding x to set s creates a 4-term arithmetic progression.
    Returns True if it is safe to add x.
    """
    # Case 1: x is the 4th term (a, a+d, a+2d, x) => x-3d, x-2d, x-d in s
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if (x - d) in s and (x - 2*d) in s and (x - 3*d) in s:
            return False
            
    # Case 2: x is the 3rd term (a, a+d, x, x+d) => x-2d, x-d in s, x+d in s
    # x-2d >= 1 => d <= (x-1)//2
    # x+d <= N
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        if (x - d) in s and (x - 2*d) in s and (x + d) in s:
            return False
            
    # Case 3: x is the 2nd term (a, x, x+d, x+2d) => x-d in s, x+d in s, x+2d in s
    # x-d >= 1 => d <= x-1
    # x+2d <= N => d <= (N-x)//2
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if (x - d) in s and (x + d) in s and (x + 2*d) in s:
            return False
            
    # Case 4: x is the 1st term (x, x+d, x+2d, x+3d) => x+d, x+2d, x+3d in s
    # x+3d <= N => d <= (N-x)//3
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if (x + d) in s and (x + 2*d) in s and (x + 3*d) in s:
            return False
            
    return True

def construct():
    deadline = time.time() + 55.0  # Safety margin
    random.seed(42) # Deterministic for reproducibility in tests, though random is used
    
    # 1. Initial Greedy Construction
    # Start with a random greedy set to have a diverse starting point
    initial_order = list(range(1, N + 1))
    random.shuffle(initial_order)
    
    current_set = set()
    for x in initial_order:
        if _can_add(current_set, x):
            current_set.add(x)
            
    current_size = len(current_set)
    best_set = set(current_set)
    best_size = current_size
    
    # 2. Simulated Annealing
    # We will perform swaps (remove one, add one) or additions (if possible).
    # Since the greedy set is maximal (no single element can be added without conflict),
    # we primarily rely on swaps to change the structure and potentially allow larger sets later.
    
    # Parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling = 0.999
    
    # Precompute list of all elements for faster iteration
    all_elements = list(range(1, N + 1))
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Generate a random move
        # 80% chance of swap, 20% chance of trying a pure add (though unlikely to succeed if maximal)
        # Actually, if the set is not maximal (we removed an element), adds might be possible.
        # But usually we start maximal. Let's focus on swaps.
        
        if current_set:
            # Pick element to remove
            y = random.choice(list(current_set))
            # Pick element to add
            x = random.choice(all_elements)
            
            if x == y:
                continue
                
            # If x is already in set, this is effectively a "no-op" or bad move logic.
            # We want to swap y for x. So x must not be in set.
            if x in current_set:
                continue

            # Check if removing y and adding x is valid
            # First, check if x can be added to (current_set - {y})
            temp_set = current_set - {y}
            
            if _can_add(temp_set, x):
                new_set = temp_set | {x}
                new_size = len(new_set)
                delta_size = new_size - current_size
                
                # Metropolis criterion
                if delta_size >= 0:
                    # Accept if better or equal (if equal, accept to explore)
                    # If equal, we might want to accept with some probability to avoid getting stuck
                    # But usually delta_size is 0 for swaps.
                    if delta_size > 0 or random.random() < 0.5:
                        current_set = new_set
                        current_size = new_size
                        
                        if current_size > best_size:
                            best_size = current_size
                            best_set = set(current_set)
                else:
                    # delta_size < 0 (worsening move)
                    # Accept with probability exp(delta_size / T)
                    # Since delta_size is integer, min -1
                    prob = math.exp(delta_size / T)
                    if random.random() < prob:
                        current_set = new_set
                        current_size = new_size
            # else: move invalid, reject
            
        # Cooling
        T *= cooling
        
        # Optional: If stuck for a long time, reheat? 
        # For now, keep it simple. The cooling schedule should allow exploration early.
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
