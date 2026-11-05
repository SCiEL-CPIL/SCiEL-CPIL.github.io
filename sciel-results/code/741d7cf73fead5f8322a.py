"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math

# EVOLVE-BLOCK-START

N = 512

def _precompute_triplets():
    """
    Precompute for each x in [1, N] a list of bitmasks.
    Each bitmask represents a set of 3 other elements {a, b, c} such that 
    {a, b, c, x} forms a 4-term arithmetic progression.
    """
    triplets = [[] for _ in range(N + 1)]
    
    # Generate all 4-term APs in [1, N]
    # AP: p, p+d, p+2d, p+3d
    for p in range(1, N + 1):
        for d in range(1, (N - p) // 3 + 1):
            t1 = p
            t2 = p + d
            t3 = p + 2 * d
            t4 = p + 3 * d
            if t4 > N:
                continue
            
            # For each element in this AP, the other 3 form a "forbidden triplet"
            # If we add t1, we must not have {t2, t3, t4} all in S.
            mask = (1 << t2) | (1 << t3) | (1 << t4)
            triplets[t1].append(mask)
            
            mask = (1 << t1) | (1 << t3) | (1 << t4)
            triplets[t2].append(mask)
            
            mask = (1 << t1) | (1 << t2) | (1 << t4)
            triplets[t3].append(mask)
            
            mask = (1 << t1) | (1 << t2) | (1 << t3)
            triplets[t4].append(mask)
            
    return triplets

def construct():
    random.seed(42)
    triplets = _precompute_triplets()
    
    # S is a bitmask of the current set
    S = 0
    # List of elements in S for fast random access
    S_list = []
    
    def can_add(x, S_mask):
        """Check if x can be added to S_mask without forming a 4-AP."""
        for mask in triplets[x]:
            if (S_mask & mask) == mask:
                return False
        return True

    # 1. Greedy Initialization
    # Try multiple random orders to find a good starting point.
    best_S = 0
    best_size = 0
    best_list = []
    
    for _ in range(20):
        current_S = 0
        current_list = []
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            if can_add(x, current_S):
                current_S |= (1 << x)
                current_list.append(x)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_S = current_S
            best_list = current_list[:]
            
    S = best_S
    S_list = best_list[:]
    
    # 2. Simulated Annealing
    # We use a time-based limit to ensure we finish within 60s.
    # Estimate: ~200-300 ops per iteration in Python for can_add.
    # 100,000 iterations ~ 0.5s - 1s. 
    # We can afford ~500,000 - 1,000,000 iterations.
    
    T_start = 5.0
    T_end = 0.05
    # Let's run for a fixed number of iterations, say 300,000, and check time.
    # Or just use a large number and break on time.
    max_time = 55.0  # seconds
    start_time = time.time()
    
    current_T = T_start
    # Cooling rate for 300k iterations
    total_iters = 300000
    cooling_rate = (T_end / T_start) ** (1.0 / total_iters)
    
    all_elements = list(range(1, N + 1))
    
    # Precompute random choices for speed? No, random.choice is fine.
    
    for i in range(total_iters):
        # Time check
        if time.time() - start_time > max_time:
            break
            
        # Generate a move
        # Probabilities:
        # 0.4 Add
        # 0.4 Swap
        # 0.2 Remove
        
        r = random.random()
        
        if r < 0.4:
            # Add move
            x = random.choice(all_elements)
            if not (S & (1 << x)):
                if can_add(x, S):
                    # Always accept improvement (Size +1)
                    S |= (1 << x)
                    S_list.append(x)
                    if len(S_list) > best_size:
                        best_size = len(S_list)
                        best_S = S
                        best_list = S_list[:]
                        # Optional: Reheat if we found a new best?
                        # current_T = max(current_T, 1.0)
                        
        elif r < 0.8:
            # Swap move: Remove x, Add y
            if S_list:
                x = random.choice(S_list)
                y = random.choice(all_elements)
                if x != y and not (S & (1 << y)):
                    S_new = S & ~(1 << x)
                    if can_add(y, S_new):
                        # Size remains same (Delta E = 0)
                        # Accept with probability 1 (or exp(0)=1)
                        # This allows random walk on the manifold of size-K sets.
                        S = S_new | (1 << y)
                        # Update list: replace x with y
                        # S_list.index(x) is O(N). 
                        # To optimize, we could use a dict for index, but N is small.
                        idx = S_list.index(x)
                        S_list[idx] = y
                        
        else:
            # Remove move: Remove x
            if S_list:
                x = random.choice(S_list)
                # Delta E = 1 (Size -1)
                # Prob = exp(-1/T)
                # If T is large, prob is high. If T is small, prob is low.
                if random.random() < math.exp(-1.0 / current_T):
                    S &= ~(1 << x)
                    S_list.remove(x)
                    # Size decreased. We don't update best.
                    
        current_T *= cooling_rate
        if current_T < T_end:
            current_T = T_end

    return sorted(best_list)

# EVOLVE-BLOCK-END
