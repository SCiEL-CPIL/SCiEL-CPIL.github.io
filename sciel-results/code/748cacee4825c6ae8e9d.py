import time
import random
import math

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    # Precompute conflicts
    # conflicts[x] is a list of masks. Each mask represents 3 other numbers in a 4-AP with x.
    # We use 0-indexed bits for the mask to fit in a standard Python int efficiently, 
    # but map 1..N to bits 0..N-1.
    # Bit i corresponds to number i+1.
    
    conflicts = [[] for _ in range(N+1)]
    
    for a in range(1, N+1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            # AP: a, b, c, e
            
            # For a: others are b, c, e
            m = (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            conflicts[a].append(m)
            
            # For b: others are a, c, e
            m = (1 << (a-1)) | (1 << (c-1)) | (1 << (e-1))
            conflicts[b].append(m)
            
            # For c: others are a, b, e
            m = (1 << (a-1)) | (1 << (b-1)) | (1 << (e-1))
            conflicts[c].append(m)
            
            # For e: others are a, b, c
            m = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            conflicts[e].append(m)

    # Helper to check validity
    def is_valid_add(x, S_bits):
        for m in conflicts[x]:
            if (S_bits & m) == m:
                return False
        return True

    best_S = set()
    best_size = 0
    best_S_bits = 0

    # Strategy: Combine Greedy Initialization with Simulated Annealing
    # We run multiple "attempts". Each attempt consists of:
    # 1. A randomized greedy construction (fast, gets a decent baseline)
    # 2. A local search phase (SA or Hill Climbing) to refine it
    
    # To exploit the "density-incremental" idea from the directive without full complexity:
    # We can try different greedy orders. The directive mentions Salem-Spencer base-3.
    # A simple structural variation is to try "block-based" greedy or just high-variance random greedy.
    # Given the tight time limit, we will stick to efficient random greedy + SA, 
    # but improve the SA move generation and cooling schedule.
    
    # We will allocate time dynamically.
    # First, do a few quick greedy runs to establish a good starting point.
    
    num_greedy_trials = 20
    for i in range(num_greedy_trials):
        if time.time() > deadline - 10.0:
            break
            
        order = list(range(1, N+1))
        random.shuffle(order)
        S = set()
        S_bits = 0
        
        for x in order:
            if is_valid_add(x, S_bits):
                S.add(x)
                S_bits |= (1 << (x-1))
                
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            best_S_bits = S_bits

    # Now, run Simulated Annealing starting from the best greedy solution found.
    # We will run SA for the remaining time.
    
    if time.time() < deadline - 1.0:
        S = best_S
        S_bits = best_S_bits
        size = len(S)
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9995 # Slower cooling
        
        last_improvement = time.time()
        
        while time.time() < deadline:
            # Pick a random move
            x = random.randint(1, N)
            
            if x in S:
                # Remove x
                # Delta size = -1
                # Accept if exp(-(-1)/T) > random -> exp(1/T) > random. Since exp(1/T) > 1, always accept?
                # Standard SA for maximization:
                # If we remove, size decreases by 1. Delta = -1.
                # Accept with prob exp(Delta / T) = exp(-1/T).
                if random.random() < math.exp(-1.0 / T):
                    S.remove(x)
                    S_bits &= ~(1 << (x-1))
                    size -= 1
            else:
                # Try to add x
                if is_valid_add(x, S_bits):
                    S.add(x)
                    S_bits |= (1 << (x-1))
                    size += 1
                    last_improvement = time.time()
                else:
                    # If invalid, we could try a swap: remove one conflicting element?
                    # For simplicity and speed, just reject.
                    pass
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Stagnation handling: Reheat and kick
            if time.time() - last_improvement > 1.5:
                # Remove 3-5 random elements to break local optimum
                k = random.randint(3, 5)
                for _ in range(k):
                    if S:
                        rem = random.choice(list(S))
                        S.remove(rem)
                        S_bits &= ~(1 << (rem-1))
                        size -= 1
                last_improvement = time.time()
                T = 1.0 # Reheat

    return sorted(best_S)

