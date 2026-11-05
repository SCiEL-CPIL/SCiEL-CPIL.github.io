import numpy as np
import time
import random

def construct():
    N = 51975
    # Find divisors of 51975
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    if N % 1 != 0: # N is always divisible by 1, but we need m > 1
        pass
    # Ensure N itself is included if it's a divisor (it is)
    divisors.append(N)
    # Remove duplicates and sort
    divisors = sorted(list(set(divisors)))
    # Filter m > 1
    divisors = [d for d in divisors if d > 1]
    
    num_mods = len(divisors)
    
    # Initialize residues
    residues = np.array([random.randint(0, d - 1) for d in divisors])
    
    # Initialize counts array
    counts = np.zeros(N, dtype=np.int32)
    
    # Helper to add a class (m, r) to counts
    def add_class(m, r, counts, delta=1):
        # Indices: r, r+m, r+2m, ... < N
        # np.arange(r, N, m)
        counts[np.arange(r, N, m)] += delta
        return counts

    # Initial setup
    for i, m in enumerate(divisors):
        add_class(m, residues[i], counts, 1)
        
    current_score = np.sum(counts == 0)
    best_score = current_score
    best_residues = residues.copy()
    
    start_time = time.time()
    timeout = 115.0
    
    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.1
    
    # Precompute index arrays for each divisor? 
    # That would use a lot of memory: sum(N/m for m in divisors).
    # N=51975. Sum of N/d is roughly N * log(log(N)) which is manageable?
    # Let's just compute on the fly or use a list of lists if memory allows.
    # N=51975 is small enough that np.arange is fast.
    
    while time.time() - start_time < timeout and T > min_T:
        # Pick a random modulus index
        idx = random.randint(0, num_mods - 1)
        m = divisors[idx]
        old_r = residues[idx]
        
        # Pick a new residue
        new_r = random.randint(0, m - 1)
        
        if new_r == old_r:
            continue
            
        # Calculate delta
        # S_old: indices covered by (m, old_r)
        # S_new: indices covered by (m, new_r)
        
        # Get indices
        # Note: This creates temporary arrays. 
        # Optimization: We can avoid creating full index arrays if we just count.
        # But we need to know counts at those specific indices.
        
        # Method: 
        # 1. Identify indices in S_old where counts == 1. These will become uncovered (score +1)
        # 2. Identify indices in S_new where counts == 0. These will become covered (score -1)
        
        # To do this efficiently with NumPy:
        # We can create boolean masks, but that is O(N) per step.
        # With ~100-200 divisors and 51975 elements, O(N) is ~50k ops.
        # In 120s, Python/NumPy can do millions of ops. 
        # Let's try the mask approach first.
        
        # Mask for old_r
        # Instead of arange, we can use modulo? No, arange is better for slicing.
        
        # Let's stick to the delta calculation:
        # delta = count(S_old, counts==1) - count(S_new, counts==0)
        
        # Generate indices
        # This is the bottleneck. 
        # Alternative: Iterate over the smaller of the two sets? No, they are same size N/m.
        
        # Let's try to optimize by precomputing the index arrays for each divisor.
        # Memory: Sum(N/m) for all divisors.
        # 51975 = 3^3 * 5^2 * 7 * 11.
        # Divisors are many. 
        # Let's just use np.arange. It's C-optimized.
        
        idx_old = np.arange(old_r, N, m)
        idx_new = np.arange(new_r, N, m)
        
        # Count how many in idx_old have count == 1
        # We need counts at these indices.
        # counts[idx_old] gives the values.
        # c_old = counts[idx_old]
        # num_old_uncover = np.sum(c_old == 1)
        
        # c_new = counts[idx_new]
        # num_new_cover = np.sum(c_new == 0)
        
        # delta = num_old_uncover - num_new_cover
        
        # If we accept, we update counts:
        # counts[idx_old] -= 1
        # counts[idx_new] += 1
        
        c_old_vals = counts[idx_old]
        c_new_vals = counts[idx_new]
        
        num_old_uncover = np.sum(c_old_vals == 1)
        num_new_cover = np.sum(c_new_vals == 0)
        
        delta = num_old_uncover - num_new_cover
        
        # Acceptance probability
        if delta < 0:
            accept = True
        else:
            # P = exp(-delta / T)
            # If delta is large, P is small.
            # Use log to avoid underflow?
            # If delta/T > 50, exp is ~0.
            if delta > 50 * T:
                accept = False
            else:
                accept = (random.random() < np.exp(-delta / T))
                
        if accept:
            # Update counts
            counts[idx_old] -= 1
            counts[idx_new] += 1
            residues[idx] = new_r
            current_score += delta
            
            if current_score < best_score:
                best_score = current_score
                best_residues = residues.copy()
                
        # Cool down
        T *= cooling_rate
        
    # Convert best_residues to list of [m, r]
    classes = [[divisors[i], int(best_residues[i])] for i in range(num_mods)]
    return classes
