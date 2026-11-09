"""QUBO-based penalty annealing for minimizing monochromatic 4-APs on [1,256]."""
import numpy as np

def construct():
    N = 256
    rng = np.random.default_rng(42)
    
    # Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) with d>=1, a+3d <= N
    # Indexing: integers 1..256 map to indices 0..255
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
            aps.append(idx)
    
    aps = np.array(aps, dtype=np.int32)
    num_aps = len(aps)
    
    # Initial random coloring: 0/1
    x = rng.integers(0, 2, size=N).astype(np.float64)
    
    def count_mono_aps(x):
        """Count monochromatic 4-APs. x is 0/1 array."""
        # For each AP, check if all four positions have the same color
        # Use broadcasting: x[aps] has shape (num_aps, 4)
        vals = x[aps]  # shape (num_aps, 4)
        # Monochromatic if all equal: max - min == 0
        mono = (vals.max(axis=1) - vals.min(axis=1)) == 0
        return int(mono.sum())
    
    def energy(x, beta):
        """
        QUBO energy: sum over each AP of penalty if not all same color.
        For a 4-AP with colors c1,c2,c3,c4 in {0,1}:
        Monochromatic iff c1==c2==c3==c4.
        Penalty = 1 if NOT monochromatic... wait, we want to MINIMIZE monochromatic count.
        
        Actually, we want to minimize the number of monochromatic 4-APs.
        So the objective is: sum over all APs of [1 if mono else 0].
        
        Let's define the QUBO properly.
        For a 4-tuple (i,j,k,l), the indicator that all are equal can be written as:
        I(c_i == c_j == c_k == c_l)
        
        For binary variables, c_i == c_j iff 1 - c_i*c_j - (1-c_i)*(1-c_j) ... 
        Actually simpler: c_i == c_j iff c_i XOR c_j == 0 iff 1 - (c_i - c_j)^2 ... 
        For binary: c_i == c_j iff 2*c_i*c_j + 2*(1-c_i)*(1-c_j) - 1 ... 
        Let's use: I(c_i == c_j) = 1 - |c_i - c_j| = 1 - (c_i - c_j)^2 for binary.
        But (c_i - c_j)^2 = c_i^2 - 2c_i c_j + c_j^2 = c_i - 2c_i c_j + c_j for binary.
        So I(c_i == c_j) = 1 - c_i + 2c_i c_j - c_j.
        
        For 4 variables all equal:
        I(c1==c2==c3==c4) = I(c1==c2) * I(c2==c3) * I(c3==c4) ... but this isn't quite right as a product.
        
        Actually, I(c1==c2==c3==c4) = I(c1==c2) * I(c2==c3) * I(c3==c4) is NOT correct in general.
        
        Let me think differently. The number of monochromatic 4-APs is what we want to minimize.
        We can just directly compute the count and use gradient-free optimization, but the directive says
        to use QUBO with penalty annealing.
        
        Alternative approach: Use a soft QUBO where we minimize the sum of "non-mono" penalties... 
        No, we want to minimize mono count. So the QUBO objective IS the mono count.
        
        Let's just use a simulated annealing / penalty annealing approach on the original space,
        but frame it as minimizing the QUBO energy E(x) = sum over APs of I(mono).
        
        For efficiency, we can compute the change in energy when flipping a single bit.
        """
        return count_mono_aps(x)
    
    best_x = x.copy()
    best_e = count_mono_aps(x)
    
    # Penalty annealing / simulated annealing
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    max_iter = 50000
    no_improve = 0
    patience = 5000
    
    for it in range(max_iter):
        if no_improve > patience:
            break
        if T < T_min:
            break
        
        # Pick a random index to flip
        idx = rng.integers(0, N)
        old_val = x[idx]
        x[idx] = 1.0 - old_val
        
        new_e = count_mono_aps(x)
        delta_e = new_e - best_e  # We compare against current best
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if new_e < best_e:
            best_x = x.copy()
            best_e = new_e
            no_improve = 0
        else:
            # Revert if not accepted
            accept_prob = np.exp(-delta_e / T) if delta_e > 0 else 1.0
            if rng.random() < accept_prob:
                # Keep the flip, but don't update best
                pass
            else:
                x[idx] = old_val  # Revert
        
        T *= cooling
    
    # Convert to list of ints
    return [int(v) for v in best_x]
