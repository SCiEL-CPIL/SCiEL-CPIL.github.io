import numpy as np
import random
import time

# EVOLVE-BLOCK-START
def construct():
    N = 1000
    MAX_SUM = 2000
    
    # Initialize with a random set or greedy
    x = np.zeros(N, dtype=int)
    # Start with a small random set to have some structure
    # Or start empty
    indices = np.random.choice(N, size=200, replace=False)
    x[indices] = 1
    
    def compute_conv(x):
        return np.convolve(x, x, mode='full')
    
    def compute_D(x):
        D = np.zeros(MAX_SUM + 1)
        # S ranges from 2 to 2000
        for S in range(2, MAX_SUM + 1):
            if S % 2 == 0:
                a = S // 2
                if 1 <= a <= N:
                    D[S] = x[a - 1]
        return D
    
    conv = compute_conv(x)
    D = compute_D(x)
    
    def get_energy(x, conv, D):
        # N_S = (conv[S-2] + D[S]) / 2
        # S ranges 2..2000, so conv index 0..1998
        counts = (conv[:MAX_SUM-1] + D[2:MAX_SUM+1]) / 2.0
        violations = np.sum(np.maximum(0, counts - 2))
        size = np.sum(x)
        # We want to maximize size, penalize violations
        # Energy to minimize: violations - lambda * size
        return violations - 0.1 * size
        
    current_energy = get_energy(x, conv, D)
    best_x = x.copy()
    best_energy = current_energy
    
    T_start = 1.0
    T_end = 0.001
    steps = 50000
    cool_rate = (T_end / T_start) ** (1.0 / steps)
    
    for step in range(steps):
        T = T_start * (cool_rate ** step)
        
        # Pick a random index to flip
        i = np.random.randint(0, N)
        
        # Calculate delta energy
        # We need to update conv and D for the flip
        # If x[i] is 0 -> 1 (add)
        # If x[i] is 1 -> 0 (remove)
        
        old_val = x[i]
        new_val = 1 - old_val
        
        # Update conv
        # conv_new = conv_old + delta
        # delta = x_old * e_i + e_i * x_old + e_i * e_i (if adding)
        # Actually, simpler: compute new conv? No, too slow.
        # Update:
        # If adding i (0->1):
        #   conv += x (shifted by i) + x (shifted by -i) + impulse at 2i
        #   Wait, convolution index k corresponds to sum k+2.
        #   e_i is at index i.
        #   x * e_i: term x[j] * e_i[k-j] = x[j] if k-j=i => j=k-i.
        #   So (x * e_i)[k] = x[k-i] if k>=i else 0.
        #   This is x shifted right by i.
        
        delta_conv = np.zeros_like(conv)
        if old_val == 0:
            # Add i
            # x * e_i
            if i < len(x):
                delta_conv[i:] += x[:len(x)-i]
            # e_i * x
            if i < len(x):
                delta_conv[:len(x)-i] += x[i:]
            # e_i * e_i
            if 2*i < len(conv):
                delta_conv[2*i] += 1
        else:
            # Remove i
            # Subtract the same terms
            if i < len(x):
                delta_conv[i:] -= x[:len(x)-i]
            if i < len(x):
                delta_conv[:len(x)-i] -= x[i:]
            if 2*i < len(conv):
                delta_conv[2*i] -= 1
                
        new_conv = conv + delta_conv
        
        # Update D
        delta_D = np.zeros(MAX_SUM + 1)
        if old_val == 0:
            # Add i, so x[i] becomes 1
            # D[S] changes if S = 2*(i+1)
            S_diag = 2 * (i + 1)
            if S_diag <= MAX_SUM:
                delta_D[S_diag] = 1
        else:
            # Remove i
            S_diag = 2 * (i + 1)
            if S_diag <= MAX_SUM:
                delta_D[S_diag] = -1
                
        new_D = D + delta_D
        
        new_energy = get_energy(x, new_conv, new_D) # Note: x is not updated yet, but get_energy uses x for size?
        # get_energy uses np.sum(x). If we haven't updated x, size is wrong.
        # Let's adjust get_energy to take size as param or update x first?
        # Better: calculate delta_energy directly?
        # Or just update x, conv, D, calc energy, and rollback if rejected.
        
        # Rollback strategy:
        x[i] = new_val
        conv = new_conv
        D = new_D
        
        # Recalc energy with updated x
        # To avoid full recompute of size, we can track size.
        # But get_energy is cheap if we pass size?
        # Let's just use np.sum(x) for simplicity, it's O(N)=1000.
        # 50000 * 1000 = 50M ops. OK.
        
        new_energy_val = get_energy(x, conv, D)
        
        delta_E = new_energy_val - current_energy
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            current_energy = new_energy_val
            if new_energy_val < best_energy:
                best_energy = new_energy_val
                best_x = x.copy()
        else:
            # Rollback
            x[i] = old_val
            conv = conv - delta_conv
            D = D - delta_D
            # current_energy unchanged
            
    # Convert best_x to list of integers
    result = [i + 1 for i in range(N) if best_x[i] == 1]
    return result
# EVOLVE-BLOCK-END
