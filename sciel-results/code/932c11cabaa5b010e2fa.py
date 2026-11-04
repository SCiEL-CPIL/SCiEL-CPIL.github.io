"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct the sequence of step heights.
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    """
    # We search for a binary sequence x in {-1, 1}^N with sum 0.
    # h = (x + 1) / 2.
    
    # Initialize random sequence with N/2 ones and N/2 minus ones
    half = N // 2
    x = np.zeros(N)
    x[:half] = 1.0
    x[half:] = -1.0
    np.random.shuffle(x)
    
    # Compute initial autocorrelation A_k = sum_i x_i x_{i+k}
    # We care about k=1..N-1
    # A[0] = N
    
    def compute_autocorr(x):
        A = np.zeros(N)
        for k in range(1, N):
            A[k] = np.dot(x, np.roll(x, -k))
        return A
    
    A = compute_autocorr(x)
    # Cost is max_{k>=1} (-A_k) = -min_{k>=1} A_k
    current_cost = -np.min(A[1:])
    
    best_x = x.copy()
    best_A = A.copy()
    best_cost = current_cost
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    steps = 50000
    cooling_rate = (T_min / T) ** (1.0 / steps)
    
    # Precompute rolls for efficiency? 
    # Actually, np.roll is fast enough for N=100.
    
    for s in range(steps):
        # Pick two random indices i, j
        i = np.random.randint(N)
        j = np.random.randint(N)
        
        # We must swap a 1 and a -1 to keep sum 0
        if x[i] == x[j]:
            continue
            
        # Calculate change in A
        # Delta A_k = -2 * (x_i * x_{i+k} + x_j * x_{j+k})
        # Note: x_i and x_j are scalars (1 or -1)
        
        # We need to update A[1:]
        # D_i[k] = x_i * x[(i+k)%N]
        # This is x_i * np.roll(x, -i)[k]
        
        # Efficient update:
        # roll_x_i = np.roll(x, -i)
        # roll_x_j = np.roll(x, -j)
        # delta_A = -2.0 * (x[i] * roll_x_i + x[j] * roll_x_j)
        # A_new = A + delta_A
        
        # However, we only need to check if the cost improves.
        # Cost is -min(A[1:]).
        # Let's just compute the new A and check min.
        
        # To save time, we can check if the minimum is likely to improve.
        # But N=100 is small.
        
        roll_i = np.roll(x, -i)
        roll_j = np.roll(x, -j)
        
        delta_A = -2.0 * (x[i] * roll_i + x[j] * roll_j)
        
        # We don't need A[0], but it changes too.
        # A[0] = sum x^2 = N. It stays N.
        # delta_A[0] should be 0?
        # x_i x_i + x_j x_j = 1+1=2. -2*(2) = -4?
        # Wait.
        # A_0 = sum x_m^2.
        # x'_i^2 = x_i^2 = 1.
        # So A_0 is invariant.
        # My formula: delta_A_0 = -2(x_i x_i + x_j x_j) = -2(1+1) = -4.
        # This is wrong.
        # Let's re-derive.
        # A_k = sum_m x_m x_{m+k}.
        # A'_k = sum_m x'_m x'_{m+k}.
        # Only m=i and m=j change.
        # Term m=i: x'_i x'_{i+k} - x_i x_{i+k} = (-x_i) x_{i+k} - x_i x_{i+k} = -2 x_i x_{i+k}.
        # Term m=j: x'_j x'_{j+k} - x_j x_{j+k} = (-x_j) x_{j+k} - x_j x_{j+k} = -2 x_j x_{j+k}.
        # Are there cross terms?
        # If i+k = j, then x'_{i+k} = x'_j = -x_j.
        # Term m=i: (-x_i)(-x_j) - x_i x_j = 2 x_i x_j - x_i x_j = x_i x_j.
        # My formula gave -2 x_i x_j.
        # So the formula is incorrect when indices overlap.
        
        # Given N=100, overlapping indices are rare but possible.
        # To be safe, just compute full A for the new x?
        # That is O(N^2) = 10,000 ops.
        # 50,000 steps * 10,000 = 5 * 10^8 ops.
        # In Python, this might be 5-10 seconds. Acceptable.
        
        # Let's do the safe O(N^2) update but optimized with NumPy dot products.
        # Actually, we can just update x, compute new A, and compare.
        
        x[i], x[j] = x[j], x[i]
        
        # Compute new A
        # A_new[k] = dot(x, roll(x, -k))
        # We can do this in a loop or vectorize?
        # Vectorizing dot product for all k is not directly possible without matrix mult.
        # Matrix mult: X @ X.T? No, circular.
        # Just loop k=1..N-1.
        
        A_new = np.zeros(N)
        for k in range(1, N):
            A_new[k] = np.dot(x, np.roll(x, -k))
            
        new_cost = -np.min(A_new[1:])
        
        # Acceptance criterion
        if new_cost < current_cost:
            current_cost = new_cost
            A = A_new
            if current_cost < best_cost:
                best_cost = current_cost
                best_x = x.copy()
                best_A = A_new
        else:
            # Reject
            if np.random.rand() < np.exp((current_cost - new_cost) / T):
                current_cost = new_cost
                A = A_new
            else:
                # Revert
                x[i], x[j] = x[j], x[i]
                
        T *= cooling_rate
        
    # Convert best_x to h
    h = (best_x + 1.0) / 2.0
    
    return h
# EVOLVE-BLOCK-END
