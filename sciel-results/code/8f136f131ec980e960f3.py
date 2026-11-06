import numpy as np

def construct():
    N = 24
    # Initial guess: Rudin-Shapiro like
    # Generate RS(32) and take first 24? Or just random?
    # RS is a good start.
    def rs_sequence(n):
        # n must be power of 2
        if n == 1:
            return [1]
        half = rs_sequence(n // 2)
        return half + [x if i % 2 == 0 else -x for i, x in enumerate(half)]
    
    # Get RS(32) and truncate to 24
    rs32 = rs_sequence(32)
    c = np.array(rs32[:N], dtype=float)
    
    # Define evaluation function
    M = 20000 # Dense sampling
    z = np.exp(2j * np.pi * np.arange(M) / M)
    
    def eval_sup(c):
        p = np.zeros(M, dtype=complex)
        for k in range(N):
            p = c[k] + z * p
        return np.max(np.abs(p))
    
    current_score = eval_sup(c)
    
    # Hill Climbing
    improved = True
    while improved:
        improved = False
        for i in range(N):
            # Flip bit i
            c[i] *= -1
            new_score = eval_sup(c)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                # Revert
                c[i] *= -1
    
    # If we have time, do a few random restarts or simulated annealing
    # But for N=24, hill climbing from RS is often enough to get to the global optimum
    # or very close.
    
    # Convert back to int list
    return [1 if x > 0 else -1 for x in c]
