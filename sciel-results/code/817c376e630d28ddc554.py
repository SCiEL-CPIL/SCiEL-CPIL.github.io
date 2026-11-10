# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    # We formulate this as a Linear Programming problem.
    # We choose a domain D = {1, 2, ..., N}.
    # Variables are v_k for k in 2..N.
    # v_1 is determined by the normalization constraint sum(v_k/k) = 0.
    # Objective: Maximize -sum_{k=2}^N v_k * log(k)/k.
    # Constraints:
    # 1. -10 <= v_k <= 10 for k in 2..N
    # 2. -10 <= v_1 <= 10  =>  -10 <= -sum_{k=2}^N v_k/k <= 10
    # 3. M(x) <= 1 for x in 1..10*N
    #    M(x) = sum_{k=1}^N v_k * floor(x/k)
    #    Substituting v_1: M(x) = sum_{k=2}^N v_k * (floor(x/k) - x/k) = -sum_{k=2}^N v_k * {x/k}
    #    So -sum_{k=2}^N v_k * {x/k} <= 1  =>  sum_{k=2}^N v_k * {x/k} >= -1
    
    N = 50  # Maximum key in domain
    # Variables: v_2, v_3, ..., v_N
    n_vars = N - 1
    
    # Objective: Minimize sum(v_k * log(k)/k)  [since we want to maximize negative]
    c = np.array([np.log(k) / k for k in range(2, N + 1)])
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 1. v_k <= 10  =>  I @ x <= 10
    # 2. -v_k <= 10 =>  -I @ x <= 10
    # 3. -sum(v_k/k) <= 10  =>  -1/k @ x <= 10
    # 4. sum(v_k/k) <= 10   =>  1/k @ x <= 10
    # 5. sum(v_k * {x/k}) >= -1  =>  -sum(v_k * {x/k}) <= 1
    
    A_ub_parts = []
    b_ub_parts = []
    
    # Bounds on v_k
    I = np.eye(n_vars)
    A_ub_parts.append(I)
    b_ub_parts.append(np.full(n_vars, 10.0))
    
    A_ub_parts.append(-I)
    b_ub_parts.append(np.full(n_vars, 10.0))
    
    # Bounds on v_1
    inv_k = np.array([1.0 / k for k in range(2, N + 1)])
    A_ub_parts.append(-inv_k.reshape(1, -1))
    b_ub_parts.append(np.array([10.0]))
    
    A_ub_parts.append(inv_k.reshape(1, -1))
    b_ub_parts.append(np.array([10.0]))
    
    # M(x) constraints
    # For x in 1..10*N:
    # sum_{k=2}^N v_k * {x/k} >= -1
    # -sum_{k=2}^N v_k * {x/k} <= 1
    max_x = 10 * N
    for x in range(1, max_x + 1):
        coeffs = np.array([-(x % k) / k for k in range(2, N + 1)])
        A_ub_parts.append(coeffs.reshape(1, -1))
        b_ub_parts.append(np.array([1.0]))
        
    A_ub = np.vstack(A_ub_parts)
    b_ub = np.concatenate(b_ub_parts)
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(-10, 10)] * n_vars, method='highs')
    
    if res.success:
        v_rest = res.x
        v_1 = -np.sum(v_rest / np.arange(2, N + 1))
        
        # Clip to be safe against floating point errors
        v_1 = np.clip(v_1, -10, 10)
        v_rest = np.clip(v_rest, -10, 10)
        
        # Re-calculate v_1 to ensure sum is exactly 0? 
        # The problem says f(1) is REPLACED by f(1) - S. 
        # So the returned value at index 1 is the final value.
        # We should return the values such that the evaluator's clipping and replacement yields a valid function.
        # However, the evaluator clips first. 
        # "values are clipped to [-10, 10]. Then f(1) is replaced..."
        # If we return v_1 and v_rest, and they are within [-10, 10], clipping does nothing.
        # Then f(1) becomes v_1 - S.
        # Our v_1 was calculated as -S. So new f(1) = -S - S = -2S?
        # NO.
        # Let's re-read carefully.
        # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the clipped values"
        # So if we return $v^{ret}$, it is clipped to $v^{clip}$.
        # $S = \sum v^{clip}_k / k$.
        # $v^{final}_1 = v^{clip}_1 - S$.
        # $v^{final}_k = v^{clip}_k$ for $k \neq 1$.
        # The constraint $M(x)$ is on $v^{final}$.
        # The objective $a$ is on $v^{final}$.
        
        # In my LP, I assumed $v_1$ is the FINAL value?
        # No, I set up the LP such that the variables are the values that satisfy the constraints on the FINAL values.
        # Let $u_k$ be the final values.
        # $\sum u_k/k = 0$.
        # $M(x)$ uses $u_k$.
        # Objective uses $u_k$.
        # $u_k \in [-10, 10]$.
        
        # If I return $u_k$ as `values`, then:
        # 1. Clip: $u_k$ is already in range, so $v^{clip} = u$.
        # 2. Calculate $S = \sum u_k/k = 0$.
        # 3. Replace $f(1)$: $u_1 - 0 = u_1$.
        # So the final values are exactly $u$.
        # So I should return the LP solution directly.
        
        domain = np.arange(1, N + 1, dtype=np.int64)
        values = np.empty(N, dtype=np.float64)
        values[0] = v_1
        values[1:] = v_rest
        
        # Verify M(x) <= 1.0001
        # M(x) = sum u_k floor(x/k)
        valid = True
        for x in range(1, max_x + 1):
            m_val = 0.0
            for k in range(1, N + 1):
                m_val += values[k-1] * (x // k)
            if m_val > 1.0001:
                valid = False
                break
        
        if valid:
            return domain, values
            
    # Fallback to a known good simple solution if LP fails
    # Using the seed or a simple heuristic
    return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

# EVOLVE-BLOCK-END
