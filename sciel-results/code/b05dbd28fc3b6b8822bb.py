# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Structural change: Use a constructive arithmetic approach based on the Möbius function
    with a targeted "greedy projection" onto the constraint set, rather than solving a massive LP.
    
    The objective is to maximize a = -sum f(k) log(k) / k subject to:
    1. sum f(k)/k = 0
    2. M(x) = sum f(k) floor(x/k) <= 1 for all x in [1, 10K]
    
    Strategy:
    1. Initialize f(k) with the Möbius function mu(k), which naturally satisfies sum mu(k) floor(x/k) = 1 for x>=1 (the identity sum_{d|x} mu(d) = 0 for x>1, 1 for x=1).
       Wait, M(x) = sum_{k=1}^K f(k) floor(x/k).
       If f(k) = mu(k), then M(x) = sum_{k=1}^K mu(k) floor(x/k).
       We know that sum_{d|n} mu(d) = [n=1].
       Also, M(x) = sum_{n=1}^x sum_{d|n} mu(d) = sum_{n=1}^x [n=1] = 1 for x >= 1.
       So M(x) = 1 for all x >= 1. This satisfies M(x) <= 1.
       
       Now, what about the constraint sum f(k)/k = 0?
       sum mu(k)/k is related to the Mertens function or similar, but it's not necessarily 0.
       Let S = sum mu(k)/k.
       The evaluator subtracts S from f(1).
       So f_eval(1) = mu(1) - S = 1 - S.
       Then sum f_eval(k)/k = 0 is satisfied.
       
       Now we need to check if M(x) <= 1 still holds for the *modified* function f_eval.
       M_eval(x) = sum_{k=1}^K f_eval(k) floor(x/k)
                = (mu(1) - S) * x + sum_{k=2}^K mu(k) floor(x/k)
                = (1 - S) * x + (M_mu(x) - mu(1)*floor(x/1))
                = (1 - S) * x + (1 - 1*x)   [since M_mu(x)=1 for x>=1, and mu(1)=1, floor(x/1)=x]
                = 1 - S*x + 1 - x
                = 2 - x(1+S)
       
       For x=1: M_eval(1) = 2 - (1+S) = 1 - S.
       For x > 1: M_eval(x) = 2 - x(1+S).
       
       If S is small, 1+S is close to 1. Then for x=2, M_eval(2) = 2 - 2(1+S) = -2S.
       If S > 0, this is negative, so <= 1.
       If S < 0, this is positive.
       
       Let's estimate S = sum_{k=1}^K mu(k)/k.
       For K=1000, S is roughly around 0.6?
       Actually, sum mu(k)/k converges to 0? No, 1/zeta(2)? No.
       sum mu(k)/k^s = 1/zeta(s). For s=1, it diverges? No, 1/zeta(1) is 0?
       Wait, 1/zeta(s) = sum mu(n)/n^s for Re(s)>1. At s=1, zeta(s) has a pole, so 1/zeta(s) -> 0.
       So S should approach 0 as K -> infinity.
       For finite K, S is some value.
       
       If S is small, M_eval(x) for large x will be 2 - x(1+S) which becomes very negative.
       The maximum of M_eval(x) is likely at x=1, which is 1-S.
       If S > 0, 1-S < 1. So M_eval(1) < 1.
       If S < 0, 1-S > 1. Then M_eval(1) > 1, which is infeasible.
       
       Let's check the sign of S for small K.
       K=1: mu(1)=1, S=1. 1-S=0 <= 1. OK.
       K=2: mu(1)=1, mu(2)=-1. S = 1 - 1/2 = 0.5. 1-S=0.5.
       K=3: mu(3)=-1. S = 0.5 - 1/3 = 1/6 > 0.
       K=4: mu(4)=0. S = 1/6.
       K=5: mu(5)=-1. S = 1/6 - 1/5 = -1/30 < 0.
       So for K=5, S is negative, so 1-S > 1. Infeasible!
       
       So pure Möbius is not always feasible after the f(1) adjustment.
       
    2. Alternative: Start with f(k) = 0 for all k, then add "primes" or "structure"?
       Or, start with Möbius, and if infeasible, scale down the whole function?
       If we scale f by alpha, M(x) scales by alpha.
       But the constraint sum f(k)/k = 0 is linear, so scaling preserves it (if it was 0).
       But the evaluator *changes* f(1) to make sum 0.
       So if we return f, the evaluator makes f_eval.
       
       Let's try a different base function: f(k) = 0 for all k.
       Then S=0, f_eval(1)=0. M(x)=0. Feasible. a=0.
       
    3. Let's try to optimize f directly using a simple iterative method or a small LP.
       Since K can be up to 200,000, but we can choose K, let's choose a moderate K (e.g., 5000) and solve an LP, but with a smarter constraint generation or a different solver strategy.
       
       Actually, the previous LP approach worked but was slow or had issues.
       Let's try a different structural approach:
       Use f(k) = 1 for k in a set A, -1 for k in B, 0 otherwise.
       This is a discrete optimization problem.
       
    4. Let's go back to the Möbius idea but with a twist.
       Instead of f(k) = mu(k), let f(k) = mu(k) * w(k) for some weights w(k).
       Or, let's just use the LP but with a much smaller K, like K=1000, which is very fast.
       
    5. Wait, the insight from inspiration 1 (score 1.0745) suggests that a better formulation or a different search strategy works.
       Let's try a simple coordinate ascent or a greedy algorithm on the Möbius basis.
       
    6. Actually, let's look at the objective: a = -sum f(k) log(k) / k.
       To maximize a, we want f(k) to be negative when log(k)/k is positive (which is always for k>=2).
       So we want f(k) to be as negative as possible for large k?
       But the constraint M(x) <= 1 limits how negative f can be.
       
       Let's try f(k) = -1 for all k=2..K, and f(1) adjusted.
       Then M(x) = f(1)*x + sum_{k=2}^K (-1) floor(x/k).
       sum_{k=2}^K floor(x/k) is approximately x * (log log x + B) - x.
       So M(x) ~ f(1)*x - x log log x.
       For M(x) to be <= 1, we need f(1) <= (1 + x log log x)/x ~ log log x.
       So f(1) can be positive.
       But we also have sum f(k)/k = 0 => f(1) + sum_{k=2}^K (-1)/k = 0 => f(1) = sum_{k=2}^K 1/k ~ log K.
       So f(1) is large positive.
       Then M(x) ~ log K * x - x log log x.
       For x=10K, M ~ log K * 10K - 10K log log(10K).
       This is positive and large, so M(x) > 1. Infeasible.
       
    7. Let's try f(k) = 1 for primes, -1 for composites? Too complex.
       
    8. Let's stick to the LP but with K=5000 and a more robust solver call.
       The previous LP code had a bug: it didn't handle the case where the LP solver failed gracefully in all paths, and the fallback was weak.
       Let's implement a clean LP with K=5000.
       
    9. Actually, a simpler structural change: use the fact that M(x) = sum f(k) floor(x/k) is a divisor sum.
       If we set f(k) = mu(k) * g(k) for some g, then M(x) = sum mu(k) g(k) floor(x/k).
       
    10. Let's try a different objective: maximize sum -log(k)/k * f(k) subject to M(x) <= 1 and sum f(k)/k = 0.
        We can solve this LP with K=2000, which is fast.
        
    Let's implement the LP with K=2000 and ensure it's robust.
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 30.0
    
    # Use a moderate K for the LP
    K = 2000
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    c = np.log(domain) / domain  # We minimize sum c_k f_k, which is equivalent to maximizing a
    
    # Constraints:
    # 1. sum f(k)/k = 0
    A_eq = (1.0 / domain).reshape(1, -1)
    b_eq = np.array([0.0])
    
    # 2. M(x) <= 1 for x in 1..10K
    x_max = 10 * K
    x_vals = np.arange(1, x_max + 1, dtype=np.int64)
    k_vals = np.arange(1, K + 1, dtype=np.int64)
    
    # Construct A_ub efficiently
    A_ub = np.floor(x_vals[:, None] / k_vals[None, :]).astype(np.float64)
    b_ub = np.ones(x_max)
    
    bounds = [(-10.0, 10.0)] * K
    
    from scipy.optimize import linprog
    
    try:
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs',
            options={'presolve': True}
        )
        
        if res.success:
            f_opt = res.x
            # Simulate evaluator
            f_clipped = np.clip(f_opt, -10.0, 10.0)
            S = np.sum(f_clipped / domain)
            f_eval = f_clipped.copy()
            f_eval[0] -= S
            
            # Verify M(x) <= 1.0001
            M_final = A_ub @ f_eval
            if np.max(M_final) > 1.0001:
                # Scale down if needed
                scale = 1.0 / np.max(M_final)
                f_eval = f_eval * scale
            
            a = -np.sum(f_eval * c)
            return domain, f_eval
        else:
            # Fallback to Möbius
            return _mobius_fallback(K)
    except Exception:
        return _mobius_fallback(K)

def _mobius_fallback(K):
    mu = np.zeros(K + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
                
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    values_clipped = np.clip(values, -10.0, 10.0)
    S = np.sum(values_clipped / domain)
    values_adj = values_clipped.copy()
    values_adj[0] -= S
    
    # Verify M(x)
    x_max = 10 * K
    M = np.zeros(x_max + 1)
    for k in range(1, K + 1):
        f_k = values_adj[k - 1]
        if f_k != 0:
            M[k::k] += f_k
            
    if np.max(M[1:]) > 1.0001:
        scale = 1.0 / np.max(M[1:])
        values_adj = values_adj * scale
        
    return domain, values_adj

# EVOLVE-BLOCK-END
