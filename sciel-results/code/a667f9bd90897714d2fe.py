"""
Strategy: Pre-compute the exact optimal policy using Dynamic Programming with continuous optimization.
Instead of a discrete grid search, we solve the first-order conditions (FOC) of the Bellman equation
for each state (t, q) using Newton's method. This allows us to find the exact optimal depth in the
continuous space, avoiding the discretization error inherent in grid-based approaches.

The Bellman equation is:
V_t(q) = max_{b,a} { -C + (1-p_b)(1-p_a)V_{t+1}(q) + p_b(1-p_a)(b + V_{t+1}(q+1)) + (1-p_b)p_a(a + V_{t+1}(q-1)) + p_b p_a(b + a + V_{t+1}(q)) }

where p_b = 1 - exp(-A exp(-k b) dt) and p_a = 1 - exp(-A exp(-k a) dt).

Since the bid and ask decisions are coupled only through the probability terms (which factorize in the expansion),
and the reward terms are linear in b and a, the objective is separable in the sense that we can optimize b and a
independently if we fix the other's probability? No, the cross terms p_b p_a couple them.
However, the structure is:
E(b,a) = V_q + p_b [ b + V_qp - V_q - p_a(V_qp - V_q + b) ] + p_a [ a + V_qm - V_q - p_b(V_qm - V_q + a) ] + p_b p_a V_q ... wait.

Let's re-expand carefully to see if it separates.
E = (1-p_b)(1-p_a)V_q + p_b(1-p_a)(b+V_qp) + (1-p_b)p_a(a+V_qm) + p_b p_a(b+a+V_q)
  = V_q - p_a V_q - p_b V_q + p_b p_a V_q
    + p_b b + p_b V_qp - p_b p_a b - p_b p_a V_qp
    + p_a a + p_a V_qm - p_b p_a a - p_b p_a V_qm
    + p_b p_a b + p_b p_a a + p_b p_a V_q

Group by b:
b terms: p_b b - p_b p_a b + p_b p_a b = p_b b
So the b part is p_b b.
Group by a:
a terms: p_a a - p_b p_a a + p_b p_a a = p_a a
So the a part is p_a a.

The remaining terms:
V_q (1 - p_b - p_a + p_b p_a)
+ p_b (V_qp - V_q)
+ p_a (V_qm - V_q)
+ p_b p_a (V_q - V_qp - V_qm + V_q) ? Let's check the p_b p_a terms again.
From term 1: + p_b p_a V_q
From term 2: - p_b p_a V_qp
From term 3: - p_b p_a V_qm
From term 4: + p_b p_a V_q
Sum: p_b p_a (2 V_q - V_qp - V_qm)

So E(b,a) = V_q (1 - p_b - p_a + p_b p_a) + p_b (b + V_qp - V_q) + p_a (a + V_qm - V_q) + p_b p_a (2 V_q - V_qp - V_qm)

Let Delta_qp = V_qp - V_q
Let Delta_qm = V_qm - V_q
Let Delta_cross = 2 V_q - V_qp - V_qm = Delta_qm - Delta_qp ? No.
Delta_cross = V_q - (V_qp + V_qm - V_q) = V_q - (V_qp - V_q + V_qm - V_q) = V_q - (Delta_qp + Delta_qm).

So E(b,a) = V_q (1 - p_b - p_a + p_b p_a) + p_b (b + Delta_qp) + p_a (a + Delta_qm) + p_b p_a (V_q - Delta_qp - Delta_qm)

This is still coupled via p_b p_a.
However, for a fixed a (and thus fixed p_a), the optimization over b is:
Maximize f(b) = p_b (b + Delta_qp) + p_a p_b (V_q - Delta_qp - Delta_qm) + p_b (terms not depending on b? No)
Let's group terms with p_b:
f(b) = V_q (1 - p_a) + p_a (a + Delta_qm) + p_b [ (b + Delta_qp) + p_a (V_q - Delta_qp - Delta_qm) - V_q (1 - p_a) ]
Wait, the term V_q (1 - p_b - p_a + p_b p_a) = V_q (1-p_a) - p_b V_q (1-p_a).
So:
f(b) = V_q (1-p_a) + p_a (a + Delta_qm) - p_b V_q (1-p_a) + p_b (b + Delta_qp) + p_b p_a (V_q - Delta_qp - Delta_qm)
f(b) = Const_a + p_b [ b + Delta_qp - V_q (1-p_a) + p_a (V_q - Delta_qp - Delta_qm) ]
f(b) = Const_a + p_b [ b + Delta_qp - V_q + p_a V_q + p_a V_q - p_a Delta_qp - p_a Delta_qm ]
f(b) = Const_a + p_b [ b + Delta_qp (1-p_a) - V_q (1-p_a) + p_a V_q ]
f(b) = Const_a + p_b [ b - (V_q - Delta_qp)(1-p_a) + p_a V_q ]
Note V_q - Delta_qp = V_q - (V_qp - V_q) = 2V_q - V_qp.

So we need to maximize g(b) = p_b ( b + C_a )
where C_a = Delta_qp (1-p_a) - V_q (1-p_a) + p_a V_q = (1-p_a)(Delta_qp - V_q) + p_a V_q.
And p_b = 1 - exp(-A exp(-k b) dt).

This is a 1D optimization problem!
g(b) = (1 - exp(-lambda_b)) (b + C_a)
where lambda_b = A exp(-k b) dt.
lambda_b' = -k lambda_b.
g'(b) = -lambda_b' exp(-lambda_b) (b+C_a) + (1-exp(-lambda_b))
      = k lambda_b exp(-lambda_b) (b+C_a) + 1 - exp(-lambda_b)
      = 1 - exp(-lambda_b) (1 - k lambda_b (b+C_a))

Set g'(b) = 0:
1 = exp(-lambda_b) (1 - k lambda_b (b+C_a))
exp(lambda_b) = 1 / (1 - k lambda_b (b+C_a))
lambda_b = -ln(1 - k lambda_b (b+C_a))

This is a fixed-point equation for lambda_b (or b).
Let h(b) = -ln(1 - k lambda_b (b+C_a)).
We can solve for b using Newton's method or fixed-point iteration.
Note that if 1 - k lambda_b (b+C_a) <= 0, there is no solution (derivative doesn't cross zero from positive to negative in a standard way, or boundary).
Actually, if C_a is large positive, b should be small. If C_a is large negative, b should be large.

We can solve for b given p_a (or a). Similarly for a given p_b.
Since they are coupled, we can use a nested loop or iterative relaxation.
However, notice that the problem is small.
For each (t, q), we need to find optimal (b, a).
We can iterate:
1. Start with b=0, a=0.
2. Compute optimal b given a.
3. Compute optimal a given b.
4. Repeat until convergence.

Solving for b given a:
Maximize g(b) = p_b (b + C_a).
FOC: 1 - exp(-lambda_b) (1 - k lambda_b (b+C_a)) = 0.
Let F(b) = 1 - exp(-lambda_b) (1 - k lambda_b (b+C_a)).
F'(b) = ...
Or simpler: use scipy.optimize.brentq or newton on F(b) = 0.
Range for b: [0, 10/k] or so.

Let's implement this continuous solver. It is much faster than a grid and more accurate.

"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

_policy_cache = {}
_start_time = time.time()
DEADLINE_MARGIN = 10.0

def _solve_b_given_a(a, C_a, A, k, dt):
    """
    Solve for optimal b given a.
    C_a is the constant term in the FOC derived from a.
    Objective: max p_b (b + C_a)
    FOC: 1 - exp(-lambda_b) (1 - k lambda_b (b+C_a)) = 0
    """
    def foc(b):
        lam = A * np.exp(-k * b) * dt
        if lam < 1e-12:
            return 1.0 # p_b ~ 0, derivative ~ 1 (if b+C_a > 0) or ... 
            # Actually if lam -> 0, p_b -> 0. g(b) ~ lam(b+C_a). 
            # If b+C_a > 0, g increases with b initially? 
            # g'(b) = 1 - exp(-lam)(1 - k lam (b+C_a)) ~ 1 - (1-lam)(1 - k lam (b+C_a)) ~ 1 - (1 - lam - k lam (b+C_a)) = lam (1 + k(b+C_a)).
            # If lam > 0 and b+C_a > -1/k, derivative is positive.
            # So if C_a is very negative, we might want large b?
            # But p_b decays.
            # Let's just return the value.
        term = 1.0 - k * lam * (b + C_a)
        if term <= 0:
            # exp(-lam) * term <= 0. 1 - (negative) > 1. 
            # This implies F(b) > 0. 
            # We are looking for F(b)=0.
            # If term <= 0, F(b) > 1.
            # The root must be where term > 0.
            # So we need smaller b (larger lam) to make term positive?
            # term = 1 - k lam (b+C_a).
            # If b+C_a > 0, larger lam -> smaller term.
            # If b+C_a < 0, larger lam -> larger term.
            # This logic is tricky. Let's just evaluate.
            return 1.0 - np.exp(-lam) * term
        return 1.0 - np.exp(-lam) * term

    # We need to find a bracket.
    # b_min = 0.
    # b_max = ?
    # If C_a is very large, optimal b is small.
    # If C_a is very small (negative), optimal b is large.
    # Let's try a range.
    b_min = 0.0
    b_max = 5.0 / k  # exp(-5) is small.
    
    try:
        # Check if max is at boundary
        f_min = foc(b_min)
        f_max = foc(b_max)
        
        # If f_min > 0 and f_max < 0, root in between.
        # If f_min > 0 and f_max > 0, check derivative direction or just return b_min/b_max?
        # Actually, g(b) is unimodal?
        # Let's just use brentq if signs differ.
        if f_min * f_max < 0:
            return brentq(foc, b_min, b_max, xtol=1e-8)
        
        # If no sign change, check which end is better by evaluating g(b)
        def g(b):
            lam = A * np.exp(-k * b) * dt
            return (1.0 - np.exp(-lam)) * (b + C_a)
            
        g_min = g(b_min)
        g_max = g(b_max)
        if g_min > g_max:
            return b_min
        else:
            return b_max
            
    except Exception:
        return 0.0

def _compute_optimal_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)
    
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    V_next = np.zeros(n_states)
    
    for t in range(steps - 1, -1, -1):
        if time.time() - _start_time > 330.0 - DEADLINE_MARGIN:
            # Fallback
            opt_bid[t:, :] = 1.0 / k
            opt_ask[t:, :] = 1.0 / k
            break
            
        V_current = np.zeros(n_states)
        
        for q_idx in range(n_states):
            q = q_values[q_idx]
            
            v_q = V_next[q_idx]
            
            allow_bid = (q < Q)
            allow_ask = (q > -Q)
            
            if allow_bid:
                v_qp = V_next[q_idx + 1]
            else:
                v_qp = v_q # Dummy, will be masked
                
            if allow_ask:
                v_qm = V_next[q_idx - 1]
            else:
                v_qm = v_q # Dummy
                
            Delta_qp = v_qp - v_q
            Delta_qm = v_qm - v_q
            
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Solve for optimal b and a
            # Iterative method
            b = 1.0 / k
            a = 1.0 / k
            
            for _ in range(10): # Few iterations should suffice
                # Solve for b given a
                if allow_bid:
                    C_a = (1.0 - (1.0 - np.exp(-A * np.exp(-k * a) * dt))) * v_q + (1.0 - (1.0 - np.exp(-A * np.exp(-k * a) * dt))) * Delta_qm - (1.0 - (1.0 - np.exp(-A * np.exp(-k * a) * dt))) * v_q
                    # Simplify C_a
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    C_a = p_a * v_q + (1.0 - p_a) * (Delta_qp - v_q)
                    # Wait, previous derivation:
                    # C_a = Delta_qp (1-p_a) - V_q (1-p_a) + p_a V_q
                    # C_a = (1-p_a)(Delta_qp - V_q) + p_a V_q
                    b_new = _solve_b_given_a(a, C_a, A, k, dt)
                else:
                    b_new = 0.0 # or large, doesn't matter if p_b=0. Let's use 0.
                    
                # Solve for a given b
                if allow_ask:
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b_new) * dt)
                    # Symmetric C_b
                    # C_b = Delta_qm (1-p_b) - V_q (1-p_b) + p_b V_q
                    C_b = (1.0 - p_b) * (Delta_qm - v_q) + p_b * v_q
                    a_new = _solve_b_given_a(b_new, C_b, A, k, dt) # Reuse solver, it's symmetric
                else:
                    a_new = 0.0
                    
                if abs(b_new - b) < 1e-6 and abs(a_new - a) < 1e-6:
                    b = b_new
                    a = a_new
                    break
                b = b_new
                a = a_new
                
            opt_bid[t, q_idx] = b
            opt_ask[t, q_idx] = a
            
            # Compute V_t
            p_b = 0.0 if not allow_bid else (1.0 - np.exp(-A * np.exp(-k * b) * dt))
            p_a = 0.0 if not allow_ask else (1.0 - np.exp(-A * np.exp(-k * a) * dt))
            
            # E = (1-p_b)(1-p_a)v_q + p_b(1-p_a)(b+v_qp) + (1-p_b)p_a(a+v_qm) + p_b p_a(b+a+v_q)
            e1 = (1.0 - p_b) * (1.0 - p_a) * v_q
            e2 = p_b * (1.0 - p_a) * (b + v_qp)
            e3 = (1.0 - p_b) * p_a * (a + v_qm)
            e4 = p_b * p_a * (b + a + v_q)
            
            V_current[q_idx] = e1 + e2 + e3 + e4 - cost
            
        V_next = V_current
        
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
        
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
