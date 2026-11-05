import time
import math
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

# Precompute primes up to 1000
PRIMES = _sieve(1000)
PRIME_SET = set(PRIMES)

def _calculate_coverage(classes, limit=10000):
    """
    Calculate the largest L such that 1..L are covered by the given classes.
    Uses a bytearray for speed.
    Returns L.
    """
    if not classes:
        return 0
    
    covered = bytearray(limit + 1)
    
    # Mark covered positions
    for p, r in classes:
        start = r if r > 0 else p
        if start < 1:
            start += p
        for n in range(start, limit + 1, p):
            covered[n] = 1
            
    # Find the largest L such that 1..L are all True
    L = 0
    for i in range(1, limit + 1):
        if covered[i]:
            L = i
        else:
            break
            
    return L

def _sum_reciprocals_exact(classes):
    """Exact Fraction sum for final verification."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += Fraction(1, p)
    return total

def _solve_ilp_for_length(L, time_limit=10):
    """
    Solve the MILP to find a set of prime residue classes covering 1..L
    with sum(1/p) <= 1.
    
    Returns:
        list of [p, r] if feasible, None otherwise.
    """
    if L == 0:
        return []
    
    n_primes = len(PRIMES)
    # Variables: x[i, j] where i is index in PRIMES, j is residue 0..p_i-1
    # Total variables = sum(p_i) for all p_i <= 1000.
    # sum(p_i) approx 76k. This is large for scipy.milp but might work with high gap.
    # Actually, 76k variables and 1000+76k constraints might be too slow for 150s total.
    # Let's restrict the search space. 
    # We know we need small primes for density. 
    # However, the ILP formulation is the key differentiator.
    # To make it tractable, we can limit the number of primes considered or use a heuristic presolve.
    # But let's try the full MILP first with a tight time limit. If it fails, fallback.
    
    # Constructing the full matrix for 76k variables is memory intensive.
    # Let's check if we can reduce the problem size.
    # A covering system for 1..L typically uses primes up to ~L.
    # But we are capped at 1000.
    
    # Alternative: Use a smaller subset of primes?
    # No, the task says primes <= 1000.
    
    # Let's try to construct the A matrix efficiently.
    # Variables: x_{p, r}
    # Constraints:
    # 1. For each prime p: sum_{r=0}^{p-1} x_{p, r} <= 1 (Distinct primes)
    # 2. Budget: sum_{p, r} (1/p) * x_{p, r} <= 1
    # 3. Coverage: For each n in 1..L, sum_{p, r: n ≡ r (mod p)} x_{p, r} >= 1
    
    # This is a huge matrix. 
    # Number of vars V = sum_{p<=1000} p = 76127.
    # Number of constraints C = 168 (primes) + 1 (budget) + L (coverage).
    # If L=1000, C=1169.
    # Sparse matrix of size 1169 x 76127.
    # This should be manageable with scipy if sparse.
    
    import scipy.sparse as sp
    
    # Map (p_idx, r) to variable index
    var_indices = {}
    current_idx = 0
    for idx, p in enumerate(PRIMES):
        for r in range(p):
            var_indices[(idx, r)] = current_idx
            current_idx += 1
            
    V = current_idx
    
    # Build A matrix (sparse)
    rows = []
    cols = []
    vals = []
    
    # Constraint 1: Sum of residues for each prime <= 1
    # We will build the full A matrix for all constraints at once to use LinearConstraint.
    # Or we can build separate blocks.
    
    # Let's build the sparse matrix directly.
    # Rows 0..167: Prime constraints
    # Row 168: Budget constraint
    # Rows 169..169+L-1: Coverage constraints
    
    for idx, p in enumerate(PRIMES):
        for r in range(p):
            v_idx = var_indices[(idx, r)]
            # Prime constraint
            rows.append(idx)
            cols.append(v_idx)
            vals.append(1.0)
            
            # Budget constraint
            rows.append(len(PRIMES))
            cols.append(v_idx)
            vals.append(1.0 / p)
            
            # Coverage constraints
            # For each n in 1..L such that n % p == r
            # This is expensive to generate in a loop.
            # n = r + k*p
            # If r == 0, n = p, 2p, ...
            # If r > 0, n = r, r+p, ...
            
            if r == 0:
                first_n = p
            else:
                first_n = r
                
            for n in range(first_n, L + 1, p):
                row_idx = len(PRIMES) + 1 + (n - 1)
                rows.append(row_idx)
                cols.append(v_idx)
                vals.append(1.0)
                
    C_total = len(PRIMES) + 1 + L
    
    A = sp.csr_matrix((vals, (rows, cols)), shape=(C_total, V))
    
    # Objective: Minimize 0 (Feasibility)
    # Actually, we can minimize the number of primes used to prefer sparser solutions?
    # Or just 0.
    c = np.zeros(V)
    
    # Bounds: 0 <= x <= 1
    lb = np.zeros(V)
    ub = np.ones(V)
    
    # Constraints:
    # A @ x >= lb_c
    # A @ x <= ub_c
    
    lb_c = np.zeros(C_total)
    ub_c = np.zeros(C_total)
    
    # Prime constraints: sum <= 1  =>  -sum >= -1, sum <= 1
    # We can use LinearConstraint(A, lb_c, ub_c)
    # But we have mixed inequalities.
    # Prime constraints: sum <= 1. So lb_c[i] = -inf, ub_c[i] = 1.
    # Budget: sum <= 1. lb_c[budget] = -inf, ub_c[budget] = 1.
    # Coverage: sum >= 1. lb_c[cover] = 1, ub_c[cover] = inf.
    
    lb_c[:] = -np.inf
    ub_c[:] = np.inf
    
    for i in range(len(PRIMES)):
        ub_c[i] = 1.0
        
    ub_c[len(PRIMES)] = 1.0  # Budget
    
    for i in range(L):
        lb_c[len(PRIMES) + 1 + i] = 1.0
        
    constraints = LinearConstraint(A, lb_c, ub_c)
    
    # Integrality: All binary
    integrality = np.ones(V)
    
    # Solve
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=Bounds(lb, ub), options={"time_limit": time_limit, "mip_gap": 0.0})
    except Exception as e:
        return None
        
    if res.status == 0 or res.status == 2: # 0: optimal, 2: feasible
        x = res.x
        classes = []
        for idx, p in enumerate(PRIMES):
            for r in range(p):
                v_idx = var_indices[(idx, r)]
                if x[v_idx] > 0.5:
                    classes.append([p, r])
        return classes
    else:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Binary search for the maximum L
    # Low: 0, High: 10000 (capped by task, but realistically <= 1000 for C=1)
    # Actually, the task says "capped at 10^5" for score, but L >= 999 is the target.
    # Let's search up to 2000 to be safe.
    
    lo = 0
    hi = 2000
    
    best_classes = []
    
    # We can also run a quick heuristic in parallel or first to get a lower bound?
    # The ILP is the main strategy.
    
    # To speed up ILP, we can start with a smaller L and increase.
    # But binary search is better.
    
    # However, ILP might be slow for large L.
    # Let's try a few specific targets or binary search with tight time limits.
    
    # First, try to find a solution for L=100.
    # If ILP is too slow, we need a fallback.
    
    # Let's do a linear scan from a reasonable estimate upwards?
    # Or binary search.
    
    # Given the complexity, binary search with a time limit per call is risky if it times out often.
    # Let's try to solve for L=1000 directly? No, that's the goal.
    
    # Strategy:
    # 1. Try to solve for L = 1000 with a short time limit.
    # 2. If fails, try L = 500, etc.
    # 3. If success, try to increase L.
    
    # Actually, let's just do a binary search between 0 and 1000 first.
    
    lo = 0
    hi = 1000
    
    # We need to be careful with time.
    # Each ILP call might take 10s.
    # Binary search takes log2(1000) ~ 10 calls. 100s total. Fits in 145s.
    
    while lo < hi:
        if time.time() > deadline - 20.0:
            break
            
        mid = (lo + hi + 1) // 2
        
        # Solve for mid
        # Use a smaller time limit if we are close to deadline
        remaining = deadline - time.time() - 5.0
        t_lim = min(10.0, remaining / 2.0)
        
        classes = _solve_ilp_for_length(mid, time_limit=t_lim)
        
        if classes is not None:
            # Check validity exactly
            if _sum_reciprocals_exact(classes) <= 1:
                # Verify coverage
                cov = _calculate_coverage(classes, limit=mid)
                if cov >= mid:
                    best_classes = classes
                    lo = mid
                else:
                    # ILP said feasible but coverage check failed? 
                    # Maybe due to floating point or logic error.
                    # Treat as infeasible to be safe, or try to fix?
                    # Let's treat as infeasible for binary search step.
                    hi = mid - 1
            else:
                # Invalid budget
                hi = mid - 1
        else:
            # Infeasible or timeout
            hi = mid - 1
            
    # Now try to extend beyond 1000 if we found a solution for 1000
    if lo >= 1000:
        # Try 1100, 1200...
        current_L = lo
        while time.time() < deadline - 10.0:
            next_L = current_L + 100
            if next_L > 10000: break
            
            t_lim = min(10.0, (deadline - time.time() - 5.0) / 2.0)
            classes = _solve_ilp_for_length(next_L, time_limit=t_lim)
            
            if classes is not None:
                if _sum_reciprocals_exact(classes) <= 1:
                    cov = _calculate_coverage(classes, limit=next_L)
                    if cov >= next_L:
                        best_classes = classes
                        current_L = next_L
                        continue
            break # Failed to extend
            
    # If no ILP solution found, use a fallback heuristic
    if not best_classes:
        # Simple fallback: Greedy
        # This is just to ensure we return *something* valid if ILP fails completely
        classes = []
        used_primes = set()
        budget = 0.0
        for p in PRIMES:
            if budget + 1.0/p <= 1.0:
                # Pick a residue that covers the most uncovered numbers?
                # For now, just pick r=0 or random.
                # This fallback is weak.
                pass
        # Better fallback: Use the seed's logic if possible?
        # Let's just return empty or a trivial valid set.
        # A trivial valid set: [[2, 0]] covers evens. L=1? No, 1 is odd.
        # [[2, 1], [3, 2]] covers 1 (3 mod 2? no. 1 mod 2 = 1. 2 mod 3 = 2).
        # 1: 1%2=1 (covered by 2,1). 2: 2%3=2 (covered by 3,2). 3: 3%2=1 (covered). 4: 4%2=0 (not covered).
        # So L=3.
        fallback = [[2, 1], [3, 2]]
        if _sum_reciprocals_exact(fallback) <= 1:
            best_classes = fallback

    return best_classes

# EVOLVE-BLOCK-END
