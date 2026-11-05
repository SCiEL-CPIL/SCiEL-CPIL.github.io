import time
import numpy as np
from fractions import Fraction
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # 1. Precompute primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # Precompute which integers 1..L are covered by each (p, r)
    # For a given L, build the constraint matrix
    
    def check_feasible(L, time_limit=30):
        """Check if we can cover 1..L with prime residue classes, sum(1/p) <= 1"""
        if L < 1:
            return True, []
        
        # Filter primes: only include primes p <= L (larger primes can't help cover 1..L efficiently)
        # Actually, any prime p > L can only cover at most 1 number in 1..L (if r != 0)
        # or L/p numbers. For p > L, it covers 0 or 1 number.
        # We'll include all primes but the ILP will decide.
        
        # Build variable indexing
        # Variables: x_{p,r} for each prime p and residue r in [0, p)
        # Total variables: sum of p for all primes in our set
        # This could be large. Let's limit to smaller primes first.
        
        # Strategy: use primes up to some bound B
        # The sum of reciprocals of primes up to B is ~ log(log(B)) + M
        # For sum <= 1, we need B around 100-200
        # Let's use primes up to 200 initially, expand if needed
        
        # Actually, let's be smarter. The density constraint sum(1/p) <= 1 means
        # we can't use too many small primes. Let's estimate how many primes we can use.
        # 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 + 1/17 + 1/19 + 1/23 + 1/29 + 1/31
        # = 0.5 + 0.333 + 0.2 + 0.143 + 0.091 + 0.077 + 0.059 + 0.053 + 0.043 + 0.034 + 0.032
        # ≈ 1.565, which already exceeds 1.
        # So we can use maybe 8-10 primes total.
        
        # Let's use primes up to 100 to keep the ILP manageable
        # That's 25 primes, with total variables = sum of p for p in first 25 primes
        # 2+3+5+7+11+13+17+19+23+29+31+37+41+43+47+53+59+61+67+71+73+79+83+89+97
        # = let's compute: 2+3=5, +5=10, +7=17, +11=28, +13=41, +17=58, +19=77, +23=100,
        # +29=129, +31=160, +37=197, +41=238, +43=281, +47=328, +53=381, +59=440,
        # +61=501, +67=568, +71=639, +73=712, +79=791, +83=874, +89=963, +97=1060
        # So ~1060 variables. That's manageable for MILP.
        
        # Let's use primes up to 100
        active_primes = [p for p in primes if p <= 100]
        if not active_primes:
            return False, []
        
        # Build variable mapping
        var_index = {}  # (p, r) -> index
        var_list = []   # list of (p, r) in order
        for p in active_primes:
            for r in range(p):
                var_index[(p, r)] = len(var_list)
                var_list.append((p, r))
        
        n_vars = len(var_list)
        n_constraints = L + n_primes  # L coverage constraints + one per prime
        
        # Objective: minimize sum of x (fewer classes is better, but any feasible is ok)
        # Actually, we just want feasibility. Set objective to 0.
        c = np.zeros(n_vars)
        
        # A matrix for constraints
        # We'll build it row by row
        A_rows = []
        
        # Coverage constraints: for each i in 1..L, sum of x_{p,r} where i % p == r >= 1
        for i in range(1, L + 1):
            row = np.zeros(n_vars)
            for (p, r), idx in var_index.items():
                if i % p == r:
                    row[idx] = 1
            A_rows.append(row)
        
        # Prime uniqueness constraints: for each prime p, sum of x_{p,r} <= 1
        for p in active_primes:
            row = np.zeros(n_vars)
            for r in range(p):
                row[var_index[(p, r)]] = 1
            A_rows.append(row)
        
        # Reciprocal budget constraint: sum of (1/p) * x_{p,r} <= 1 for all p,r
        # This is: for each p, (1/p) * sum_r x_{p,r} <= 1
        # But since sum_r x_{p,r} <= 1, this is automatically satisfied if 1/p <= 1, which is true.
        # Wait, no. The constraint is sum over ALL selected classes of 1/p <= 1.
        # So: sum_{p} (1/p) * (sum_r x_{p,r}) <= 1
        # This is a single constraint.
        row = np.zeros(n_vars)
        for (p, r), idx in var_index.items():
            row[idx] = 1.0 / p
        A_rows.append(row)
        
        A = np.array(A_rows)
        
        # Constraints:
        # Rows 0..L-1: A[i] @ x >= 1 (coverage)
        # Rows L..L+n_primes-1: A[i] @ x <= 1 (prime uniqueness)
        # Row L+n_primes: A[i] @ x <= 1 (budget)
        
        lb = np.zeros(len(A_rows))
        ub = np.zeros(len(A_rows))
        
        # Coverage: >= 1
        lb[0:L] = 1
        ub[0:L] = np.inf
        
        # Prime uniqueness: <= 1
        lb[L:L + len(active_primes)] = 0
        ub[L:L + len(active_primes)] = 1
        
        # Budget: <= 1
        lb[L + len(active_primes)] = 0
        ub[L + len(active_primes)] = 1
        
        constraints = LinearConstraint(A, lb, ub)
        
        # Variable bounds: 0 <= x <= 1
        var_bounds = Bounds(0, 1)
        
        # Integrality: all integer
        integrality = np.ones(n_vars)
        
        # Solve
        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=var_bounds,
                options={"time_limit": time_limit, "mip_rel_gap": 0.0}
            )
            
            if res.status == 0:  # Optimal
                # Extract solution
                classes = []
                for (p, r), idx in var_index.items():
                    if res.x[idx] > 0.5:
                        classes.append([p, r])
                return True, classes
            elif res.status == 1:  # Infeasible
                return False, []
            else:
                # Timeout or other status
                # Try to get best feasible solution if available
                if res.x is not None:
                    classes = []
                    for (p, r), idx in var_index.items():
                        if res.x[idx] > 0.5:
                            classes.append([p, r])
                    # Verify
                    if verify_classes(classes, L):
                        return True, classes
                return False, []
                
        except Exception as e:
            return False, []
    
    def verify_classes(classes, L):
        """Verify that classes cover 1..L and sum(1/p) <= 1"""
        if not classes:
            return L <= 0
        
        # Check distinct primes
        primes_used = set()
        for p, r in classes:
            if p in primes_used:
                return False
            primes_used.add(p)
            if r < 0 or r >= p:
                return False
        
        # Check budget
        s = Fraction(0)
        for p, r in classes:
            s += Fraction(1, p)
        if s > 1:
            return False
        
        # Check coverage
        covered = np.zeros(L + 1, dtype=bool)
        for p, r in classes:
            for i in range(1, L + 1):
                if i % p == r:
                    covered[i] = True
        return np.all(covered[1:])
    
    # Binary search for maximum L
    # Start with a reasonable lower bound
    # The greedy approach might give us a hint, but let's start from a conservative value
    
    # First, let's try to find a feasible L using a quick heuristic to set a lower bound
    # Then binary search upward
    
    # Quick heuristic: use small primes with optimal residues
    def quick_heuristic(L):
        """Quick check using greedy assignment"""
        # Use primes up to 50
        active_primes = [p for p in primes if p <= 50]
        residues = {p: 0 for p in active_primes}
        counts = np.zeros(L + 1, dtype=np.int32)
        
        for p in sorted(active_primes):
            best_r = 0
            best_gain = -1
            for r in range(p):
                start = r if r > 0 else p
                if start > L:
                    gain = 0
                else:
                    idx = np.arange(start, L + 1, p)
                    gain = np.sum(counts[idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[p] = best_r
            start = best_r if best_r > 0 else p
            if start <= L:
                idx = np.arange(start, L + 1, p)
                counts[idx] += 1
        
        # Check if all covered
        if np.all(counts[1:] > 0):
            return True, [[p, residues[p]] for p in active_primes]
        return False, []
    
    # Find initial lower bound
    lo = 1
    hi = 1000  # Upper bound to search
    
    # First, check if even L=100 is feasible
    feasible, classes = check_feasible(100, time_limit=15)
    if not feasible:
        # Try smaller
        for test_L in [50, 30, 20, 10, 5, 1]:
            feasible, classes = check_feasible(test_L, time_limit=10)
            if feasible:
                lo = test_L
                break
        else:
            # Can't even cover 1? Return a trivial solution
            return [[2, 1]]  # Covers odd numbers, sum=1/2 <= 1
    
    # Now binary search for maximum L
    best_L = lo
    best_classes = classes
    
    # Expand search range
    while time.time() < deadline - 30:
        # Try to go higher
        test_L = min(best_L + max(1, best_L // 4), hi)
        feasible, classes = check_feasible(test_L, time_limit=20)
        
        if feasible:
            best_L = test_L
            best_classes = classes
            # Try to go higher
            hi = min(hi * 2, 100000)
        else:
            # Binary search between best_L and test_L
            low = best_L
            high = test_L - 1
            while low <= high and time.time() < deadline - 10:
                mid = (low + high) // 2
                feasible, classes = check_feasible(mid, time_limit=15)
                if feasible:
                    best_L = mid
                    best_classes = classes
                    low = mid + 1
                else:
                    high = mid - 1
            break
    
    # Final verification
    if best_classes and verify_classes(best_classes, best_L):
        return best_classes
    
    # Fallback
    return [[2, 1]]
