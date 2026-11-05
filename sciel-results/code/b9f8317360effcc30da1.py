import time
import random
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    all_numbers = list(range(1, N + 1))
    
    def get_score(A):
        return sum(1.0 / x for x in A)
    
    def check_valid_addition(A_set, x, exclude=None):
        """Check if adding x to A_set (with exclude removed) keeps it 3-AP-free."""
        for u in A_set:
            if u == exclude:
                continue
            # Case 1: x is the middle term. AP: (u, x, v) where u+v=2x
            v1 = 2 * x - u
            if 0 < v1 <= N and v1 != u and v1 != exclude and v1 in A_set:
                return False
            # Case 2: x is an end term, u is the other end. m=(x+u)/2
            s = x + u
            if s % 2 == 0:
                m = s // 2
                if 0 < m <= N and m != x and m != u and m != exclude and m in A_set:
                    return False
        return True

    def greedy_init():
        A = set()
        for x in range(1, N + 1):
            if check_valid_addition(A, x, exclude=None):
                A.add(x)
        return A

    def local_search(A, time_limit=None):
        if time_limit is None:
            time_limit = deadline
        A = set(A)
        score = get_score(A)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try swaps: remove x, add y where y < x
            for x in sorted(A, reverse=True):
                if time.time() > time_limit:
                    break
                for y in range(1, x):
                    if y in A:
                        continue
                    if check_valid_addition(A, y, exclude=x):
                        A.remove(x)
                        A.add(y)
                        score = get_score(A)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                # Try adding elements
                for y in range(1, N + 1):
                    if y in A:
                        continue
                    if check_valid_addition(A, y, exclude=None):
                        A.add(y)
                        score = get_score(A)
                        improved = True
                        break
        return A, score

    # MILP formulation
    def solve_milp(time_limit=None):
        if time_limit is None:
            time_limit = deadline
        if time.time() > time_limit - 5.0:
            return None, 0.0
        
        # Enumerate all 3-term APs
        # A 3-AP is (a, b, c) with a < b < c and 2b = a + c
        # Equivalently: a + c = 2b, a < b < c
        # For each middle b, and each a < b, c = 2b - a > b
        ap_constraints = []
        for b in range(1, N + 1):
            for a in range(1, b):
                c = 2 * b - a
                if c > b and c <= N:
                    # Constraint: x_a + x_b + x_c <= 2
                    ap_constraints.append((a - 1, b - 1, c - 1))
        
        if not ap_constraints:
            return set(range(1, N + 1)), sum(1.0 / x for x in range(1, N + 1))
        
        n = N
        # Objective: maximize sum(1/x * x_i) => minimize -sum(1/x * x_i)
        c = np.array([-1.0 / x for x in range(1, N + 1)])
        
        # Build constraint matrix
        m = len(ap_constraints)
        A_mat = np.zeros((m, n))
        for i, (ia, ib, ic) in enumerate(ap_constraints):
            A_mat[i, ia] = 1.0
            A_mat[i, ib] = 1.0
            A_mat[i, ic] = 1.0
        
        # Constraints: 0 <= x_i <= 1, and A_mat @ x <= 2
        constraints = LinearConstraint(A_mat, ub=2.0)
        integrality = np.ones(n)
        bounds = Bounds(lb=0.0, ub=1.0)
        
        try:
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": max(1, time_limit - time.time() - 2.0),
                         "mip_rel_gap": 0.01}
            )
            if res.status == 0 and res.x is not None:
                sol = np.round(res.x).astype(int)
                A_milp = set(i + 1 for i in range(n) if sol[i] == 1)
                if A_milp:
                    return A_milp, get_score(A_milp)
        except Exception:
            pass
        return None, 0.0

    # Main logic
    # Start with greedy
    best_A = greedy_init()
    best_score = get_score(best_A)
    
    # Local search on greedy
    best_A, best_score = local_search(best_A, deadline - 20.0)
    
    # Try MILP if we have time
    if time.time() < deadline - 10.0:
        milp_A, milp_score = solve_milp(deadline - 5.0)
        if milp_A is not None and milp_score > best_score:
            best_A = milp_A
            best_score = milp_score
            # Refine with local search
            if time.time() < deadline - 5.0:
                best_A, best_score = local_search(best_A, deadline - 2.0)
    
    # Random restarts with perturbation
    while time.time() < deadline - 3.0:
        current_A = set(best_A)
        num_remove = random.randint(1, 3)
        for _ in range(num_remove):
            if current_A:
                to_remove = random.choice(list(current_A))
                current_A.remove(to_remove)
        
        current_A, current_score = local_search(current_A, deadline - 2.0)
        
        if current_score > best_score:
            best_score = current_score
            best_A = current_A
    
    return list(best_A)
