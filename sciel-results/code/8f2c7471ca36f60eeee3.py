import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80.0
    
    N = 256
    nums = list(range(2, 257))
    n = len(nums)
    
    # Build conflict graph
    adj = [set() for _ in range(n)]
    conflicts = []
    for i in range(n):
        a = nums[i]
        for j in range(i + 1, n):
            b = nums[j]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                conflicts.append((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    weights = [Fraction(1, nums[i]) for i in range(n)]
    float_weights = [float(w) for w in weights]
    
    best_solution = set()
    best_weight = Fraction(0)
    
    def update_best(selected):
        nonlocal best_solution, best_weight
        w = sum(weights[i] for i in selected)
        if w > best_weight:
            best_weight = w
            best_solution = set(selected)
    
    # Greedy with different orderings
    def greedy(order):
        selected = set()
        for i in order:
            if all(j not in selected for j in adj[i]):
                selected.add(i)
        return selected
    
    # Multiple greedy strategies
    order1 = sorted(range(n), key=lambda i: -float_weights[i])
    update_best(greedy(order1))
    
    order2 = sorted(range(n), key=lambda i: (-float_weights[i], -len(adj[i])))
    update_best(greedy(order2))
    
    order3 = sorted(range(n), key=lambda i: (-float_weights[i] / max(1, len(adj[i])), -len(adj[i])))
    update_best(greedy(order3))
    
    # Local search improvement
    def local_improve(selected):
        current = set(selected)
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            for i in range(n):
                if i not in current and all(j not in current for j in adj[i]):
                    new_set = current | {i}
                    if sum(weights[j] for j in new_set) > sum(weights[j] for j in current):
                        current = new_set
                        improved = True
            for i in list(current):
                for j in range(n):
                    if j not in current:
                        new_set = (current - {i}) | {j}
                        valid = all(k not in adj[j] for k in new_set if k != j)
                        if valid and sum(weights[k] for k in new_set) > sum(weights[k] for k in current):
                            current = new_set
                            improved = True
                            break
                if improved:
                    break
        return current
    
    improved = local_improve(best_solution)
    update_best(improved)
    
    # Branch and bound
    def branch_and_bound():
        nonlocal best_weight, best_solution
        
        def bb(available, selected, depth):
            nonlocal best_weight, best_solution
            if time.time() > DEADLINE - 5:
                return
            ub = sum(weights[i] for i in available) + sum(weights[i] for i in selected)
            if ub <= best_weight:
                return
            if not available:
                update_best(selected)
                return
            v = max(available, key=lambda i: float_weights[i])
            new_avail = set(i for i in available if i != v and i not in adj[v])
            new_sel = selected | {v}
            bb(new_avail, new_sel, depth + 1)
            if time.time() > DEADLINE - 5:
                return
            new_avail2 = set(available)
            new_avail2.discard(v)
            bb(new_avail2, selected, depth + 1)
        
        bb(set(range(n)), set(), 0)
    
    try:
        branch_and_bound()
    except RecursionError:
        pass
    except Exception:
        pass
    
    # MIP if time remains
    if time.time() < DEADLINE - 10:
        c = np.array([-float_weights[i] for i in range(n)])
        m = len(conflicts)
        if m > 0:
            A = np.zeros((m, n))
            for k, (i, j) in enumerate(conflicts):
                A[k, i] = 1.0
                A[k, j] = 1.0
            lb_arr = np.zeros(m)
            ub_arr = np.ones(m)
        else:
            A = np.zeros((1, n))
            lb_arr = np.zeros(1)
            ub_arr = np.ones(1)
        
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        constraints = LinearConstraint(A, lb=lb_arr, ub=ub_arr)
        integrality = np.ones(n)
        
        try:
            time_limit = max(1, int(DEADLINE - time.time()) - 5)
            result = milp(c=c, constraints=constraints, integrality=integrality,
                         bounds=bounds, options={'time_limit': time_limit})
            if result is not None and result.success and result.x is not None:
                mip_set = set(i for i in range(n) if result.x[i] > 0.5)
                valid = True
                for a in mip_set:
                    for b in adj[a]:
                        if b in mip_set and b > a:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    update_best(mip_set)
        except Exception:
            pass
    
    # Final verification
    result_list = [nums[i] for i in best_solution]
    valid = True
    for idx_a in range(len(result_list)):
        for idx_b in range(idx_a + 1, len(result_list)):
            a, b = result_list[idx_a], result_list[idx_b]
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                valid = False
                break
        if not valid:
            break
    
    if not valid or not result_list:
        order = sorted(range(n), key=lambda i: -float_weights[i])
        selected = set()
        for i in order:
            if all(j not in selected for j in adj[i]):
                selected.add(i)
        result_list = [nums[i] for i in selected]
    
    if not result_list:
        result_list = [2]
    
    return result_list
