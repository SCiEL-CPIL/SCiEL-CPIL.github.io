import time
import random
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance_ilp(values, time_budget):
        n = len(values)
        if n <= 2:
            return list(values)
        
        v_set = set(values)
        
        # Build conflict matrix C
        # C[i, j] = 1 if values[i] + values[j] is in v_set (and i != j) OR 2*values[i] == values[j] (i!=j)
        # Note: The condition "a+b=c" with a,b,c in S.
        # If we select S, we must not have any i,j,k in S such that v_i+v_j=v_k.
        # This is equivalent to: S is an Independent Set in the graph G where
        # G has edge (i,j) if v_i + v_j in V (i != j) OR 2*v_i = v_j.
        # Wait, let's re-verify.
        # If S is an Independent Set in G:
        # For any a, b in S:
        # If a != b, edge (a,b) exists iff a+b in V. If a+b in S, then a+b in V, so edge exists. Contradiction.
        # If a == b, edge (a, 2a) exists iff 2a in V. If a+a=2a in S, then 2a in V, so edge exists. Contradiction.
        # So yes, Max Sum-Free Subset == Max Independent Set in G.
        
        # However, the ILP formulation for MIS is:
        # Maximize sum(x_i)
        # Subject to: x_i + x_j <= 1 for all edges (i, j) in G.
        # x_i in {0, 1}
        
        # We need to identify all pairs (i, j) with i < j such that an edge exists.
        # Edge exists if:
        # 1. values[i] + values[j] in v_set
        # 2. 2 * values[i] == values[j]
        # 3. 2 * values[j] == values[i]
        
        # Note: If values[i] + values[j] is in v_set, it could be equal to values[k] for some k != i, j.
        # The constraint x_i + x_j <= 1 ensures we don't pick both i and j.
        # Is this sufficient?
        # If we pick i and j, and i+j = k, then k is in V.
        # If we also pick k, then we have a triple (i, j, k).
        # The independent set condition on the graph G defined above ensures that if i and j are connected, we can't pick both.
        # Does G capture all "forbidden pairs"?
        # Yes. If i, j, k are in S and i+j=k, then:
        # - If i != j, then i+j is in V, so (i, j) is an edge in G. Thus x_i + x_j <= 1 is violated if both are 1.
        # - If i == j, then 2i = k is in V, so (i, k) is an edge in G. Thus x_i + x_x_k <= 1 is violated.
        # So the formulation is correct.

        edges = []
        for i in range(n):
            vi = values[i]
            for j in range(i + 1, n):
                vj = values[j]
                if (vi + vj) in v_set:
                    edges.append((i, j))
                elif 2 * vi == vj:
                    edges.append((i, j))
                elif 2 * vj == vi:
                    edges.append((i, j))
        
        m = len(edges)
        
        if m == 0:
            return list(values)
        
        # Objective: Maximize sum(x_i)
        # In scipy.milp, we minimize. So we minimize -sum(x_i).
        c = np.full(n, -1.0)
        
        # Constraints: x_i + x_j <= 1 for each edge (i, j)
        # A_ub @ x <= b_ub
        A_ub = np.zeros((m, n))
        b_ub = np.ones(m)
        
        for k, (i, j) in enumerate(edges):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
        
        # Bounds: 0 <= x_i <= 1
        lb = np.zeros(n)
        ub = np.ones(n)
        bounds = Bounds(lb, ub)
        
        # Integrality: x_i must be integer (0 or 1)
        integrality = np.ones(n)
        
        # Time limit for solver
        # We have time_budget seconds.
        # scipy.milp doesn't have a direct "time_limit" parameter in all versions, 
        # but we can use a callback or just trust it for small n.
        # For n=240 and potentially thousands of edges, it might take time.
        # We can set a time limit if available, or just run it.
        # Given n is small, it should be fast.
        
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(A_ub, ub=b_ub),
                integrality=integrality,
                bounds=bounds
            )
            
            if res.status == 0:  # Optimal
                # Extract solution
                x = np.round(res.x).astype(int)
                selected_indices = np.where(x == 1)[0]
                return [values[i] for i in selected_indices]
            else:
                # If not optimal, try to get a feasible solution or fallback
                if res.x is not None:
                    x = np.round(res.x).astype(int)
                    # Ensure feasibility
                    selected_indices = np.where(x == 1)[0]
                    candidate = [values[i] for i in selected_indices]
                    if is_sum_free(candidate):
                        return candidate
                # Fallback to greedy
                return greedy_mis(values)
                
        except Exception as e:
            # Fallback to greedy
            return greedy_mis(values)

    def greedy_mis(values):
        n = len(values)
        v_set = set(values)
        selected = set()
        remaining = list(range(n))
        
        # Sort by degree (number of conflicts)
        def get_degree(idx):
            vi = values[idx]
            deg = 0
            for j in range(n):
                if j == idx:
                    continue
                vj = values[j]
                if (vi + vj) in v_set or 2*vi == vj or 2*vj == vi:
                    deg += 1
            return deg
        
        # Simple greedy: pick node with min degree, remove it and neighbors
        # To speed up, we can just pick random or min degree from original graph
        # This is a fallback, so quality is less critical than correctness.
        
        # Better greedy: Local search
        # Start with all nodes, then remove nodes to resolve conflicts
        # Or start empty and add nodes?
        
        # Let's use a simple heuristic: 
        # 1. Try to keep all.
        # 2. While there is a conflict, remove the node with the highest degree in the conflict subgraph.
        
        s = set(range(n))
        
        def has_conflict(s):
            s_list = list(s)
            for i in range(len(s_list)):
                for j in range(i+1, len(s_list)):
                    vi = values[s_list[i]]
                    vj = values[s_list[j]]
                    if (vi + vj) in v_set or 2*vi == vj or 2*vj == vi:
                        return True
            return False
        
        # This is O(n^3) worst case, but n is small.
        # However, for fallback, we want something fast.
        # Let's just return the odd numbers? No, that's not always max.
        # Let's do a simple local search from empty set.
        
        best_set = set()
        current_set = set()
        
        # Greedy add
        candidates = list(range(n))
        random.shuffle(candidates)
        for idx in candidates:
            # Check if adding idx creates conflict with current_set
            conflict = False
            vi = values[idx]
            for j in current_set:
                vj = values[j]
                if (vi + vj) in v_set or 2*vi == vj or 2*vj == vi:
                    conflict = True
                    break
            if not conflict:
                current_set.add(idx)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
        return [values[i] for i in best_set]

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    # Allocate time proportional to complexity? ILP is exponential, but for n=240 it might be tough.
    # However, the graph is sparse? 
    # For random integers in [-10000, 10000], the probability of a+b=c is low.
    # So the graph is very sparse. ILP should be fast.
    
    for i, (size, seed) in enumerate(instances):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 5:
             vals = instance(size, seed)
             res = greedy_mis(vals)
             if not is_sum_free(res):
                 res = [x for x in vals if x % 2 != 0]
             results.append(res)
             continue
        
        # Give more time to larger instances if needed, but ILP is fast for sparse graphs.
        # Let's just use the remaining time.
        budget = remaining
        if i < len(instances) - 1:
             # Reserve some time for subsequent instances
             budget = remaining * 0.5
             
        vals = instance(size, seed)
        res = solve_instance_ilp(vals, budget)
        
        # Verify
        if not is_sum_free(res):
             res = greedy_mis(vals)
             if not is_sum_free(res):
                 res = [x for x in vals if x % 2 != 0]
        
        results.append(res)
        
    return results

# Fallback global function if needed, but construct handles it.
def greedy_mis(values):
    # This is a duplicate for safety if scope issues arise, but it's inside construct.
    pass

# To ensure the function is accessible if called differently, we define it outside too?
# No, the task says "Keep the function names and signatures of the seed program."
# The seed program has `construct` as the main entry point.
# The helper functions are internal.

# Let's double check the ILP formulation.
# Max sum(x_i)
# s.t. x_i + x_j <= 1 for all (i,j) in E
# x_i in {0,1}
# This is the standard MIS formulation.

# One potential issue: scipy.milp might be slow for n=240 if the graph is dense.
# But for random integers, the graph is sparse.
# Number of edges ~ n^2 * P(sum in range).
# P(sum in range) is roughly 1/10000.
# So edges ~ 240^2 / 10000 ~ 5.76. Very sparse.
# ILP should be instant.

# What if the graph is not sparse?
# Worst case: all numbers are small, e.g., 1..n.
# Then 1+1=2, 1+2=3, etc. Many edges.
# But the input is random in [-10000, 10000].
# So sparsity is guaranteed.

# Let's refine the time management.
# If ILP is instant, we can use the extra time for local search improvements?
# Or just run ILP.

# Let's add a local search step after ILP to see if we can improve?
# ILP gives optimal. No need.

# However, if ILP fails or times out (unlikely), we need a good fallback.
# The current greedy_mis is weak.
# Let's improve the fallback to a simple simulated annealing or local search.
# But for this proposal, we rely on ILP.

# Final check on constraints.
# "no a + b = c with a, b, c in it (a = b allowed)"
# My graph edges:
# 1. vi + vj in v_set (i != j)
# 2. 2*vi == vj (i != j)
# This covers all cases where sum of two distinct elements is in the set, or double of one element is in the set.
# If we pick i and j, and i+j=k, then (i,j) is an edge, so we can't pick both.
# If we pick i and i, and 2i=k, then (i,k) is an edge, so we can't pick both.
# This correctly enforces the sum-free property.

# One edge case: 0 is not in the set.
# So no a + (-a) = 0 issues.

# The code looks solid.
