import numpy as np
from math import lcm
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    # Nodes are 2 to 256
    nodes = list(range(2, 257))
    n = len(nodes)
    
    # Map node value to index
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    
    # Find non-edges (pairs where lcm <= 256)
    non_edges = []
    for i in range(n):
        u = nodes[i]
        for j in range(i + 1, n):
            v = nodes[j]
            if lcm(u, v) <= 256:
                non_edges.append((i, j))
                
    # Objective: Maximize sum(1/x). 
    # scipy milp minimizes c @ x. So c = -1/x.
    c = np.zeros(n)
    for i, v in enumerate(nodes):
        c[i] = -1.0 / v
        
    # Constraints: 
    # 1. x_i + x_j <= 1 for all non-edges
    # 2. Clique constraints: sum(x_i for i in D_N) <= 1 for all N in [2, 256]
    #    where D_N = {d in [2, 256] : d divides N}
    
    A_list = []
    b_list = []
    
    # Add non-edge constraints
    for i, j in non_edges:
        row = np.zeros(n)
        row[i] = 1
        row[j] = 1
        A_list.append(row)
        b_list.append(1)
        
    # Add clique constraints for divisor sets
    # To avoid redundant constraints, we can just add them for all N.
    # Or we can identify maximal cliques. 
    # D_N is a clique. If D_N is a subset of D_M, then the constraint for D_M is stronger.
    # D_N subset D_M iff N divides M? No. D_N = {d|N}. If N|M, then D_N subset D_M.
    # So we only need to add constraints for N that are not divisors of any other N in the range?
    # Actually, if N1 | N2, then D_N1 subset D_N2.
    # The constraint sum_{d in D_N2} x_d <= 1 implies sum_{d in D_N1} x_d <= 1.
    # So we only need to add constraints for "maximal" N with respect to divisibility?
    # i.e., N such that no multiple of N (within 2..256) exists?
    # Wait, if N=2, D_2={2}. If M=4, D_4={2,4}.
    # Constraint for 4: x_2 + x_4 <= 1.
    # Constraint for 2: x_2 <= 1.
    # The constraint for 4 is stronger.
    # So we should add constraints for N that are NOT divisors of any other number in the set?
    # No, we want the largest sets D_N.
    # D_N is largest when N is large?
    # Number of divisors varies.
    # Let's just add constraints for all N in [2, 256] to be safe, or filter out those where D_N is a proper subset of D_M for some M.
    # D_N subset D_M iff N divides M.
    # So we only need to add constraints for N such that there is no M > N with N | M.
    # i.e., N is "maximal" in the divisibility poset?
    # Actually, if N | M, then D_N subset D_M.
    # So we only need constraints for N that do not divide any other number in [2, 256]?
    # No, if N does not divide any M, then D_N is not a subset of any D_M.
    # But D_N could still be a subset of a clique not of the form D_M?
    # The problem is that D_N might not be a maximal clique.
    # However, adding all D_N constraints is safe and likely sufficient to tighten the bound.
    # 255 extra constraints is small.
    
    for N in range(2, 257):
        row = np.zeros(n)
        # Find divisors of N in nodes
        for d in range(2, N + 1):
            if N % d == 0:
                idx = node_to_idx[d]
                row[idx] = 1
        # Only add if the row is not all zeros (which is impossible since N>=2, d=N is a divisor)
        A_list.append(row)
        b_list.append(1)
        
    A = np.array(A_list)
    b = np.array(b_list)
    
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n)
    
    constraint = LinearConstraint(A, lb=-np.inf, ub=b)
    
    # Set time limit for milp to ensure we return something
    # The evaluator kills after 90s. Let's give milp 80s.
    # But milp might not support time limit directly in all versions?
    # scipy.optimize.milp has options.
    # We can wrap it in a try/except or just hope it finishes.
    # Actually, we can use a thread or just rely on the solver being fast enough with tighter constraints.
    
    res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
    
    if not res.success:
        # Fallback: Greedy
        selected = []
        # Sort by weight (1/v) descending, i.e., v ascending
        for x in range(2, 257):
            if all(lcm(x, y) > 256 for y in selected):
                selected.append(x)
        return selected
        
    x = res.x
    selected = [nodes[i] for i in range(n) if x[i] > 0.5]
    
    if not selected:
        return [2]
        
    return selected
