import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin for 90s wall-clock

    n = 256

    # 1. Build the conflict graph.
    # Two numbers x and y are "conflict neighbors" if there exists a z in [1, n]
    # such that {x, y, z} forms a non-constant 3-term arithmetic progression.
    # This is equivalent to:
    # - z = 2y - x is in [1, n] and z != x, z != y
    # - z = 2x - y is in [1, n] and z != x, z != y
    # - z = x + y is in [1, n] (if x and y are endpoints, middle is (x+y)/2, which requires x+y even)
    #   Actually, if x, y are endpoints, the middle is (x+y)/2. For this to be an integer in [1, n],
    #   x+y must be even and 2 <= x+y <= 2n.
    #   Wait, the definition of 3-AP {a, b, c} is a+c=2b.
    #   If we pick two numbers x, y, they conflict if they can be part of a 3-AP with a third number z.
    #   Case 1: x, y are adjacent in the AP. Then z = 2y - x or z = 2x - y.
    #   Case 2: x, y are endpoints. Then z = (x+y)/2. This requires x+y even.
    
    # For Max Weight Independent Set (MWIS) on graph G:
    # We want to find a subset S of vertices such that no two vertices in S are adjacent in G.
    # If x and y are in S, they must NOT form a 3-AP with any other number z in [1, n].
    # This condition is slightly stronger than just "x and y are not adjacent in the 3-AP conflict graph" 
    # because the conflict graph G we build above represents "x and y cannot be in the same 3-AP-free set 
    # IF the third element is also in the set".
    # Wait. The constraint is: "without any nonconstant k=3 term arithmetic progression".
    # This means for ANY three distinct elements a, b, c in S, it must NOT be that a+c=2b.
    # This is a 3-uniform hypergraph constraint.
    # MWIS is for graphs (2-uniform).
    # However, we can transform this.
    # If we define a graph G where x and y are adjacent if they *cannot* coexist in any 3-AP-free set?
    # No, x and y CAN coexist unless they are part of a 3-AP with a third element z that is ALSO in the set.
    # So the constraint is: For all distinct x, y, z in S, x+z != 2y.
    # This is not a simple pairwise constraint. It's a 3-way constraint.
    # Therefore, standard MWIS on a graph is not directly applicable unless we do something else.
    
    # However, the prompt suggests: "Model the problem as a Maximum Weight Independent Set on a graph by introducing auxiliary nodes for each arithmetic progression to enforce the constraint that no three terms are selected".
    # This is a reduction to MWIS on a larger graph?
    # Or maybe the "graph" is the conflict graph where edges represent "if I pick x, I cannot pick y"?
    # If I pick x, I can pick y UNLESS there exists a z such that {x,y,z} is a 3-AP. But z might not be picked.
    # So x and y are not strictly forbidden from coexisting.
    # This means the problem is NOT a standard MWIS on a fixed graph. It's a Max Weight Independent Set in a 3-uniform hypergraph.
    
    # BUT, there is a known trick.
    # If we view the problem as finding a maximal subset with no 3-AP, we can use branch and bound on the hypergraph.
    # The "inspiration" suggests using MWIS with auxiliary nodes. This usually refers to converting the 3-uniform constraint into 2-uniform constraints via a transformation, but that's complex.
    # Alternatively, maybe the "graph" is the complement of the "3-AP conflict" relation?
    # Let's re-read carefully: "introducing auxiliary nodes for each arithmetic progression".
    # This sounds like a vertex cover or independent set formulation on a bipartite-like structure?
    
    # Actually, a simpler structural change from the current DFS (which is essentially a tree search on variables 1..n) is to use a **Maximum Clique** approach on the **Complement of the Conflict Graph** if we can define a valid conflict graph.
    # But since the constraint is 3-wise, we cannot simply define a pairwise conflict graph that captures the full constraint.
    # If x and y are in the set, we must ensure no z completes the AP.
    # This is equivalent to: The set S is an independent set in the 3-uniform hypergraph H where edges are the 3-APs.
    
    # How to solve Max Weight Independent Set in a 3-uniform hypergraph?
    # Standard Branch and Bound (like the current one) is the way.
    # The current program does DFS with B&B.
    # The proposed strategy is "Maximum Weight Independent Set on a graph... with maximum clique reduction and pivot-based branching".
    # This implies we should treat it as a Graph problem. Is it possible to reduce it?
    # Note: A set with no 3-AP is also known as a cap set (in 1D, it's just a 3-AP-free set).
    # There is no simple pairwise graph reduction.
    
    # HOWEVER, we can still use the *algorithmic techniques* of Maximum Clique (Tomita's algorithm) which are superior to simple DFS.
    # Tomita's algorithm for Maximum Clique uses:
    # 1. Pivoting to select the branching vertex.
    # 2. Coloring bound (greedy coloring of the candidate set to provide an upper bound on the clique size).
    # 
    # Can we adapt this?
    # We want to maximize weight.
    # We can model the search as finding a maximum weight clique in a graph G' where G' is the **compatibility graph**.
    # Two vertices x, y are adjacent in G' if they are "compatible".
    # But compatibility is not transitive and not pairwise static.
    # If x and y are compatible, and y and z are compatible, x and z might not be.
    # More importantly, x and y are compatible ONLY IF the third element of their AP is NOT in the set.
    # So the graph G' depends on the set S.
    
    # This means we cannot pre-compute a static graph G'.
    # Therefore, "Maximum Clique on a static graph" is not directly applicable.
    
    # Let's reconsider the "auxiliary nodes" hint.
    # Maybe the strategy is to use **Integer Linear Programming** (ILP) via `scipy`?
    # SciPy is available.
    # We can formulate this as an ILP:
    # Maximize sum(1/i * x_i)
    # Subject to:
    # x_i in {0, 1}
    # For every 3-AP {a, b, c}: x_a + x_b + x_c <= 2
    # 
    # This is a valid formulation!
    # We can use `scipy.optimize.milp` (if available in recent SciPy) or `scipy.optimize.linprog` with integer constraints?
    # `scipy.optimize.linprog` supports `method='highs'` which can handle integer variables?
    # Actually, `scipy.optimize.milp` was added in SciPy 1.9.
    # Let's check if `scipy.optimize.milp` is available.
    # If so, this is a MASSIVELY different strategy: using a state-of-the-art MILP solver (HiGHS) instead of custom Python B&B.
    # HiGHS is extremely fast and effective for such problems.
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        # This is the new strategy: MILP formulation
        # Variables: x_1, ..., x_256
        # Objective: maximize sum(1/i * x_i) -> minimize -sum(1/i * x_i)
        
        c = [-1.0 / i for i in range(1, n + 1)]
        
        # Constraints:
        # 1. x_i >= 0 (handled by bounds)
        # 2. x_i <= 1 (handled by bounds)
        # 3. For each 3-AP (a, b, c): x_a + x_b + x_c <= 2
        
        # Generate all 3-APs
        aps = []
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                c_val = 2 * b - a
                if c_val <= n and c_val > b:
                    aps.append((a - 1, b - 1, c_val - 1))
        
        # Build A matrix for constraints
        # Number of constraints = len(aps)
        # We need a sparse matrix A such that A @ x <= b
        # For each AP (i, j, k), row is 1 at i, j, k
        
        import scipy.sparse as sp
        
        rows = []
        cols = []
        data = []
        
        for idx, (i, j, k) in enumerate(aps):
            rows.extend([idx, idx, idx])
            cols.extend([i, j, k])
            data.extend([1, 1, 1])
        
        A = sp.csr_matrix((data, (rows, cols)), shape=(len(aps), n))
        b = [2] * len(aps)
        
        constraints = LinearConstraint(A, ub=b)
        integrality = [1] * n
        bounds = Bounds(lb=0, ub=1)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            # Due to floating point, threshold
            elements = [int(i) + 1 for i in range(n) if x[i] > 0.5]
            
            # Verify
            elem_set = set(elements)
            for a in elements:
                for b in elements:
                    if a == b: continue
                    c_val = 2 * b - a
                    if c_val in elem_set and c_val != a and c_val != b:
                        # Invalid, fall back to greedy
                        break
                else:
                    continue
                break
            else:
                return elements
            
    except Exception:
        pass

    # Fallback: Improved Branch and Bound with better heuristics
    # Since the MILP might fail or not be available, we implement a robust B&B.
    # Strategy: Use a greedy coloring bound similar to Maximum Clique algorithms, adapted for weighted sets.
    # Or simply use a very fast heuristic + local search.
    
    # Let's implement a strong local search (Simulated Annealing) as the fallback,
    # which is structurally different from the current DFS B&B.
    
    import numpy as np
    
    def can_add(x, S):
        # S is a set of ints
        # Check if adding x creates a 3-AP
        # For each y in S, check if (x+y)/2 is in S, or 2y-x is in S, or 2x-y is in S
        # More efficiently:
        # If we add x, we need to ensure no 3-AP involving x and two existing elements.
        # Let existing elements be y, z.
        # 1. x is middle: y+z = 2x -> z = 2x-y. If 2x-y in S, conflict.
        # 2. x is endpoint: 
        #    a) x, y are endpoints, middle m = (x+y)/2. If m in S, conflict.
        #    b) y, x are endpoints, middle m = (y+x)/2. Same.
        #    c) y is middle, x and z are endpoints. x+z=2y -> z = 2y-x. If 2y-x in S, conflict.
        
        # So for each y in S:
        # - if 2*x - y in S: conflict (x is endpoint, y is middle, other endpoint 2x-y)
        # - if 2*y - x in S: conflict (y is endpoint, x is middle? No. 2y-x = z => x, y, z with y middle? No.
        #   Let's check: AP is a, b, c. a+c=2b.
        #   If x is in the AP, it can be a, b, or c.
        #   Case 1: x is middle (b). Then a+c=2x. We have y in S. If y is a, then c=2x-y. If c in S, conflict.
        #   Case 2: x is endpoint (a). Then c=2b-x. We have y in S.
        #     Subcase 2a: y is middle (b). Then c=2y-x. If c in S, conflict.
        #     Subcase 2b: y is other endpoint (c). Then b=(x+y)/2. If (x+y)/2 in S, conflict.
        
        # So for each y in S:
        # 1. if (2*x - y) in S: conflict (x middle, y endpoint)
        # 2. if (2*y - x) in S: conflict (y middle, x endpoint)
        # 3. if (x + y) % 2 == 0 and ((x + y) // 2) in S: conflict (x,y endpoints)
        
        for y in S:
            if (2 * x - y) in S:
                return False
            if (2 * y - x) in S:
                return False
            if (x + y) % 2 == 0:
                if ((x + y) // 2) in S:
                    return False
        return True

    # Greedy initialization
    S = set()
    for x in range(1, n + 1):
        if can_add(x, S):
            S.add(x)
    
    best_S = set(S)
    best_score = sum(1.0 / x for x in S)
    
    # Local Search: Swap moves
    # Try to remove one element and add another (or more) to improve score.
    # Since we want to maximize sum of reciprocals, we prefer small numbers.
    # However, small numbers have more conflicts.
    
    current_S = set(S)
    current_score = best_score
    
    # Precompute conflicts for speed?
    # Let's just run a simple hill climbing with random restarts if time permits.
    
    time_limit = 50.0
    start_ls = time.time()
    
    while time.time() - start_ls < time_limit:
        # Randomly pick an element to remove
        if not current_S:
            break
        removed = random.choice(list(current_S))
        current_S.remove(removed)
        
        # Try to add elements in order of 1/x (ascending x)
        improved = False
        for x in range(1, n + 1):
            if x in current_S:
                continue
            if can_add(x, current_S):
                # Check if adding x improves score
                # Since we are replacing 'removed' with potentially multiple 'x's?
                # No, this is a "remove one, add greedy" move.
                # It's better to do "remove one, add one" swap or "remove one, add set".
                # Let's stick to "remove one, then greedily add as many as possible from remaining candidates".
                # This is a "kick".
                pass
        
        # Actually, a simpler move:
        # Pick a random element r in S to remove.
        # Pick a random element a not in S to add.
        # If valid, accept if score improves.
        
        # Let's do a simple swap hill climb.
        if time.time() - start_ls > time_limit:
            break
            
        # Pick random remove
        if not current_S:
            break
        r = random.choice(list(current_S))
        
        # Pick random add
        candidates = [x for x in range(1, n + 1) if x not in current_S]
        if not candidates:
            break
        a = random.choice(candidates)
        
        # Try swap
        new_S = set(current_S)
        new_S.remove(r)
        
        if can_add(a, new_S):
            new_S.add(a)
            new_score = sum(1.0 / x for x in new_S)
            if new_score > current_score:
                current_S = new_S
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(current_S)
    
    return list(best_S)

# Note: The MILP approach is the primary structural change.
# If scipy.optimize.milp is not available or fails, the fallback is a local search.
# The current program is a DFS B&B. This is a MILP solver + Local Search.
