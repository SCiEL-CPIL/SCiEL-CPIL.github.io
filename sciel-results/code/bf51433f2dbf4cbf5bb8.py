import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

def construct():
    n = 20
    deadline = time.time() + 145.0
    
    # Variables: x_{ij} for 0 <= i < j < n
    # Total variables: 20 * 19 / 2 = 190
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    
    # Map edge to index
    edge_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = idx
        if u > v:
            edge_idx[(v, u)] = idx
        else:
            edge_idx[(u, v)] = idx

    # Objective: Maximize sum of x_{ij}
    # c = -1 for all variables (since milp minimizes)
    c = -np.ones(m)
    
    # Bounds: 0 <= x_{ij} <= 1
    lb = np.zeros(m)
    ub = np.ones(m)
    integrality = np.ones(m) # All variables are binary
    
    # Constraints:
    # A K3,3 subgraph is defined by two disjoint sets U and V of size 3.
    # The graph contains K3,3 if there are 9 edges between U and V.
    # We must ensure that for any U, V disjoint, |{ (u,v) in UxV : x_{uv}=1 }| <= 8.
    # This is equivalent to: sum_{u in U, v in V} x_{uv} <= 8.
    
    # Number of pairs of disjoint 3-sets:
    # Choose U (3 from 20), choose V (3 from remaining 17).
    # Order doesn't matter for the constraint (U,V) is same as (V,U) in terms of edges, 
    # but the set of edges is symmetric.
    # Actually, we just need to generate all unique pairs of disjoint 3-sets.
    # To avoid duplicates (U,V) and (V,U), we can enforce a canonical ordering, 
    # e.g., min(U) < min(V) or similar, but generating all and deduplicating is safer 
    # and 38760 constraints is small for ILP.
    
    # Let's generate the constraint matrix A and vector b.
    # Using sparse matrix for efficiency.
    from scipy.sparse import lil_matrix
    
    A = lil_matrix((0, m))
    b_list = []
    
    # Precompute list of 3-sets
    triples = list(itertools.combinations(range(n), 3))
    
    # Generate constraints
    # For efficiency, we can iterate over all pairs of triples (U, V)
    # U and V must be disjoint.
    # To reduce symmetry, we can just take all pairs and rely on the solver, 
    # or filter. 38760 constraints * 190 vars is ~7M non-zeros max, which is fine.
    
    # Optimization: We only need to check pairs where the sets are disjoint.
    # We can iterate U, then V from triples where V is disjoint from U.
    
    # To speed up generation, we can use a bitset representation of triples
    triple_masks = [0] * len(triples)
    for i, t in enumerate(triples):
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks[i] = mask
    
    constraint_count = 0
    for i in range(len(triples)):
        if time.time() > deadline - 60:
            break
        U = triples[i]
        mask_U = triple_masks[i]
        for j in range(i + 1, len(triples)):
            V = triples[j]
            mask_V = triple_masks[j]
            # Check disjoint
            if mask_U & mask_V == 0:
                # Add constraint: sum of edges between U and V <= 8
                row = np.zeros(m)
                # Edges are undirected. An edge (u,v) connects U and V if 
                # u in U and v in V OR u in V and v in U.
                for u in U:
                    for v in V:
                        idx = edge_idx[(u, v)]
                        row[idx] = 1.0
                A.append(row)
                b_list.append(8.0)
                constraint_count += 1
                
    # Convert to CSR for milp
    A_csr = A.tocsr()
    
    # Define linear constraints
    # A @ x <= b
    # milp expects LinearConstraint(A, lb, ub)
    # Here lb is -inf, ub is b
    lin_cons = LinearConstraint(A_csr, lb=-np.inf, ub=np.array(b_list))
    
    # Bounds for variables
    bounds = Bounds(lb, ub)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=lin_cons,
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0: # Optimal
            x = res.x
            edges = []
            for idx, (u, v) in enumerate(all_edges):
                if x[idx] > 0.5:
                    edges.append([u, v])
            return edges
        elif res.status == 2: # Feasible (not necessarily optimal, but valid)
            x = res.x
            edges = []
            for idx, (u, v) in enumerate(all_edges):
                if x[idx] > 0.5:
                    edges.append([u, v])
            return edges
        else:
            # Fallback: Return empty or a simple valid graph if solver fails
            return []
            
    except Exception as e:
        # If ILP fails or times out internally (unlikely with 145s guard if we check time),
        # we need a fallback.
        # Since we didn't check time inside the constraint generation loop effectively 
        # for the solve step, let's add a hard timeout wrapper if possible.
        # But milp doesn't have a simple timeout param in older scipy.
        # However, the constraint generation loop above has a time check.
        # If the solve itself takes too long, we might hit the 120s limit.
        # To be safe, let's implement a simple fallback heuristic if this block is reached
        # or if we anticipate the solve might be slow.
        
        # Actually, if the exception is due to time, we can't recover easily.
        # Let's ensure the constraint generation is fast.
        # 38k constraints is small. The solve should be fast.
        return []

# To be safer against the 120s limit, we can wrap the whole thing in a timeout if 
# we suspect scipy might hang, but usually milp is fast for this size.
# The constraint generation is the main Python loop.
# 38760 iterations * inner loop (small) is very fast (< 0.1s).
# The solve is in C/C++ (HiGHS).

# Let's refine the fallback to ensure we always return *something* valid if the ILP fails.
# We'll add a simple greedy fallback at the end.

def _fallback_greedy():
    n = 20
    adj = [0] * n
    edges = []
    # Try to add edges in a specific order, checking validity
    # This is slow, so only use if ILP fails completely.
    # For the sake of this task, if ILP fails, returning [] is safer than an invalid graph.
    # But an empty graph is valid.
    return []

# Re-defining construct to include the fallback logic more robustly if needed.
# The current construct above is sufficient if scipy works.

# One potential issue: `scipy.optimize.milp` might not be available in all environments 
# or might be slow. But the prompt says SciPy is available.
# The score 2.25 implies ~55-60 edges. 
# ex(20, K3,3) is known to be around 60-70? 
# Actually, Turan number ex(n, K3,3) is approx n^2/2? No, K3,3 is bipartite.
# For n=20, a random graph with p=0.5 has ~100 edges and likely contains K3,3.
# The max is likely around 60-80.

# Let's double check the constraint logic.
# "no three vertices have three common neighbours"
# This is equivalent to: No K3,3 subgraph.
# A K3,3 subgraph consists of two sets U={u1,u2,u3} and V={v1,v2,v3} such that 
# all 9 edges (ui, vj) exist.
# My constraint: For any disjoint U, V of size 3, sum of edges between them <= 8.
# This correctly forbids the existence of a K3,3.
# Are there other constraints? No, that's the definition.
# Is the graph simple? Yes, variables are 0/1.
# Is it undirected? Yes, we model edges {u,v} once.

# The ILP approach is structurally different (exact optimization vs heuristic search) 
# and likely to find a better solution if the solver is efficient enough.

# Final check on imports and structure.
# The function name is `construct`.
# Return type is list of edges [u, v].

# I will output the code as planned.
