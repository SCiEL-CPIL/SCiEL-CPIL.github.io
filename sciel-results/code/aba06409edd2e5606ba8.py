import time
import random
import sys
from itertools import combinations

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency matrix as list of sets or bitmasks
    # Using bitmasks for fast operations
    adj = [0] * n
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
    
    def get_cliques_4():
        """Find all K4 and Independent K4 (i.e., K4 in complement)"""
        k4s = []
        ik4s = []
        # A K4 is a set of 4 vertices where all pairs are connected.
        # We can find them by iterating over pairs and finding common neighbors that form triangles?
        # Simpler: Iterate all 4-combinations? n=37, C(37,4) = 66045. This is fast.
        # As n grows to 40, C(40,4) = 91390. Still fast.
        # As n grows to 42, C(42,4) = 111930. Still fast.
        
        for quad in combinations(range(n), 4):
            is_k4 = True
            is_ik4 = True
            for u, v in combinations(quad, 2):
                if adj[u] & (1 << v):
                    is_ik4 = False
                else:
                    is_k4 = False
            if is_k4:
                k4s.append(quad)
            if is_ik4:
                ik4s.append(quad)
        return k4s, ik4s

    # Initial constraints
    k4s, ik4s = get_cliques_4()
    
    best_n = n
    best_edges = edges
    
    # Helper to check if a partial assignment is valid for the CSP
    # We will use a simple backtracking solver for the new vertex
    
    def try_add_vertex(current_n, current_adj, current_edges, k4s, ik4s):
        """
        Attempts to add a new vertex with index current_n.
        Returns (success, new_adj, new_edges, new_k4s, new_ik4s)
        """
        # Constraints for the new vertex v
        # For each k4, sum(x_u for u in k4) <= 3
        # For each ik4, sum(x_u for u in ik4) >= 1
        
        # We will use backtracking.
        # To make it efficient, we can pre-process constraints.
        # But since n is small (<50), simple backtracking with forward checking should work.
        
        # Variables: 0 to current_n - 1
        # x[u] is 1 if connected to u, 0 otherwise.
        
        # We need to find an assignment x[0..current_n-1]
        
        # Heuristic: Order variables by how many constraints they are involved in?
        # Or just random order to allow multiple tries.
        order = list(range(current_n))
        random.shuffle(order)
        
        # Forward checking state
        # For each constraint, track remaining capacity.
        # k4 constraints: cap = 3, used = 0
        # ik4 constraints: cap = 1 (min), used = 0 (this is tricky for backtracking, usually we track remaining required)
        
        # Let's represent constraints as:
        # k4: [list_of_vars, max_sum]
        # ik4: [list_of_vars, min_sum]
        
        cons_k4 = [(list(c), 3) for c in k4s]
        cons_ik4 = [(list(c), 1) for c in ik4s]
        
        # State for backtracking
        # x: dict var -> value
        # For each constraint, we can maintain:
        # k4: remaining_allowed = max_sum - sum_assigned
        # ik4: remaining_required = min_sum - sum_assigned (if negative, satisfied)
        
        # To speed up, we can just check constraints at each step?
        # Number of constraints is ~1000-2000. Checking all is O(C).
        # Depth is n (~40). Total ops ~ 40 * 2000 = 80k. Very fast.
        
        def backtrack(idx, x):
            if idx == len(order):
                # Check if all constraints satisfied
                # We can maintain counts, but checking at end is fine if we prune early
                # However, we should prune during assignment.
                # Let's implement a check function that returns True if partial assignment is consistent.
                # Actually, it's better to maintain the state.
                pass
            
            # This simple recursive structure is hard to maintain state cleanly in Python without global vars or closures with nonlocal.
            # Let's use a simpler approach: Randomized Local Search on the assignment x.
            # Or, since the CSP is small, use a simple greedy + backtracking.
            
            return None

        # Let's use a simpler heuristic: 
        # 1. Start with a random assignment.
        # 2. While invalid, find a violated constraint and flip a variable in it.
        # This is Min-SAT / Local Search for the CSP.
        
        # Initialize x randomly
        x = [random.randint(0, 1) for _ in range(current_n)]
        
        # We need to ensure no K5 and no IK5.
        # Violations:
        # - Any k4 with sum(x) == 4 (K5 formed)
        # - Any ik4 with sum(x) == 0 (IK5 formed)
        
        # Local Search
        max_iter = 5000
        for _ in range(max_iter):
            # Check violations
            viol_k4 = None
            viol_ik4 = None
            
            # Count sums for all constraints
            # Optimization: Only check constraints that might be violated?
            # For now, check all.
            sums_k4 = [sum(x[u] for u in c) for c in k4s]
            sums_ik4 = [sum(x[u] for u in c) for c in ik4s]
            
            for i, s in enumerate(sums_k4):
                if s == 4:
                    viol_k4 = i
                    break
            
            if viol_k4 is None:
                for i, s in enumerate(sums_ik4):
                    if s == 0:
                        viol_ik4 = i
                        break
            
            if viol_k4 is None and viol_ik4 is None:
                # Valid assignment found
                return x
            
            # Flip a variable in the violated constraint
            if viol_k4 is not None:
                # Choose a random variable in the k4 to flip (prefer 1->0 to reduce sum)
                c = k4s[viol_k4]
                # Find a u in c with x[u]==1
                candidates = [u for u in c if x[u] == 1]
                if candidates:
                    u = random.choice(candidates)
                    x[u] = 0
                else:
                    # Should not happen if sum==4
                    u = random.choice(c)
                    x[u] = 0 
            else:
                # viol_ik4 is not None
                c = ik4s[viol_ik4]
                # Find a u in c with x[u]==0
                candidates = [u for u in c if x[u] == 0]
                if candidates:
                    u = random.choice(candidates)
                    x[u] = 1
                else:
                    u = random.choice(c)
                    x[u] = 1
        
        return None

    # Main Loop: Try to add vertices
    # We might get stuck. We can also try to restart with a different random seed or perturb the graph.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Try to add a vertex
        x = try_add_vertex(n, adj, edges, k4s, ik4s)
        if x is not None:
            # Add the vertex
            new_v = n
            adj.append(0)
            for u in range(n):
                if x[u]:
                    adj[new_v] |= (1 << u)
                    adj[u] |= (1 << new_v)
                    edges.append((min(u, new_v), max(u, new_v)))
            
            n += 1
            if n > best_n:
                best_n = n
                best_edges = edges[:]
            
            # Update k4s and ik4s
            # This is expensive if we recalculate from scratch.
            # But n is small, so C(n,4) is manageable.
            k4s, ik4s = get_cliques_4()
            
            # If we reach a high number, we might want to stop or continue
            if n >= 42:
                break
        else:
            # Failed to add vertex.
            # Try a few random restarts of the local search within try_add_vertex?
            # Or perturb the current graph?
            # For now, just break or try a few times before giving up on this branch.
            # To avoid infinite loop, we can limit consecutive failures.
            if attempts % 10 == 0:
                # Perturb: Flip a random edge?
                # This might break validity. We need to check validity after perturbation.
                # Given complexity, let's just stop if we fail a few times.
                # Actually, let's try restarting the local search with different random seeds inside try_add_vertex is not possible without refactoring.
                # Let's just break if we fail 5 times in a row.
                pass
            # To prevent infinite loop, we add a counter
            if attempts > 50:
                break

    return best_n, best_edges

# EVOLVE-BLOCK-END
