"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # All numbers from 2 to 256
    numbers = list(range(2, 257))
    n = len(numbers)
    
    # Map number to index
    num_to_idx = {num: i for i, num in enumerate(numbers)}
    
    # Precompute conflicts: conflict[i][j] = True if lcm(numbers[i], numbers[j]) <= 256
    # For max weight independent set via max flow, we need to model this as
    # a maximum weight closure problem.
    
    # Strategy: Use the standard reduction from max weight independent set to
    # min cut / max flow.
    # 
    # For each node i with weight w_i > 0:
    #   - Add edge source -> i with capacity w_i
    #   - Add edge i -> sink with capacity 0
    # 
    # For each node i with weight w_i < 0:
    #   - Add edge source -> i with capacity 0
    #   - Add edge i -> sink with capacity -w_i
    # 
    # For each conflict edge (i, j):
    #   - Add edge i -> j with capacity INF
    #   - Add edge j -> i with capacity INF
    # 
    # The max weight independent set = total positive weights - min cut value
    # The nodes in the independent set are those reachable from source in the
    # residual graph after max flow.
    
    # But wait - all weights are positive (reciprocals). So we need a different approach.
    # 
    # For all-positive weights, the problem is: select a subset with no conflicts
    # to maximize sum of weights.
    # 
    # Standard reduction for max weight independent set with all positive weights:
    # Create a graph where:
    #   - Source S, Sink T
    #   - For each node i: edge S->i with capacity w_i
    #   - For each conflict (i,j): edge i->j with capacity INF and j->i with capacity INF
    #   - For each node i: edge i->T with capacity w_i
    # 
    # Hmm, that's not quite right. Let me think more carefully.
    # 
    # Actually, the standard reduction for maximum weight independent set is:
    #   - Create bipartite graph if the original graph is bipartite
    #   - For general graphs, use the following:
    #     * For each vertex v with weight w_v:
    #       * If w_v > 0: add edge (S, v) with capacity w_v
    #       * If w_v < 0: add edge (v, T) with capacity -w_v
    #     * For each edge (u,v) in the conflict graph:
    #       * Add edge (u, v) with capacity INF
    #       * Add edge (v, u) with capacity INF
    #     * The max weight independent set = sum of all positive weights - min cut
    # 
    # Since all our weights are positive, this becomes:
    #   - For each vertex v: add edge (S, v) with capacity w_v
    #   - For each conflict (u,v): add edges (u,v) and (v,u) with capacity INF
    #   - Need to add edges from v to T somehow...
    # 
    # Actually, I think I'm overcomplicating this. Let me use a different approach.
    # 
    # For maximum weight independent set with all positive weights, we can use:
    #   - Create a flow network with source S and sink T
    #   - For each node i: edge S->i with capacity w_i
    #   - For each node i: edge i->T with capacity w_i  
    #   - For each conflict (i,j): edge i->j with capacity INF and j->i with capacity INF
    # 
    # No, that's still not right.
    # 
    # Let me use the correct reduction. The maximum weight independent set problem
    # can be solved by reducing to minimum cut as follows:
    # 
    # Given a graph G = (V, E) with weights w: V -> R (can be positive or negative):
    # 1. Create a flow network with source s and sink t
    # 2. For each vertex v:
    #    - If w_v > 0: add edge (s, v) with capacity w_v
    #    - If w_v < 0: add edge (v, t) with capacity -w_v
    # 3. For each edge (u, v) in E:
    #    - Add edge (u, v) with capacity INF
    #    - Add edge (v, u) with capacity INF
    # 4. Compute max flow / min cut
    # 5. Let (A, B) be the min cut (A contains s, B contains t)
    # 6. The maximum weight independent set is V \ (A ∪ B) \ {s, t}
    #    Wait, that's not right either.
    # 
    # Actually, the correct result is:
    # - Nodes in A (reachable from s in residual graph) that are NOT s form part of the solution
    # - The max weight independent set weight = sum of all positive weights - min_cut_value
    # - The independent set consists of nodes that are on the source side of the cut
    #   but we need to exclude nodes that have negative weight...
    # 
    # I'm getting confused. Let me just implement a simpler exact approach using
    # branch and bound with good pruning, since the current approach already works.
    # 
    # Actually, let me try a different exact approach: since the conflict graph has
    # special structure (lcm-based), let me use the fact that if a | b then lcm(a,b) = b.
    # So a and b conflict iff b <= 256 and a divides b... no wait, that's not right either.
    # lcm(a,b) = a*b/gcd(a,b) <= 256.
    
    # Let me just improve the current branch and bound approach with better pruning.
    
    # Precompute conflicts
    conflicts = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a, b = numbers[i], numbers[j]
            if (a * b) // gcd(a, b) <= 256:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Weights
    weights = [Fraction(1, numbers[i]) for i in range(n)]
    
    # Sort nodes by weight descending (higher weight first)
    node_order = sorted(range(n), key=lambda i: -weights[i])
    
    best_weight = Fraction(0)
    best_set = set()
    
    def branch_and_bound(idx, current_set, forbidden):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
        
        # Upper bound: current weight + sum of all remaining weights not forbidden
        current_w = sum(weights[i] for i in current_set)
        remaining_w = sum(weights[node_order[i]] for i in range(idx, n) 
                        if node_order[i] not in forbidden and node_order[i] not in current_set)
        
        if current_w + remaining_w <= best_weight:
            return
        
        if idx == n:
            if current_w > best_weight:
                best_weight = current_w
                best_set = set(current_set)
            return
        
        node = node_order[idx]
        
        # Option 1: Skip this node
        branch_and_bound(idx + 1, current_set, forbidden)
        
        # Option 2: Add this node if not forbidden
        if node not in forbidden:
            new_forbidden = forbidden | conflicts[node] | {node}
            branch_and_bound(idx + 1, current_set | {node}, new_forbidden)
    
    branch_and_bound(0, set(), set())
    
    if not best_set:
        return [2]
    
    return list(numbers[i] for i in best_set)
# EVOLVE-BLOCK-END
