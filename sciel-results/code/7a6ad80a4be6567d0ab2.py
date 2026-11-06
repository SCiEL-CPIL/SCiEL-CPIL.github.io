import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    
    # Precompute weights (reciprocals)
    w = [0.0] + [1.0 / i for i in range(1, n + 1)]
    
    # Build conflict graph using bitsets for efficiency
    # adj[i] is a bitset where bit j is set if i and j are in a 3-AP
    adj = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            c = 2 * b - a
            if c > n:
                break
            # a, b, c form a 3-AP
            # Conflicts: a-c, b-c (a-b is not a conflict pair in the graph sense for 3-AP free sets? 
            # Wait: A set is 3-AP free if it contains NO {a,b,c} with c=2b-a.
            # The graph we are building is the "conflict graph" where an edge exists between x and y 
            # if there exists a z such that {x,y,z} is a 3-AP.
            # If we pick x and y, we must not pick z.
            # Actually, the standard formulation for "no 3-AP" is an Independent Set in a graph 
            # where edges connect pairs that *cannot* coexist?
            # No. If x and y are in the set, and z is in the set, and they form a 3-AP, it's invalid.
            # This is a 3-uniform hypergraph independent set problem.
            # However, we can map it to a graph problem?
            # Let's look at the structure. 
            # If we include x and y, we forbid z.
            # This is not a simple pairwise conflict graph.
            # BUT, the current seed program uses `can_add` which checks if adding x creates a 3-AP 
            # with existing pairs.
            
            # Let's re-evaluate the problem structure.
            # A set S is 3-AP free iff for all a,b in S, 2b-a is not in S.
            # This is equivalent to: The set S contains no 3-term AP.
            
            # Can we model this as a Max Weight Independent Set on a graph?
            # Consider the graph G where vertices are [1..n].
            # If we define edges such that an edge (u,v) means u and v cannot be in the same set?
            # If u and v are in S, does that forbid some w? Yes, w=2v-u or 2u-v.
            # So u and v are not directly conflicting in the sense of "cannot be together". 
            # They are together, but they forbid a third node.
            # This is a 3-uniform hypergraph.
            
            # However, there is a known reduction or a specific structure for 3-APs.
            # Let's stick to the bitset representation of the set S and use a search algorithm 
            # that handles the 3-uniform constraint directly, or use the fact that 
            # "3-AP free" sets are related to Sidon sets or similar, but simpler.
            
            # Actually, for small n (256), we can use a Branch and Bound on the set S.
            # But the state space is 2^256. We need strong pruning.
            
            # Alternative: The problem is equivalent to finding a Maximum Weight Independent Set 
            # in a specific graph if we transform the problem?
            # No, it's a 3-uniform hypergraph.
            
            # Let's use a different approach: **Recursive Search with Bitset Pruning**.
            # We process numbers from 1 to n.
            # At each step, we decide to include or exclude i.
            # We maintain the current set S (as a bitset) and the "forbidden" set F.
            # If we add x, any y,z in S such that {x,y,z} is a 3-AP makes the set invalid.
            # So, if we add x, we must ensure that for all y in S, 2y-x is not in S and 2x-y is not in S.
            
            # Let's define the search tree.
            # To maximize the sum of reciprocals, we want to include small numbers.
            # We can use a Branch and Bound where the upper bound is the sum of reciprocals 
            # of all remaining numbers that *could* potentially be added.
            
            # Key Insight: 
            # If we include x, it restricts future choices.
            # Specifically, if x and y are in S, then 2y-x and 2x-y are forbidden.
            
            # Let's implement a DFS with the following state:
            # current_index: next number to consider
            # current_set: bitset of chosen numbers
            # current_weight: sum of reciprocals of chosen numbers
            # 
            # Pruning:
            # 1. If time is up, return best found.
            # 2. Upper bound: current_weight + sum(w[i] for i in current_index..n)
            #    If this bound <= best_found, prune.
            # 3. Feasibility: Can we even add any more numbers?
            
            # To make the bound tighter, we can precompute for each i, 
            # which future numbers are "blocked" by i alone? 
            # No, blocking requires pairs.
            
            # Let's try a heuristic Branch and Bound.
            # Order: 1, 2, ..., n.
            # Since w[i] decreases, including early numbers is valuable.
            
            # Optimization: 
            # Instead of checking all pairs in S when adding x, 
            # we can maintain a bitset `blocked` of numbers that cannot be added 
            # due to the current set S.
            # If we add x, we compute new_blocked.
            # new_blocked = blocked | {y in S | 2y-x} | {y in S | 2x-y}
            # Note: 2y-x and 2x-y must be in range [1, n].
            
            # This allows O(1) or O(k) check for feasibility of adding x?
            # No, we still need to check if x is in `blocked`.
            # If x is in `blocked`, we cannot add x.
            
            # So the state is (index, S, blocked, weight).
            # But S and blocked are large bitsets.
            # We can pass them by reference or recreate them? 
            # Recreating is expensive. 
            # Given n=256, bitsets fit in 4 uint64s. 
            # Python integers are arbitrary precision, so bit operations are fast.
            
            # Let's implement this DFS.
            
    # Precompute bit masks for efficiency?
    # We'll use Python ints as bitsets.
    
    best_weight = 0.0
    best_set = 0
    
    # Precompute reciprocal sums for suffixes for quick upper bound
    suffix_sum = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + w[i]
    
    def dfs(idx, S, blocked, weight):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
            
        # Prune if upper bound is not better
        # Upper bound assumes we can add all remaining numbers
        ub = weight + suffix_sum[idx]
        if ub <= best_weight:
            return
            
        if idx > n:
            if weight > best_weight:
                best_weight = weight
                best_set = S
            return
            
        # Option 1: Skip idx
        dfs(idx + 1, S, blocked, weight)
        
        # Option 2: Take idx
        # Check if idx is blocked
        if not (blocked & (1 << (idx - 1))):
            new_S = S | (1 << (idx - 1))
            new_blocked = blocked
            # Update blocked: 
            # For every y in S, 2y - idx and 2idx - y are blocked.
            # We need to iterate through bits in S.
            # To speed this up, we can precompute for each idx, 
            # the list of y such that 2y-idx is in [1,n] and 2idx-y is in [1,n].
            # But S is dynamic.
            
            # Let's iterate through set bits in S.
            # Since n=256, S has at most ~50-100 elements typically?
            # Worst case size of 3-AP free set in [1,256]?
            # Szemeredi's theorem implies o(n). For n=256, max size is likely < 100.
            # Iterating through bits is acceptable.
            
            s_bits = S
            while s_bits:
                lsb = s_bits & -s_bits
                y_idx = lsb.bit_length()
                # y is the number
                y = y_idx
                
                # Calculate 2y - idx
                v1 = 2 * y - idx
                if 1 <= v1 <= n:
                    new_blocked |= (1 << (v1 - 1))
                    
                # Calculate 2idx - y
                v2 = 2 * idx - y
                if 1 <= v2 <= n:
                    new_blocked |= (1 << (v2 - 1))
                
                s_bits ^= lsb
            
            # Also, idx itself is now in S, so it's "used", 
            # but does it block itself? No, but it blocks other pairs.
            # The logic above covers pairs (y, idx).
            
            dfs(idx + 1, new_S, new_blocked, weight + w[idx])
            
    # Start DFS
    dfs(1, 0, 0, 0.0)
    
    # Extract result
    if best_set == 0:
        # Fallback: just return [1]
        return [1]
        
    result = []
    for i in range(1, n + 1):
        if best_set & (1 << (i - 1)):
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
