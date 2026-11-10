"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 128
    # Map numbers 1..128 to indices 0..127
    # Bitset representation: Python integers are arbitrary precision, 
    # so we can use int as a bitset of size 128.
    
    # Precompute LCMs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = (i * j) // math.gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Identify bad triples: {a, b, c} such that lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We store for each node (index 0..127) a bitset of nodes it is connected to in a bad triple
    # Actually, for MIS in hypergraph, we can reduce to graph MIS by adding edges for each hyperedge?
    # No, that's not correct. A hyperedge constraint is "not all 3 are selected".
    # This is a 3-SAT-like constraint or hitting set.
    # However, if we have a bad triple {a,b,c}, we cannot select all three.
    # This is equivalent to finding a maximum independent set in a graph where we add edges?
    # No, adding edges between all pairs in a bad triple would forbid selecting any pair, which is too strong.
    # We need to solve Maximum 3-Set Packing or similar? No, we want to MAXIMIZE selected items such that NO bad triple is fully contained.
    # This is the "Maximum Independent Set in a 3-uniform Hypergraph" problem.
    
    # Let's use a branch and bound approach on the hypergraph.
    # State:
    # - available: bitset of nodes that can still be added
    # - current_set: bitset of nodes currently in the set
    # - current_size: count of nodes in current_set
    
    # Optimization:
    # If a node v is in a bad triple with u, w, and u, w are already in current_set, then v CANNOT be added.
    # So, when we add a node v to current_set, we must remove from 'available' any node x that forms a bad triple with v and some other node in current_set.
    
    # Precompute for each pair (u, v) the list of w such that {u, v, w} is a bad triple.
    # Actually, for a fixed pair (u, v), there might be multiple w's.
    # Let's precompute `bad_partners[u][v]` as a bitset of w's.
    
    bad_partners = [[0]*(n+1) for _ in range(n+1)]
    # Iterate all triples to find bad ones
    # 128 choose 3 is ~350k, which is fine.
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    # {a, b, c} is a bad triple
                    bad_partners[a][b] |= (1 << (c-1))
                    bad_partners[b][a] |= (1 << (c-1))
                    bad_partners[a][c] |= (1 << (b-1))
                    bad_partners[c][a] |= (1 << (b-1))
                    bad_partners[b][c] |= (1 << (a-1))
                    bad_partners[c][b] |= (1 << (a-1))
    
    # Convert to 0-indexed for bit operations
    # Index i corresponds to number i+1
    # Bit i in an integer corresponds to number i+1
    
    # Heuristic for branching:
    # Pick a node with high degree in the "conflict graph" induced by current set?
    # Or just pick the smallest available node?
    # Better: Pick a node that is in the most bad triples with other available nodes?
    # Let's try a simple recursive search with pruning.
    
    # Upper bound: current_size + popcount(available)
    # We can improve this bound.
    
    best_size = 0
    best_set = 0  # Bitset of the best set found
    
    # Initial available set: all nodes 0..127
    full_available = (1 << n) - 1
    
    def popcount(x):
        return bin(x).count('1')
    
    # Branch and Bound
    # To make it faster, we can order the nodes.
    # Let's try to select nodes in a specific order to prune earlier.
    # However, the choice of which node to branch on matters.
    # Let's branch on the node that creates the most conflicts.
    
    # We need to efficiently calculate the new available set when adding a node v.
    # If we add v, we must remove any x in available such that there exists y in current_set 
    # such that {v, x, y} is a bad triple.
    # This means x is in bad_partners[v][y] for some y in current_set.
    # So, forbidden = OR of bad_partners[v][y] for all y in current_set.
    # new_available = available & ~forbidden.
    
    # To speed up the OR, we can precompute nothing more, but we can iterate y in current_set.
    # Since current_set size is at most ~100, this loop is small.
    
    # Let's implement the search.
    # We will use a stack or recursion. Recursion depth 128 is fine.
    
    # To improve speed, we can use a greedy heuristic to get a good initial lower bound.
    # Greedy: iterate 1..128, add if valid.
    greedy_set = 0
    greedy_available = full_available
    for i in range(n):
        v = i
        # Check if v can be added
        # Check against all y in greedy_set
        ok = True
        temp = greedy_set
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            temp ^= lsb
            # Check if {v, y, ?} is bad. 
            # Actually, we need to check if adding v creates a bad triple with two existing nodes?
            # No, the condition is "no three distinct elements".
            # If we add v, we create a bad triple if there are y, z in current_set such that {v, y, z} is bad.
            # Wait, my previous logic "remove x if {v, x, y} is bad" was for preventing future additions.
            # When ADDING v, we must ensure that no pair (y, z) in current_set forms a bad triple with v.
            # i.e., for all y, z in current_set, {v, y, z} is NOT bad.
            
            # How to check this quickly?
            # For a fixed v, the set of y such that {v, y, z} is bad for SOME z in current_set is complex.
            # Easier: For each y in current_set, check if there is a z in current_set such that {v, y, z} is bad.
            # This is O(|current_set|^2) per addition. 128^3 is ~2M operations. Acceptable for greedy.
            
            # Let's just do the naive check for the greedy phase to get a start.
            pass # Placeholder, will implement proper check below
        
        # Let's do a simpler greedy: 
        # Try adding v. Check all pairs (y, z) in current_set.
        # If any pair (y, z) makes {v, y, z} bad, reject.
        
        # We'll just run the B&B directly, but with a good initial bound from a quick greedy.
        pass

    # Quick Greedy for lower bound
    g_set = []
    for i in range(1, n+1):
        ok = True
        # Check all pairs in g_set
        for a_idx in range(len(g_set)):
            for b_idx in range(a_idx+1, len(g_set)):
                a = g_set[a_idx]
                b = g_set[b_idx]
                # Check if {i, a, b} is bad
                if lcm_table[i][a] == lcm_table[i][b] == lcm_table[a][b]:
                    ok = False
                    break
            if not ok: break
        if ok:
            g_set.append(i)
    
    best_size = len(g_set)
    # Convert g_set to bitset
    best_set = 0
    for v in g_set:
        best_set |= (1 << (v-1))
    
    # Now B&B
    # State: available (int), current (int), size (int)
    # We process nodes in order 0..127? 
    # If we force an order (e.g., must decide on node 0, then 1, etc.), it becomes a subset sum style.
    # But we can skip nodes.
    # Standard MIS B&B branches on a node: include or exclude.
    
    # Let's use the "include/exclude" branching on the lowest set bit in available?
    # Or a node with high conflict?
    # Let's try branching on the node with the highest "degree" in the hypergraph induced by available.
    # Degree of v = number of bad triples involving v and two other nodes in available.
    # This calculation is expensive.
    
    # Alternative: Branch on the first available node.
    # This is less efficient but simpler.
    # Let's try to optimize the "include" case.
    # If we include v:
    #   We must remove from available any node x such that there exists y in current 
    #   such that {v, x, y} is bad.
    #   Wait, if we include v, we also must ensure that v doesn't form a bad triple with TWO nodes in current.
    #   If such a pair exists, we CANNOT include v.
    
    # So, before branching on v:
    # 1. Check if v is "safe" to add given current.
    #    Safe means: No pair (y, z) in current such that {v, y, z} is bad.
    # 2. If safe, branch Include:
    #    new_current = current | (1<<v)
    #    new_available = available & ~(1<<v)
    #    Additionally, remove from new_available any x such that {v, x, y} is bad for some y in new_current.
    #    Wait, if we add v, future nodes x must not form a bad triple with v and any y in new_current.
    #    So forbidden_x = OR_{y in new_current} bad_partners[v][y]
    #    new_available &= ~forbidden_x
    # 3. Branch Exclude:
    #    new_available = available & ~(1<<v)
    #    new_current = current
    
    # Pruning:
    # If size + popcount(available) <= best_size, prune.
    
    # Let's implement this.
    
    def is_safe(v, current):
        # Check if adding v creates a bad triple with any pair in current
        # v is index 0..127
        # current is bitset
        # Iterate through y in current
        # For each y, check if there is z in current such that {v, y, z} is bad
        # {v, y, z} is bad iff z is in bad_partners[v+1][y+1]
        # So, for a fixed y, the set of z that make it bad is bad_partners[v+1][y+1]
        # We need to check if (current & bad_partners[v+1][y+1]) != 0
        # If yes, then there is a z in current that forms a bad triple with v and y.
        
        temp = current
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            temp ^= lsb
            
            # Check if any z in current is a bad partner of (v, y)
            # bad_partners is 1-indexed
            if (current & bad_partners[v+1][y+1]) != 0:
                return False
        return True

    def solve(available, current, size):
        nonlocal best_size, best_set
        
        # Time check
        if time.time() > deadline:
            return
        
        # Prune
        if size + popcount(available) <= best_size:
            return
        
        if available == 0:
            if size > best_size:
                best_size = size
                best_set = current
            return
        
        # Select a node to branch on
        # Heuristic: pick a node that is in many bad triples with available nodes?
        # For now, pick the lowest set bit.
        # Or pick a node that is "critical".
        # Let's pick the node with the smallest index for determinism and simplicity first.
        # v_idx = (available & -available).bit_length() - 1
        
        # Better heuristic: Pick a node v that maximizes the number of nodes it would remove from available if included.
        # This is the "degree" in the conflict graph.
        # Let's just pick the first one for now, it might be slow.
        # Let's try to pick a node that is part of a bad triple with two available nodes.
        
        # Let's stick to lowest set bit for now to ensure correctness and see if it's fast enough.
        v_lsb = available & -available
        v_idx = v_lsb.bit_length() - 1
        
        # Branch 1: Exclude v
        solve(available ^ v_lsb, current, size)
        
        # Branch 2: Include v
        # Check safety
        if is_safe(v_idx, current):
            new_current = current | v_lsb
            # Calculate forbidden nodes for future
            forbidden = 0
            # Iterate y in new_current
            temp = new_current
            while temp:
                lsb = temp & -temp
                y = lsb.bit_length() - 1
                temp ^= lsb
                forbidden |= bad_partners[v_idx+1][y+1]
            
            # Remove v and forbidden from available
            new_available = available ^ v_lsb
            new_available &= ~forbidden
            
            # Note: new_available might contain nodes that are no longer available due to previous steps, 
            # but we passed 'available' which was already filtered.
            
            solve(new_available, new_current, size + 1)
            
    # Start search
    # We can order the nodes to improve performance.
    # If we sort nodes by some heuristic, we can't just use bit operations easily on the "next" node.
    # The standard MIS algorithm usually works on a graph. Here we have a hypergraph.
    # The logic above assumes we process nodes in a fixed order (0 to 127).
    # This is valid.
    
    solve(full_available, 0, 0)
    
    # Convert best_set back to list
    result = []
    for i in range(n):
        if best_set & (1 << i):
            result.append(i + 1)
            
    if not result:
        return [1] # Fallback
        
    return result

# EVOLVE-BLOCK-END
