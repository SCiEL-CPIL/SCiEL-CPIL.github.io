"""Construct a C5-free graph with alpha < 9, maximizing n."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def has_c5(adj, n):
        """Check if graph contains a 5-cycle."""
        for v in range(n):
            nb = list(adj[v])
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    a, b = nb[i], nb[j]
                    if a == b:
                        continue
                    # Check if a and b have a common neighbor other than v
                    # that forms a 5-cycle: v-a-x-b-y-v
                    # Actually need: v connected to a and b, a connected to x,
                    # b connected to y, x connected to y, and x,y != v,a,b
                    # Let's do it more carefully
                    common_ab = adj[a] & adj[b]
                    for x in common_ab:
                        if x == v:
                            continue
                        # Now need y such that y ~ b, y ~ x, y != v, a, b
                        # and the cycle is v-a-x-y-b-v (length 5)
                        common_xy = adj[x] & adj[b]
                        for y in common_xy:
                            if y in (v, a, b, x):
                                continue
                            # Check v-y edge (should be there since y~b and we need cycle)
                            # Actually cycle is v-a-x-y-b-v, need edge b-v (yes) and y-b (yes)
                            # and v-a (yes), a-x (yes), x-y (yes)
                            # So we have a 5-cycle!
                            return True
        return False

    def has_independent_9(adj, n):
        """Check if graph has an independent set of size 9."""
        # Use backtracking with pruning
        # Greedy coloring bound: if we can't find 9 independent, return False
        # Simple approach: try to find 9 mutually non-adjacent vertices
        def backtrack(remaining, chosen, adj_set):
            if len(chosen) == 9:
                return True
            if not remaining:
                return False
            # Upper bound: even if we take all remaining, can we reach 9?
            if len(chosen) + len(remaining) < 9:
                return False
            v = remaining[0]
            # Option 1: include v
            new_chosen = chosen + [v]
            # Vertices that conflict with v (adjacent to v) can't be chosen
            new_remaining = [u for u in remaining[1:] if u not in adj_set[v]]
            if backtrack(new_remaining, new_chosen, adj_set):
                return True
            # Option 2: exclude v
            if backtrack(remaining[1:], chosen, adj_set):
                return True
            return False

        # Convert to set for faster lookup
        adj_set = [set(adj[i]) for i in range(n)]
        return backtrack(list(range(n)), [], adj_set)

    def validate(adj, n):
        """Check both constraints exactly."""
        if has_c5(adj, n):
            return False
        if has_independent_9(adj, n):
            return False
        return True

    def greedy_construct(seed, target_n, adj_init=None):
        """
        Greedily build a graph by adding vertices one at a time.
        Each new vertex is connected to some existing vertices.
        We maintain: no C5, alpha < 9.
        
        Strategy: Start with a base, then try to add vertices.
        For each new vertex, decide its adjacency to existing vertices.
        """
        if adj_init is None:
            adj = [set() for _ in range(target_n)]
            n = 0
        else:
            adj = [set(s) for s in adj_init]
            n = len(adj)

        random.seed(seed)
        
        # Start with a small valid base (e.g., a few K4s)
        # Actually, let's start from scratch and build up
        adj = [set() for _ in range(target_n)]
        n = 0
        
        for new_v in range(target_n):
            if time_left() < 5.0:
                break
            if n == 0:
                n = 1
                continue
            # Decide which existing vertices to connect to new_v
            # We want to be "dense enough" to prevent independent 9-sets
            # but "sparse enough" to avoid C5
            # 
            # Heuristic: connect to a random subset, then check validity
            # Try multiple random adjacency patterns
            best_pattern = None
            best_score = -1
            
            for attempt in range(50):
                if time_left() < 5.0:
                    break
                # Randomly decide adjacency to each existing vertex
                pattern = set()
                for v in range(n):
                    if random.random() < 0.5:
                        pattern.add(v)
                
                # Check if adding new_v with this pattern creates a C5
                # A C5 involving new_v: new_v - a - b - c - d - new_v
                # where a,d are neighbors of new_v, and a-b-c-d is a path of length 3
                # with a and d both adjacent to new_v
                creates_c5 = False
                for a in pattern:
                    for d in pattern:
                        if a == d:
                            continue
                        # Need path a - x - y - d where x,y not in {new_v, a, d}
                        # and x != d, y != a, x != y
                        for x in adj[a]:
                            if x == d or x == new_v:
                                continue
                            for y in adj[x]:
                                if y in (a, d, x, new_v):
                                    continue
                                if d in adj[y]:
                                    creates_c5 = True
                                    break
                            if creates_c5:
                                break
                        if creates_c5:
                            break
                    if creates_c5:
                        break
                
                if creates_c5:
                    continue
                
                # Check independence number constraint
                # After adding new_v, does the graph have an independent 9-set?
                # Quick check: what's the max independent set size?
                # If new_v is added, independent sets either include it or not.
                # If include new_v: can't include its neighbors. Max independent in G - N(new_v) - {new_v} + 1
                # This is complex; let's just do a quick greedy bound
                # 
                # For now, just check if the current graph (without new_v) has alpha >= 9
                # and whether adding new_v helps or hurts
                # 
                # Simple approach: check if adding this vertex keeps alpha < 9
                # We'll do a quick check by seeing if there's an independent set of 9
                # that includes new_v: new_v + 8 independent from non-neighbors
                # Or an independent set of 9 not including new_v: already checked for old graph
                
                # Quick alpha check for the new graph
                # Build temporary adjacency
                temp_adj = [set(adj[i]) for i in range(n)]
                temp_adj.append(set())
                for v in pattern:
                    temp_adj[v].add(n)
                    temp_adj[n].add(v)
                
                if has_independent_9(temp_adj, n + 1):
                    continue
                
                # Score: prefer denser connections (helps reduce alpha)
                score = len(pattern)
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
            
            if best_pattern is not None:
                for v in best_pattern:
                    adj[v].add(new_v)
                    adj[new_v].add(v)
                n += 1
            else:
                # Try with no connections (isolated vertex) - but this hurts alpha
                # Or try minimal connections
                # Let's try connecting to just 1 random vertex
                if n >= 1:
                    v = random.randint(0, n - 1)
                    adj[v].add(new_v)
                    adj[new_v].add(v)
                    # Quick check
                    temp_adj = [set(adj[i]) for i in range(n + 1)]
                    if has_independent_9(temp_adj, n + 1) or has_c5(temp_adj, n + 1):
                        # Revert
                        adj[v].discard(new_v)
                        adj[new_v].discard(v)
                    else:
                        n += 1
        
        return adj, n

    # Try multiple seeds and approaches
    best_n = 0
    best_adj = None
    
    # Approach 1: Greedy construction
    for seed in range(100):
        if time_left() < 10.0:
            break
        adj, n = greedy_construct(seed, 35)
        if n > best_n:
            # Validate
            if validate(adj, n):
                best_n = n
                best_adj = [set(s) for s in adj]
            if best_n >= 33:
                break
    
    # Approach 2: Start from 8 K4s and try to add vertices
    if time_left() > 15.0:
        # 8 disjoint K4s
        adj = [set() for _ in range(32)]
        for b in range(8):
            for i in range(4):
                for j in range(i + 1, 4):
                    u = 4 * b + i
                    v = 4 * b + j
                    adj[u].add(v)
                    adj[v].add(u)
        
        if validate(adj, 32):
            if 32 > best_n:
                best_n = 32
                best_adj = [set(s) for s in adj]
        
        # Try to add vertices to the 8-K4 construction
        for extra_seed in range(50):
            if time_left() < 10.0:
                break
            random.seed(extra_seed)
            adj2 = [set(s) for s in adj]
            n2 = 32
            
            for new_v in range(32, 35):
                if time_left() < 5.0:
                    break
                # Try to connect new_v to some existing vertices
                found = False
                for attempt in range(100):
                    if time_left() < 5.0:
                        break
                    # Connect to random subset
                    pattern = set()
                    for v in range(n2):
                        if random.random() < 0.3:
                            pattern.add(v)
                    
                    # Check C5
                    creates_c5 = False
                    for a in pattern:
                        for d in pattern:
                            if a == d:
                                continue
                            for x in adj2[a]:
                                if x == d:
                                    continue
                                for y in adj2[x]:
                                    if y in (a, d, x):
                                        continue
                                    if d in adj2[y]:
                                        creates_c5 = True
                                        break
                                if creates_c5:
                                    break
                            if creates_c5:
                                break
                        if creates_c5:
                            break
                    
                    if creates_c5:
                        continue
                    
                    # Check alpha
                    temp_adj = [set(adj2[i]) for i in range(n2)]
                    temp_adj.append(set())
                    for v in pattern:
                        temp_adj[v].add(n2)
                        temp_adj[n2].add(v)
                    
                    if has_independent_9(temp_adj, n2 + 1):
                        continue
                    
                    # Valid!
                    for v in pattern:
                        adj2[v].add(n2)
                        adj2[n2].add(v)
                    n2 += 1
                    found = True
                    break
                
                if not found:
                    break
            
            if n2 > best_n:
                if validate(adj2, n2):
                    best_n = n2
                    best_adj = [set(s) for s in adj2]
                if best_n >= 34:
                    break
    
    # Fallback: 8 disjoint K4s (n=32)
    if best_adj is None or best_n < 32:
        adj = [set() for _ in range(32)]
        for b in range(8):
            for i in range(4):
                for j in range(i + 1, 4):
                    u = 4 * b + i
                    v = 4 * b + j
                    adj[u].add(v)
                    adj[v].add(u)
        best_n = 32
        best_adj = adj
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in sorted(adj[i]):
            if i < j:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
# EVOLVE-BLOCK-END
