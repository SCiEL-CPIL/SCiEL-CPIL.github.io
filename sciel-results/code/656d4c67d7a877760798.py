"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    # Fallback: the known K9 construction
    def k9_fallback():
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

    def has_c6_in_colour(colour_edges, n):
        """Check if a set of edges (as a set of frozensets) contains a C6."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for (u, v) in colour_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Look for 6-cycles: a0-a1-a2-a3-a4-a5-a0
        # We need to find 6 distinct vertices forming a cycle
        # For efficiency, check paths of length 5 that close
        for a0 in range(n):
            for a1 in adj[a0]:
                if a1 <= a0:
                    continue
                for a2 in adj[a1]:
                    if a2 == a0:
                        continue
                    for a3 in adj[a2]:
                        if a3 == a0 or a3 == a1:
                            continue
                        for a4 in adj[a3]:
                            if a4 in (a0, a1, a2):
                                continue
                            for a5 in adj[a4]:
                                if a5 in (a0, a1, a2, a3):
                                    continue
                                if a0 in adj[a5]:
                                    return True
        return False

    def check_no_c6(color_assignment, n):
        """Check that no colour class contains a C6."""
        for c in range(4):
            edges_c = []
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if color_assignment[idx] == c:
                        edges_c.append((i, j))
                    idx += 1
            if has_c6_in_colour(edges_c, n):
                return False
        return True

    def backtrack(n, color_assignment, pos, time_limit):
        """Backtracking search for valid colouring of K_n."""
        if time.time() > time_limit:
            return None
        
        if pos == n * (n - 1) // 2:
            return color_assignment[:]
        
        # Determine which pair this position corresponds to
        # pos is the index in lexicographic order of pairs (i,j), i<j
        # Find (i, j) from pos
        i = 0
        remaining = pos
        while remaining >= (n - i - 1):
            remaining -= (n - i - 1)
            i += 1
        j = i + 1 + remaining
        
        for c in range(4):
            color_assignment[pos] = c
            
            # Quick check: does adding this edge create a C6 in colour c?
            # We need to check if there's now a C6 in colour c
            # For efficiency, only check if this edge could be part of a new C6
            # A new C6 must use this edge (i,j)
            # So we need a path of length 4 from i to j in colour c avoiding this edge
            valid = True
            # Check for C6 using edge (i,j)
            # Need vertices a,b,c,d such that i-a-b-c-d-j is a path in colour c
            # where all vertices are distinct and not i or j
            # Actually, the C6 is i-x-y-z-w-j-i, so we need a path of length 4 from i to j
            # through 4 intermediate vertices, all connected by colour c edges
            
            # Build colour-c adjacency (excluding the new edge for now, but it's included)
            adj_c = [set() for _ in range(n)]
            for k in range(pos):
                # Find pair for position k
                ii = 0
                rem = k
                while rem >= (n - ii - 1):
                    rem -= (n - ii - 1)
                    ii += 1
                jj = ii + 1 + rem
                cc = color_assignment[k]
                if cc == c:
                    adj_c[ii].add(jj)
                    adj_c[jj].add(ii)
            # Add current edge
            adj_c[i].add(j)
            adj_c[j].add(i)
            
            # Check if there's a path of length 4 from i to j in adj_c
            # i -> v1 -> v2 -> v3 -> v4 -> j
            # v1, v2, v3, v4 must be distinct and not i or j
            found_c6 = False
            for v1 in adj_c[i]:
                if v1 == j:
                    continue
                for v2 in adj_c[v1]:
                    if v2 == i or v2 == j or v2 == v1:
                        continue
                    for v3 in adj_c[v2]:
                        if v3 == i or v3 == j or v3 == v1 or v3 == v2:
                            continue
                        if v3 in adj_c:
                            for v4 in adj_c[v3]:
                                if v4 == i or v4 == j or v4 == v1 or v4 == v2 or v4 == v3:
                                    continue
                                if j in adj_c[v4]:
                                    found_c6 = True
                                    break
                        if found_c6:
                            break
                    if found_c6:
                        break
                if found_c6:
                    break
            if found_c6:
                valid = False
            
            if valid:
                result = backtrack(n, color_assignment, pos + 1, time_limit)
                if result is not None:
                    return result
        
        color_assignment[pos] = -1
        return None

    # Try increasing n from 10 upward
    best_result = k9_fallback()
    
    for n in range(10, 21):
        if time.time() > deadline - 10:
            break
        
        # For larger n, the search space is huge. Use a time-limited approach.
        # Try a few randomised restarts
        import random
        found = False
        max_restarts = 5 if n <= 12 else 2
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # For n > 12, pure backtracking is likely infeasible in 150s
            # Use a heuristic: try to extend the K9 construction or use random init
            if n <= 12:
                color_assignment = [-1] * (n * (n - 1) // 2)
                result = backtrack(n, color_assignment, 0, deadline - 5)
                if result is not None:
                    best_result = {'n': n, 'colors': result}
                    found = True
                    break
            else:
                # For larger n, use a different strategy
                # Try to find a valid colouring using a more structured approach
                # or accept that we can't reach these in time
                # Use random assignment with local repair
                break
        
        if found:
            continue
        else:
            # If we couldn't find for this n, try next but with less time
            # Actually, if we can't find n, we likely can't find n+1 either
            # but let's try one more level
            if n >= 12:
                break

    return best_result
# EVOLVE-BLOCK-END
