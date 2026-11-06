import numpy as np
import random
import time

def construct():
    start_time = time.time()
    max_time = 85.0
    n = 512
    max_size = 48

    def is_valid_set(S):
        """Check if no element is the mean of any subset of >=2 other elements."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        if m < 3:
            return True
        
        # For each element x in S, check if x = mean of some subset T of S\{x}, |T|>=2
        # x = sum(T)/|T| => sum(T) = x * |T|
        # For each possible subset size k >= 2, check if there's a subset of S\{x} of size k with sum = x*k
        
        # Use DP for each x (or batch)
        # For efficiency, precompute subset sums for each cardinality
        # dp[k][s] = can we pick k elements from S\{x} with sum s
        
        # Actually, let's do it per-element but with early termination
        for i, x in enumerate(S_sorted):
            # S without x
            others = [y for j, y in enumerate(S_sorted) if j != i]
            max_k = len(others)
            if max_k < 2:
                continue
            
            # DP: dp[k][s] = True if we can pick k elements from others with sum s
            # Max sum = max_k * n
            max_sum = max_k * n
            dp = [set() for _ in range(max_k + 1)]
            dp[0].add(0)
            
            for y in others:
                for k in range(len(dp) - 1, 0, -1):
                    for s in list(dp[k - 1]):
                        new_s = s + y
                        if new_s <= max_sum:
                            dp[k].add(new_s)
            
            # Check: for k >= 2, is x*k in dp[k]?
            for k in range(2, max_k + 1):
                if x * k in dp[k]:
                    return False
        
        return True

    def build_conflict_graph(candidates):
        """Build a conflict graph where edge (i,j) means i and j conflict.
        Two elements conflict if adding one creates a mean violation with the other.
        
        Actually, for the independent set approach, we need:
        - Vertex: each candidate element
        - Edge: two elements are incompatible (can't both be in a valid set)
        
        The simplest conflicts are 3-term APs: if a, b, c form an AP (a+c=2b),
        then b is the mean of {a,c}, so b can't coexist with both a and c.
        
        But more generally, any subset mean creates conflicts.
        
        For a practical approach, let's define:
        - Two elements x, y conflict if there exists a third element z in candidates
          such that one of them is the mean of a subset containing the other.
        
        This is complex. Instead, let's use a simpler heuristic:
        - Build a graph where edges represent 3-term AP conflicts
        - Then verify the final set with full DP
        
        Actually, let me think differently. The constraint is:
        No element is the mean of any subset of >=2 OTHER elements.
        
        So if we have elements {a, b, c} with a + c = 2b, then b is the mean of {a,c}.
        This means in any valid set, we cannot have all three of a, b, c.
        
        More generally, for any subset T of size k>=2, if mean(T) is also in the set, that's invalid.
        
        For the conflict graph approach, let's say:
        - Two elements x, y are "in conflict" if they form a 2-term "mean" situation with some third element
        - Specifically, if x = (y + z)/2 for some z, then x conflicts with both y and z
          (because {y,z} has mean x)
        
        So the conflict graph has an edge between x and y if:
        - 2x = y + z for some z in candidates (x is mean of y,z)
        - 2y = x + z for some z in candidates (y is mean of x,z)
        - More generally, but let's start with 3-term APs
        
        Let me build the graph based on 3-term APs:
        For each pair (i,j), check if there exists k such that one is the mean of the other two.
        """
        m = len(candidates)
        # conflict[i] = set of indices j such that i and j conflict
        conflict = [set() for _ in range(m)]
        
        # For 3-term APs: a, b, c with a+c=2b
        # b is the mean of {a,c}
        # So b conflicts with a and c (in the sense that you can't have all three)
        # But actually, b conflicts with the PAIR {a,c}, not individually.
        # 
        # For a proper conflict graph for independent set, we need:
        # If selecting both i and j forces a violation, then edge (i,j).
        # 
        # This is actually a hypergraph problem, not a graph.
        # For a 3-term AP a,b,c: you can't select all three, but you can select any two.
        # So it's a 3-uniform hyperedge, not a graph edge.
        #
        # Let me use a different approach: greedy construction with backtracking,
        # but guided by avoiding 3-term APs first.
        
        # Alternative: use the fact that the primary constraint is 3-term APs.
        # Build a graph where edge (i,j) exists if i and j are part of a 3-term AP
        # with some third element. Then find a large independent set in this graph.
        # An independent set in this graph has no 3-term AP (as a necessary condition).
        # Then verify with full DP.
        
        # Edge (i,j) if there exists k != i,j such that:
        # - candidates[i] + candidates[j] = 2*candidates[k] (k is mean of i,j)
        # - candidates[i] + candidates[k] = 2*candidates[j] (j is mean of i,k)  
        # - candidates[j] + candidates[k] = 2*candidates[i] (i is mean of j,k)
        
        # For efficiency, for each pair (i,j), check if (i+j)/2 is in candidates
        # or if (2*i - j) is in candidates or (2*j - i) is in candidates
        
        cand_set = set(candidates)
        for i in range(m):
            for j in range(i + 1, m):
                ci, cj = candidates[i], candidates[j]
                # Check if there's a third element that forms a 3-term AP
                # Case 1: k = (ci + cj) / 2, need ci + cj even and k in cand_set, k != ci, k != cj
                if (ci + cj) % 2 == 0:
                    k = (ci + cj) // 2
                    if k in cand_set and k != ci and k != cj:
                        conflict[i].add(j)
                        conflict[j].add(i)
                        continue
                
                # Case 2: i is the mean: 2*ci = cj + k => k = 2*ci - cj
                k = 2 * ci - cj
                if k in cand_set and k != ci and k != cj:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                
                # Case 3: j is the mean: 2*cj = ci + k => k = 2*cj - ci
                k = 2 * cj - ci
                if k in cand_set and k != ci and k != cj:
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        return conflict

    def greedy_independent_set(conflict, m, random_order=None):
        """Greedy algorithm to find a large independent set."""
        if random_order is None:
            random_order = list(range(m))
            random.shuffle(random_order)
        
        selected = set()
        # Sort by degree (ascending) for better results
        degrees = {i: len(conflict[i]) for i in range(m)}
        sorted_nodes = sorted(random_order, key=lambda x: degrees[x])
        
        for node in sorted_nodes:
            # Check if node is adjacent to any selected node
            if not (conflict[node] & selected):
                selected.add(node)
        
        return selected

    def local_search_improve(selected, conflict, m, max_iters=1000):
        """Try to improve the independent set via local search."""
        current = set(selected)
        current_size = len(current)
        
        for _ in range(max_iters):
            improved = False
            
            # Try adding a node
            candidates_to_add = [i for i in range(m) if i not in current and not (conflict[i] & current)]
            if candidates_to_add:
                # Pick one that minimizes conflicts with non-selected nodes
                best_add = min(candidates_to_add, key=lambda i: len(conflict[i] - current))
                current.add(best_add)
                improved = True
            
            # Try swapping: remove one node, add another
            if not improved:
                # For each node in current, try removing it and adding a better one
                for remove_node in list(current):
                    current.remove(remove_node)
                    # Find best node to add
                    candidates_to_add = [i for i in range(m) if i not in current and not (conflict[i] & current)]
                    if candidates_to_add:
                        best_add = min(candidates_to_add, key=lambda i: len(conflict[i] - current))
                        current.add(best_add)
                        if len(current) > current_size:
                            current_size = len(current)
                            improved = True
                            break
                        else:
                            # Revert
                            current.remove(best_add)
                            current.add(remove_node)
                
                if improved:
                    continue
                
                # No improvement, try random perturbation
                if len(current) > 0:
                    # Remove a random node and try to re-add
                    remove_node = random.choice(list(current))
                    current.remove(remove_node)
                    candidates_to_add = [i for i in range(m) if i not in current and not (conflict[i] & current)]
                    if candidates_to_add:
                        best_add = random.choice(candidates_to_add)
                        current.add(best_add)
            
            if not improved:
                break
        
        return current

    best_set = []
    best_size = 0

    # Strategy: 
    # 1. Use greedy construction guided by avoiding 3-term APs
    # 2. Use local search to improve
    # 3. Verify with full DP
    # 4. Multiple restarts with different orderings

    for restart in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Random candidate ordering
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # Build conflict graph
        conflict = build_conflict_graph(candidates)
        m = len(candidates)
        
        # Greedy independent set
        selected = greedy_independent_set(conflict, m)
        
        # Local search improvement
        selected = local_search_improve(selected, conflict, m, max_iters=500)
        
        # Convert back to actual values
        S = [candidates[i] for i in selected]
        
        # Verify with full DP
        if is_valid_set(S):
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        # Also try greedy construction directly
        # Greedy: add elements in order, check validity
        S_greedy = []
        # Use a faster validity check for incremental construction
        # For each new element x, check if x is mean of any subset of S_greedy (size >= 2)
        # and if any element in S_greedy is mean of any subset of S_greedy + {x}
        
        # Precompute subset sums for S_greedy as we go
        max_sum_so_far = 0
        dp = [set() for _ in range(max_size + 1)]
        dp[0].add(0)
        current_max_k = 0
        
        for x in candidates:
            if time.time() - start_time > max_time:
                break
            
            valid = True
            
            # Check 1: x is not mean of any subset of S_greedy (size >= 2)
            for k in range(2, current_max_k + 1):
                if x * k in dp[k]:
                    valid = False
                    break
            
            # Check 2: No element y in S_greedy is mean of any subset of S_greedy + {x}
            # Subset must include x (otherwise it would have been caught by check 1 for y)
            # sum(T) + x = (k+1) * y => sum(T) = (k+1)*y - x, where T is subset of S_greedy, |T| = k >= 1
            if valid:
                for y in S_greedy:
                    for k in range(1, current_max_k + 1):
                        target = (k + 1) * y - x
                        if target >= 0 and target in dp[k]:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Update DP
                for k in range(current_max_k, -1, -1):
                    for s in list(dp[k]):
                        new_s = s + x
                        if new_s <= max_size * n:
                            dp[k + 1].add(new_s)
                
                S_greedy.append(x)
                current_max_k += 1
                
                if current_max_k == max_size:
                    break
        
        if len(S_greedy) > best_size and is_valid_set(S_greedy):
            best_size = len(S_greedy)
            best_set = S_greedy[:]

    return best_set
