import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Total differences needed: n_rows * C(k+1, 2)
    # For n_rows=7, k=5: 7 * 15 = 105 differences
    # Scope must be at least 105 (counting bound)
    
    # We'll search for scope M, starting from lower bound
    # For each M, generate all valid candidate rows with max element <= M
    # Then find a clique of size n_rows in the compatibility graph
    
    min_scope = n_rows * (k * (k + 1) // 2)
    
    best_rows = None
    best_scope = float('inf')
    
    # Try increasing scopes
    for M in range(min_scope, min_scope + 50):  # Search up to 50 above lower bound
        if time.time() > deadline:
            break
        
        # Generate all valid candidate rows with max element exactly M
        # (or <= M, but we want to minimize max, so focus on rows with max = M)
        # A row is [0, a1, a2, ..., ak] with 0 < a1 < a2 < ... < ak = M
        # All C(k+1, 2) differences must be distinct within the row
        
        candidates = []
        used_diffs_in_row = set()
        
        # Generate candidate rows ending at M
        # We need k-1 intermediate values from {1, ..., M-1}
        # This is C(M-1, k-1) possibilities, which could be large
        # For M=105, k=5: C(104, 4) ~ 4.5M - too many
        
        # Better approach: use greedy/backtracking to generate a diverse set
        # of candidate rows, not all of them
        
        # Strategy: Generate candidates using randomized greedy with different seeds
        # Each candidate is a complete row [0, x1, x2, ..., xk] with max <= M
        
        num_candidates = min(5000, M * 50)  # Scale with M
        candidates = []
        
        for attempt in range(num_candidates):
            if time.time() > deadline:
                break
            
            # Generate a random valid row with max element <= M
            row = [0]
            current_max = 0
            row_diffs = set()
            
            for pos in range(k):
                # Choose next element
                # Must be > current_max, <= M
                # And all new differences must not be in row_diffs
                
                found = False
                # Try random values in (current_max, M]
                # Start with values closer to M to maximize spread
                candidates_x = list(range(M, current_max, -1))
                random.shuffle(candidates_x)
                
                for x in candidates_x:
                    # Check if x - 0 = x is already used
                    if x in row_diffs:
                        continue
                    
                    # Check all differences x - a for a in row
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d == 0:
                            continue
                        if d in row_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        row.append(x)
                        row_diffs |= new_diffs
                        current_max = x
                        found = True
                        break
                
                if not found:
                    break
            
            if len(row) == k + 1 and row[-1] <= M:
                # This is a valid candidate
                # Store the row and its difference set
                diff_set = frozenset(row_diffs)
                candidates.append((row, diff_set))
        
        if len(candidates) < n_rows:
            continue
        
        # Now find a clique of size n_rows in the compatibility graph
        # Two rows are compatible if their difference sets are disjoint
        
        # Use branch and bound for maximum clique
        # Nodes: candidate indices
        # Edge: if diff_sets are disjoint
        
        n = len(candidates)
        
        # Precompute compatibility matrix (bitsets for speed)
        # For n up to 5000, this is 5000*5000/2 = 12.5M pairs
        # Each check is set intersection, which is O(min(|A|, |B|))
        # This might be too slow. Let's limit candidates.
        
        if n > 2000:
            # Too many candidates, sample
            candidates = random.sample(candidates, 2000)
            n = 2000
        
        # Build adjacency using bitsets
        # adj[i] is a set of j such that i and j are compatible
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            if time.time() > deadline:
                break
            di = candidates[i][1]
            for j in range(i + 1, n):
                dj = candidates[j][1]
                # Check if disjoint
                if not (di & dj):
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Find maximum clique of size n_rows
        # Use simple backtracking with ordering
        
        def find_clique(target_size):
            """Find a clique of size target_size using backtracking"""
            # Order vertices by degree (descending) for better pruning
            order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
            
            # Map original indices to ordered positions
            pos_in_order = {v: i for i, v in enumerate(order)}
            
            # Rebuild adjacency in ordered form
            adj_ordered = [set() for _ in range(n)]
            for i in range(n):
                for j in adj[order[i]]:
                    adj_ordered[i].add(pos_in_order[j])
            
            def backtrack(candidates_list, clique):
                if len(clique) == target_size:
                    return clique
                if not candidates_list:
                    return None
                if len(candidates_list) + len(clique) < target_size:
                    return None
                
                # Take the first candidate
                v = candidates_list[0]
                
                # Option 1: Include v
                new_candidates = [u for u in candidates_list[1:] if u in adj_ordered[v]]
                result = backtrack(new_candidates, clique + [v])
                if result is not None:
                    return result
                
                # Option 2: Exclude v
                return backtrack(candidates_list[1:], clique)
            
            return backtrack(list(range(n)), [])
        
        clique = find_clique(n_rows)
        
        if clique is not None:
            # Extract the rows
            rows = [candidates[v][0] for v in clique]
            scope = max(max(r) for r in rows)
            
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                
                # Verify validity
                all_diffs = set()
                valid = True
                for row in rows:
                    row_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in all_diffs or d in row_diffs:
                                valid = False
                                break
                            row_diffs.add(d)
                    if not valid:
                        break
                    all_diffs |= row_diffs
                
                if valid:
                    if best_scope <= min_scope:
                        break  # Hit lower bound, optimal
                    # Don't break otherwise, try to improve
                else:
                    # Shouldn't happen, but skip
                    pass
        
        # If we found a valid solution, we could stop early if scope is close to optimal
        if best_rows is not None and best_scope <= min_scope + 10:
            # Good enough, but keep searching if time allows
            pass
    
    if best_rows is None:
        # Fallback: simple greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    return best_rows
